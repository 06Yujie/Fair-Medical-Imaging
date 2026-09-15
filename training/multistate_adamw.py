"""Five-state AdamW with candidate displacement composition and one write."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer



STREAMS = ("disease", "attribute", "contrastive", "row", "relation")
AUXILIARY_STREAMS = tuple(name for name in STREAMS if name != "disease")


def _dot(first, second, indices):
    terms = [
        (first[index] * second[index]).sum()
        for index in indices
        if first[index] is not None and second[index] is not None
    ]
    if terms:
        return sum(terms)
    return next(
        (value.new_zeros(()) for value in (*first, *second) if value is not None),
        torch.tensor(0.0),
    )


def _norm(values, indices):
    return _dot(values, values, indices).clamp_min(0.0).sqrt()


def _add(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _project_displacement(disease_gradients, displacement, indices):
    dot = _dot(disease_gradients, displacement, indices)
    disease_sq = _dot(disease_gradients, disease_gradients, indices).clamp_min(1e-30)
    # A displacement harms disease to first order when g_d^T delta > 0.
    harmful = bool((dot > 0).item())
    coefficient = dot / disease_sq if harmful else dot.new_zeros(())
    projected = tuple(
        (
            value - coefficient.to(value.dtype) * disease_gradients[index]
            if harmful and index in indices and value is not None
            and disease_gradients[index] is not None
            else value
        )
        for index, value in enumerate(displacement)
    )
    after = _dot(disease_gradients, projected, indices)
    return projected, dot, after, harmful


def _enforce_final_disease_descent(disease_gradients, deltas, indices):
    """Put the complete update in the strict disease-descent half-space.

    This is deliberately performed after auxiliary composition and decoupled
    weight decay, but before any parameter write.  A small negative margin is
    used because an ordinary projection of a harmful update only guarantees a
    zero first-order derivative, not descent.
    """
    dot_before = _dot(disease_gradients, deltas, indices)
    disease_sq = _dot(disease_gradients, disease_gradients, indices).clamp_min(0.0)
    disease_norm = disease_sq.sqrt()
    delta_norm = _norm(deltas, indices)
    has_disease_direction = bool((disease_sq > 0).item())
    applied = bool((dot_before >= 0).item()) and has_disease_direction
    projected = deltas
    if applied:
        # Scale-aware strict margin; the disease_sq term also covers delta == 0.
        margin = 1e-6 * (disease_norm * delta_norm + disease_sq)
        coefficient = (dot_before + margin) / disease_sq
        projected = tuple(
            (
                value - coefficient.to(value.dtype) * disease_gradients[index]
                if index in indices and value is not None
                and disease_gradients[index] is not None
                else value
            )
            for index, value in enumerate(projected)
        )
    dot_after = _dot(disease_gradients, projected, indices)
    if has_disease_direction and bool((dot_after >= 0).item()):
        # One round-off correction makes the pre-write invariant explicit.
        margin = 1e-6 * (disease_norm * _norm(projected, indices) + disease_sq)
        coefficient = (dot_after + margin) / disease_sq
        projected = tuple(
            (
                value - coefficient.to(value.dtype) * disease_gradients[index]
                if index in indices and value is not None
                and disease_gradients[index] is not None
                else value
            )
            for index, value in enumerate(projected)
        )
        dot_after = _dot(disease_gradients, projected, indices)
    if has_disease_direction and not bool((dot_after < 0).item()):
        raise FloatingPointError(
            "Unable to construct a strict disease-descent update before parameter write"
        )
    return projected, dot_before, dot_after, applied, has_disease_direction


class MultiStateDiseasePriorityAdamW(Optimizer):
    """Independent Adam states for disease and all four auxiliary objectives.

    ``step_components`` computes five candidate displacements without touching
    parameters, projects auxiliary displacements against the disease gradient,
    applies separate backbone/projector budgets, then writes each parameter
    exactly once. Decoupled weight decay is also applied exactly once.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps, and weight_decay must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0,1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self, closure=None):
        raise RuntimeError("Use step_components() so independent Adam states cannot be mixed.")

    def _candidate(self, parameter, gradient, group, stream):
        if gradient is None:
            return None
        state = self.state[parameter]
        prefix = f"{stream}_"
        step_key, mean_key, square_key = prefix + "step", prefix + "exp_avg", prefix + "exp_avg_sq"
        if mean_key not in state:
            state[step_key] = 0
            state[mean_key] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state[square_key] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        state[step_key] += 1
        step = state[step_key]
        beta1, beta2 = group["betas"]
        mean, square = state[mean_key], state[square_key]
        mean.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        square.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        denominator = square.sqrt().div_(math.sqrt(bias2)).add_(group["eps"])
        return mean.div(denominator).mul_(-group["lr"] / bias1)

    @torch.no_grad()
    def step_components(
        self,
        component_gradients,
        disease_gradients,
        disease_indices,
        budget_groups,
        auxiliary_budget_ratio,
        candidate_scales=None,
        protect_decorrelation_budget=False,
        independent_contrastive_budget_ratio=None,
    ):
        if independent_contrastive_budget_ratio is not None:
            if not protect_decorrelation_budget:
                raise ValueError("Independent contrastive budget requires protect_decorrelation_budget")
            if (not math.isfinite(independent_contrastive_budget_ratio)
                    or independent_contrastive_budget_ratio < 0):
                raise ValueError("Independent contrastive budget must be finite and nonnegative")
        # Keep lr_scheduler bookkeeping consistent with a regular optimizer step.
        self._opt_called = True
        parameters, groups = [], []
        for group in self.param_groups:
            for parameter in group["params"]:
                parameters.append(parameter)
                groups.append(group)
        if any(len(component_gradients[name]) != len(parameters) for name in STREAMS):
            raise ValueError("Component gradient tuples do not match optimizer parameters")

        candidate_scales = {} if candidate_scales is None else candidate_scales
        unknown_scales = set(candidate_scales) - set(AUXILIARY_STREAMS)
        if unknown_scales:
            raise ValueError(f"Unknown candidate scales: {sorted(unknown_scales)}")
        candidates = {name: [] for name in STREAMS}
        for index, (parameter, group) in enumerate(zip(parameters, groups)):
            for name in STREAMS:
                candidates[name].append(
                    self._candidate(parameter, component_gradients[name][index], group, name)
                )
        candidates = {name: tuple(values) for name, values in candidates.items()}

        # Scale Adam candidate displacements rather than relying only on a loss
        # coefficient: Adam is approximately invariant to gradient magnitude.
        for name in AUXILIARY_STREAMS:
            scale = float(candidate_scales.get(name, 1.0))
            if scale < 0:
                raise ValueError("Candidate scales must be non-negative")
            candidates[name] = tuple(
                None if value is None else value * (
                    groups[index].get("contrastive_candidate_scale", scale)
                    if name == "contrastive" else scale)
                for index, value in enumerate(candidates[name])
            )

        displacement_projection = {}
        for name in AUXILIARY_STREAMS:
            projected = candidates[name]
            before = _dot(disease_gradients, projected, disease_indices)
            harmful = False
            for indices in budget_groups.values():
                projected, _, _, group_harmful = _project_displacement(
                    disease_gradients, projected, indices
                )
                harmful = harmful or group_harmful
            after = _dot(disease_gradients, projected, disease_indices)
            candidates[name] = projected
            displacement_projection[name] = {
                "dot_before": before,
                "dot_after": after,
                "harmful": before.new_tensor(float(harmful)),
            }

        group_scales = {}
        contrastive_group_scales = {}
        auxiliary = tuple(
            sum(
                (candidates[stream][index] for stream in AUXILIARY_STREAMS
                 if candidates[stream][index] is not None),
                start=torch.zeros_like(parameters[index]),
            )
            for index in range(len(parameters))
        )
        for name, indices in budget_groups.items():
            disease_norm = _norm(candidates["disease"], indices)
            if protect_decorrelation_budget:
                # Reserve the existing non-contrastive directions first. The alignment update
                # can only use the remaining norm budget, never rescale them.
                protected = tuple(sum((candidates[stream][i]
                    for stream in ("attribute", "row", "relation")
                    if candidates[stream][i] is not None), start=torch.zeros_like(parameters[i]))
                    for i in range(len(parameters)))
                auxiliary_norm = _norm(protected, indices)
            else:
                auxiliary_norm = _norm(auxiliary, indices)
            scale = min(
                1.0,
                float(
                    float(auxiliary_budget_ratio)
                    * disease_norm
                    / auxiliary_norm.clamp_min(1e-30)
                ),
            )
            group_scales[name] = scale
            streams_to_scale = ("attribute", "row", "relation") if protect_decorrelation_budget else AUXILIARY_STREAMS
            for stream in streams_to_scale:
                values = list(candidates[stream])
                for index in indices:
                    if values[index] is not None:
                        values[index] = values[index] * scale
                candidates[stream] = tuple(values)
            contrastive_scale = scale
            if protect_decorrelation_budget:
                base = tuple(v * scale if i in indices else v for i, v in enumerate(protected))
                contrastive = candidates["contrastive"]
                if independent_contrastive_budget_ratio is not None:
                    # Separate cap, measured against this module's disease
                    # candidate. Never scale up a weak/absent prototype-guided alignment candidate
                    # and never take budget away from the protected streams.
                    contrastive_norm = _norm(contrastive, indices)
                    contrastive_scale = min(1.0, float(
                        float(independent_contrastive_budget_ratio) * disease_norm
                        / contrastive_norm.clamp_min(1e-30)))
                else:
                    radius = float(auxiliary_budget_ratio) * disease_norm
                    a = _dot(contrastive, contrastive, indices).clamp_min(0.0)
                    b = _dot(base, contrastive, indices)
                    slack = (radius.square() - _dot(base, base, indices)).clamp_min(0.0)
                    # Positive root of ||base + t*contrastive||^2 <= radius^2.
                    contrastive_scale = (min(1.0, max(0.0, float(
                        (-b + (b.square() + a * slack).sqrt()) / a)))
                        if float(a) > 0 else 1.0)
                candidates["contrastive"] = tuple(
                    v * contrastive_scale if i in indices and v is not None else v
                    for i, v in enumerate(contrastive))
            contrastive_group_scales[name] = contrastive_scale

        deltas = []
        decay_deltas = []
        for index, (parameter, group) in enumerate(zip(parameters, groups)):
            delta = candidates["disease"][index]
            for stream in AUXILIARY_STREAMS:
                delta = _add(delta, candidates[stream][index])
            has_gradient = any(
                component_gradients[stream][index] is not None for stream in STREAMS
            )
            decay = None
            if has_gradient and group["weight_decay"]:
                decay = parameter * (-group["lr"] * group["weight_decay"])
                delta = _add(delta, decay)
            deltas.append(delta)
            decay_deltas.append(decay)

        deltas = tuple(deltas)
        if not all(value is None or bool(torch.isfinite(value).all().item()) for value in deltas):
            raise FloatingPointError("Non-finite final displacement before parameter write")
        (
            deltas,
            dot_before_safety,
            actual_dot,
            safety_applied,
            has_disease_direction,
        ) = _enforce_final_disease_descent(disease_gradients, deltas, disease_indices)
        for parameter, delta in zip(parameters, deltas):
            if delta is not None:
                parameter.add_(delta)

        zero = actual_dot.new_zeros(())
        result = {
            "disease_dot_before_safety_projection": dot_before_safety,
            "disease_dot_actual_delta": actual_dot,
            "disease_actual_descent": zero.new_tensor(float(actual_dot < 0)),
            "disease_safety_projection_applied": zero.new_tensor(float(safety_applied)),
            "disease_gradient_nonzero": zero.new_tensor(float(has_disease_direction)),
            "actual_delta_norm": _norm(deltas, set(range(len(deltas)))),
        }
        for name, scale in group_scales.items():
            result[f"{name}_auxiliary_budget_scale"] = zero.new_tensor(scale)
        for name, scale in contrastive_group_scales.items():
            result[f"{name}_contrastive_budget_scale"] = zero.new_tensor(scale)
        for name, indices in budget_groups.items():
            reference_norm = _norm(candidates["disease"], indices).clamp_min(1e-30)
            reserved = tuple(sum((candidates[stream][i]
                for stream in ("attribute", "row", "relation")
                if candidates[stream][i] is not None), start=torch.zeros_like(parameters[i]))
                for i in range(len(parameters)))
            combined = tuple(_add(r, c) for r, c in zip(reserved, candidates["contrastive"]))
            result[f"{name}_contrastive_disease_norm_ratio"] = _norm(candidates["contrastive"], indices) / reference_norm
            result[f"{name}_reserved_auxiliary_disease_norm_ratio"] = _norm(reserved, indices) / reference_norm
            result[f"{name}_combined_auxiliary_disease_norm_ratio"] = _norm(combined, indices) / reference_norm
        for name in ("row", "relation"):
            result[f"{name}_budgeted_delta_norm"] = _norm(candidates[name], disease_indices)
        for name, values in displacement_projection.items():
            result[f"{name}_displacement_dot_before"] = values["dot_before"]
            result[f"{name}_displacement_dot_after"] = values["dot_after"]
            result[f"{name}_displacement_harmful"] = values["harmful"]
        return result
