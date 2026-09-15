"""Disease-priority gradient surgery with unscaled fp32 gradients."""

from __future__ import annotations

import torch
import torch.distributed as dist



def _gradients(loss, parameters, retain_graph):
    return tuple(
        None if gradient is None else gradient.float()
        for gradient in torch.autograd.grad(
            loss, parameters, retain_graph=retain_graph, allow_unused=True
        )
    )


def _dot(first, second, selected_ids, parameters):
    terms = [
        (left * right).sum()
        for parameter, left, right in zip(parameters, first, second)
        if id(parameter) in selected_ids and left is not None and right is not None
    ]
    if terms:
        return sum(terms)
    return next(
        (gradient.new_zeros(()) for gradient in (*first, *second) if gradient is not None),
        torch.tensor(0.0),
    )


def _norm(gradients, selected_ids, parameters):
    return _dot(gradients, gradients, selected_ids, parameters).clamp_min(0.0).sqrt()


def _finite(gradients):
    return all(
        gradient is None or bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
    )


def _project_against_primary(primary, auxiliary, overlap_ids, parameters):
    dot = _dot(primary, auxiliary, overlap_ids, parameters)
    primary_sq = _dot(primary, primary, overlap_ids, parameters).clamp_min(1e-30)
    conflict = bool((dot < 0).item())
    coefficient = dot / primary_sq if conflict else dot.new_zeros(())
    adjusted = tuple(
        (
            aux - coefficient.to(dtype=aux.dtype) * main
            if conflict
            and id(parameter) in overlap_ids
            and main is not None
            and aux is not None
            else aux
        )
        for parameter, main, aux in zip(parameters, primary, auxiliary)
    )
    primary_norm = _norm(primary, overlap_ids, parameters)
    auxiliary_norm = _norm(auxiliary, overlap_ids, parameters)
    adjusted_dot = _dot(primary, adjusted, overlap_ids, parameters)
    adjusted_norm = _norm(adjusted, overlap_ids, parameters)
    return adjusted, {
        "cosine_before": dot / (primary_norm * auxiliary_norm).clamp_min(1e-30),
        "cosine_after": adjusted_dot / (primary_norm * adjusted_norm).clamp_min(1e-30),
        "conflict": dot.new_tensor(float(conflict)),
    }


def _project_against_primary_by_groups(primary, auxiliary, overlap_groups, parameters):
    all_ids = set().union(*overlap_groups) if overlap_groups else set()
    before_dot = _dot(primary, auxiliary, all_ids, parameters)
    primary_norm = _norm(primary, all_ids, parameters)
    auxiliary_norm = _norm(auxiliary, all_ids, parameters)
    adjusted = auxiliary
    conflict = False
    for group_ids in overlap_groups:
        adjusted, group = _project_against_primary(
            primary, adjusted, group_ids, parameters
        )
        conflict = conflict or bool(group["conflict"].item())
    after_dot = _dot(primary, adjusted, all_ids, parameters)
    adjusted_norm = _norm(adjusted, all_ids, parameters)
    return adjusted, {
        "cosine_before": before_dot / (primary_norm * auxiliary_norm).clamp_min(1e-30),
        "cosine_after": after_dot / (primary_norm * adjusted_norm).clamp_min(1e-30),
        "conflict": before_dot.new_tensor(float(conflict)),
    }


def _synchronize_component(gradients):
    """All-reduce one objective before any nonlinear gradient surgery."""
    if not (dist.is_available() and dist.is_initialized()):
        return gradients
    present = [gradient for gradient in gradients if gradient is not None]
    if not present:
        return gradients
    flat = torch._utils._flatten_dense_tensors(present)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(dist.get_world_size())
    synchronized = iter(torch._utils._unflatten_dense_tensors(flat, present))
    return tuple(None if gradient is None else next(synchronized) for gradient in gradients)


def _clip_component(gradients, max_norm):
    if max_norm <= 0:
        return gradients, 1.0
    squared = sum(
        gradient.square().sum() for gradient in gradients if gradient is not None
    )
    if isinstance(squared, int):
        return gradients, 1.0
    norm = squared.clamp_min(0.0).sqrt()
    scale = min(1.0, float(max_norm / norm.clamp_min(1e-30)))
    return tuple(
        None if gradient is None else gradient * scale for gradient in gradients
    ), scale


def disease_priority_multistate_step(
    disease_loss,
    weighted_attribute_loss,
    weighted_contrastive_loss,
    weighted_row_loss,
    weighted_relation_loss,
    optimizer,
    disease_parameters,
    budget_parameter_groups,
    auxiliary_update_budget_ratio=1.0,
    max_component_grad_norm=0.0,
    decorrelation_active=True,
    contrastive_candidate_scale=1.0,
    protect_decorrelation_budget=False,
    independent_contrastive_budget_ratio=None,
):
    """Five-gradient, five-state disease-priority update with one parameter write."""
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    disease_ids = {id(parameter) for parameter in disease_parameters}
    disease_indices = {
        parameter_indices[parameter_id]
        for parameter_id in disease_ids
        if parameter_id in parameter_indices
    }
    budget_groups = {
        name: {
            parameter_indices[id(parameter)]
            for parameter in group_parameters
            if id(parameter) in parameter_indices
        }
        for name, group_parameters in budget_parameter_groups.items()
    }
    budget_id_groups = [
        {id(parameters[index]) for index in indices}
        for indices in budget_groups.values()
    ]

    disease_grads = _gradients(disease_loss, parameters, retain_graph=True)
    attribute_grads = _gradients(
        weighted_attribute_loss, parameters, retain_graph=True
    )
    contrastive_grads = _gradients(
        weighted_contrastive_loss, parameters, retain_graph=decorrelation_active
    )
    if decorrelation_active:
        row_grads = _gradients(weighted_row_loss, parameters, retain_graph=True)
        relation_grads = _gradients(weighted_relation_loss, parameters, retain_graph=False)
    else:
        row_grads = tuple(None for _ in parameters)
        relation_grads = tuple(None for _ in parameters)
    components = {
        "disease": disease_grads,
        "attribute": attribute_grads,
        "contrastive": contrastive_grads,
        "row": row_grads,
        "relation": relation_grads,
    }
    # Projection is nonlinear, so DDP must reduce each component first.
    components = {
        name: _synchronize_component(gradients)
        for name, gradients in components.items()
    }
    invalid = [name for name, gradients in components.items() if not _finite(gradients)]
    if invalid:
        raise FloatingPointError(f"Non-finite unscaled fp32 gradients: {invalid}")

    raw_components = components.copy()
    adjusted, projection = {}, {}
    for name in ("attribute", "contrastive", "row", "relation"):
        adjusted[name], projection[name] = _project_against_primary_by_groups(
            components["disease"], components[name], budget_id_groups, parameters
        )
    adjusted["disease"] = components["disease"]
    clip_scales = {}
    for name in ("disease", "attribute", "contrastive", "row", "relation"):
        adjusted[name], clip_scales[name] = _clip_component(
            adjusted[name], max_component_grad_norm
        )
    if not all(_finite(gradients) for gradients in adjusted.values()):
        raise FloatingPointError("Projection or clipping produced non-finite fp32 gradients")

    step_diagnostics = optimizer.step_components(
        adjusted,
        adjusted["disease"],
        disease_indices,
        budget_groups,
        auxiliary_update_budget_ratio,
        candidate_scales={"contrastive": contrastive_candidate_scale},
        protect_decorrelation_budget=protect_decorrelation_budget,
        independent_contrastive_budget_ratio=independent_contrastive_budget_ratio,
    )
    all_ids = {id(parameter) for parameter in parameters}
    norms = {
        name: _norm(gradients, all_ids, parameters)
        for name, gradients in raw_components.items()
    }
    norm_sum = sum(norms.values()).clamp_min(1e-30)
    zero = disease_loss.detach().new_zeros(())
    result = {
        "disease_gradient_norm": norms["disease"],
        "attribute_gradient_norm": norms["attribute"],
        "contrastive_gradient_norm": norms["contrastive"],
        "row_gradient_norm": norms["row"],
        "relation_gradient_norm": norms["relation"],
        "disease_gradient_share": norms["disease"] / norm_sum,
        "attribute_gradient_share": norms["attribute"] / norm_sum,
        "contrastive_gradient_share": norms["contrastive"] / norm_sum,
        "row_gradient_share": norms["row"] / norm_sum,
        "relation_gradient_share": norms["relation"] / norm_sum,
        "all_gradients_finite": zero.new_ones(()),
    }
    for name in ("disease", "attribute", "contrastive", "row", "relation"):
        result[f"{name}_gradient_clip_scale"] = zero.new_tensor(clip_scales[name])
    for name in ("attribute", "contrastive", "row", "relation"):
        result[f"{name}_cosine_before"] = projection[name]["cosine_before"]
        result[f"{name}_cosine_after"] = projection[name]["cosine_after"]
        result[f"{name}_gradient_conflict"] = projection[name]["conflict"]
    result.update(step_diagnostics)
    return result
