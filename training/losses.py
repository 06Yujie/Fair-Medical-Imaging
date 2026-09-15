"""Classification, prototype-guided alignment, and decorrelation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _connected_zero(*values: torch.Tensor) -> torch.Tensor:
    return sum((value.sum() for value in values), start=values[0].new_zeros(())) * 0.0


def attribute_distance_positive_weights(
    disease_labels: torch.Tensor,
    attributes: torch.Tensor,
    alpha: float = 1.0,
    weighting: str = "difference_count",
) -> torch.Tensor:
    """Same-disease positive weights based on demographic differences."""
    labels = disease_labels.reshape(-1).long()
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    if weighting == "uniform":
        return positive_mask.float()
    if weighting != "difference_count":
        raise ValueError(f"Unknown SupCon positive weighting: {weighting!r}")
    differences = attributes[:, None, :].ne(attributes[None, :, :]).sum(dim=2)
    return positive_mask.float() * (1.0 + alpha * differences.float())


def attribute_aware_cross_group_supcon_loss(
    features: torch.Tensor,
    disease_labels: torch.Tensor,
    attributes: torch.Tensor,
    tau: float = 0.1,
    alpha: float = 1.0,
    positive_weighting: str | None = None,
) -> torch.Tensor:
    """Cross-group supervised contrastive baseline without prototype weighting."""
    if features.ndim != 2 or attributes.ndim != 2:
        raise ValueError("Expected features [B,D] and attributes [B,A].")
    if tau <= 0 or alpha < 0:
        raise ValueError("tau must be positive and alpha non-negative.")
    if features.shape[0] != attributes.shape[0]:
        raise ValueError("Batch dimensions do not match.")
    if features.shape[0] < 2:
        return _connected_zero(features)
    labels = disease_labels.reshape(-1).long()
    normalized = F.normalize(features.float(), dim=1)
    logits = normalized @ normalized.T / tau
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    if positive_weighting is None:
        cross_group = attributes[:, None, :].ne(attributes[None, :, :]).any(dim=2)
        positive_weights = positive_mask.to(logits.dtype) * (
            1.0 + alpha * cross_group.to(logits.dtype)
        )
    else:
        positive_weights = attribute_distance_positive_weights(
            labels, attributes, alpha, positive_weighting
        ).to(logits.dtype)
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_sum = positive_weights.sum(dim=1)
    valid = positive_sum > 0
    if not valid.any():
        return _connected_zero(features)
    return -(
        (log_prob * positive_weights).sum(dim=1) / positive_sum.clamp_min(1e-12)
    )[valid].mean()


def attribute_invariance_loss(
    disease_features: torch.Tensor,
    disease_labels: torch.Tensor,
    attributes: torch.Tensor,
) -> torch.Tensor:
    """Mean squared distance of normalized same-disease, cross-group pairs."""
    if disease_features.ndim != 2 or attributes.ndim != 2:
        raise ValueError("Expected disease_features [B,D] and attributes [B,A].")
    if disease_features.shape[0] != attributes.shape[0]:
        raise ValueError("Batch dimensions do not match.")
    if disease_features.shape[0] < 2:
        return _connected_zero(disease_features)
    labels = disease_labels.reshape(-1).long()
    same_disease = labels[:, None].eq(labels[None, :])
    upper = torch.triu(torch.ones_like(same_disease, dtype=torch.bool), diagonal=1)
    normalized = F.normalize(disease_features.float(), dim=1, eps=1e-12)
    squared_distances = torch.cdist(normalized, normalized).square()
    terms = []
    for index in range(attributes.shape[1]):
        different = attributes[:, index, None].ne(attributes[None, :, index])
        mask = same_disease & different & upper
        if mask.any():
            terms.append(squared_distances[mask].mean())
    return torch.stack(terms).mean() if terms else _connected_zero(disease_features)


def normalized_conditional_cross_covariance_loss(
    attribute_features: torch.Tensor,
    disease_features: torch.Tensor,
    disease_labels: torch.Tensor,
) -> torch.Tensor:
    """Experimental cross-covariance penalty with sample-count normalization."""
    if attribute_features.ndim != 2 or disease_features.ndim != 2:
        raise ValueError("Features must have shape [B,D].")
    labels = disease_labels.reshape(-1).long()
    terms = []
    for disease_class in torch.unique(labels):
        mask = labels == disease_class
        if int(mask.sum()) < 2:
            continue
        z_a = attribute_features[mask].float()
        z_d = disease_features[mask].float()
        z_a = z_a - z_a.mean(dim=0, keepdim=True)
        z_d = z_d - z_d.mean(dim=0, keepdim=True)
        terms.append((z_a.T @ z_d / int(mask.sum())).square().mean())
    return torch.stack(terms).mean() if terms else _connected_zero(attribute_features, disease_features)


@torch.no_grad()
def sinkhorn_balanced_assignment(logits: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """Balanced class-local assignments; valid even when samples < prototypes."""
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [samples, prototypes>0].")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    values = logits.detach().float()
    values = values - values.max()
    q = torch.exp(values).T.clamp_min(1e-12)  # [K,N]
    q /= q.sum().clamp_min(1e-12)
    prototypes, samples = q.shape
    for _ in range(iterations):
        q /= q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q /= float(prototypes)
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= float(samples)
    q *= float(samples)
    return q.T.contiguous()


def prototype_assignments(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normalized_prototypes: torch.Tensor,
    prototype_mask: torch.Tensor,
    tau_p: float = 0.2,
    sinkhorn_iterations: int = 3,
    assignment_temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return natural ``p``, balanced ``q.detach()``, and prototype CE loss.

    Sinkhorn ``q`` is only a collapse-resistant training target.  Callers must
    use detached natural softmax probabilities ``p`` as prototype estimates.
    """
    if tau_p <= 0:
        raise ValueError("tau_p must be positive.")
    if assignment_temperature is not None and assignment_temperature <= 0:
        raise ValueError("assignment_temperature must be positive.")
    labels = labels.reshape(-1).long()
    if embeddings.ndim != 2 or normalized_prototypes.ndim != 3:
        raise ValueError("Expected embeddings [B,D] and prototypes [2,K,D].")
    maximum = normalized_prototypes.shape[1]
    probabilities = embeddings.new_zeros((embeddings.shape[0], maximum), dtype=torch.float32)
    targets = embeddings.new_zeros((embeddings.shape[0], maximum), dtype=torch.float32)
    term_sums = []
    valid_samples = 0
    for disease_class in (0, 1):
        sample_mask = labels == disease_class
        count = int(sample_mask.sum())
        prototype_count = int(prototype_mask[disease_class].sum())
        if count == 0:
            continue
        logits = (
            embeddings[sample_mask].float()
            @ normalized_prototypes[disease_class, :prototype_count].float().T
            / tau_p
        )
        p = torch.softmax(logits, dim=1)
        assignment_logits = (logits if assignment_temperature is None else
                             logits.detach() * (tau_p / assignment_temperature))
        q = sinkhorn_balanced_assignment(assignment_logits, sinkhorn_iterations)
        probabilities[sample_mask, :prototype_count] = p
        targets[sample_mask, :prototype_count] = q
        if count >= 2:
            term_sums.append(-(q * torch.log_softmax(logits, dim=1)).sum())
            valid_samples += count
    loss = (
        torch.stack(term_sums).sum() / valid_samples
        if term_sums
        else _connected_zero(embeddings, normalized_prototypes)
    )
    return probabilities, targets.detach(), loss


def prototype_probabilities(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normalized_prototypes: torch.Tensor,
    prototype_mask: torch.Tensor,
    tau_p: float = 0.2,
) -> torch.Tensor:
    """Natural class-local softmax prototype probabilities without Sinkhorn."""
    if tau_p <= 0:
        raise ValueError("tau_p must be positive.")
    labels = labels.reshape(-1).long()
    maximum = normalized_prototypes.shape[1]
    probabilities = embeddings.new_zeros(
        (embeddings.shape[0], maximum), dtype=torch.float32
    )
    for disease_class in (0, 1):
        sample_mask = labels == disease_class
        prototype_count = int(prototype_mask[disease_class].sum())
        if not sample_mask.any():
            continue
        logits = (
            embeddings[sample_mask].float()
            @ normalized_prototypes[disease_class, :prototype_count].float().T
            / tau_p
        )
        probabilities[sample_mask, :prototype_count] = torch.softmax(logits, dim=1)
    return probabilities


def prototype_guided_contrastive_loss(
    z_d: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    tau: float = 0.1,
    gamma: float = 1.0,
    alpha: float = 1.0,
    sigma: float = 0.5,
    rho: float = 1.0,
    use_prototype_weights: bool = True,
    use_attribute_kernel: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prototype-guided cross-group contrastive loss (Eqs. 2-5)."""
    if tau <= 0 or sigma <= 0:
        raise ValueError("Temperatures and sigma must be w_positive.")
    if gamma < 0 or alpha < 0:
        raise ValueError("gamma and alpha must be non-w_negative.")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1].")
    if z_d.ndim != 2 or a.ndim != 2 or p.ndim != 2:
        raise ValueError("Expected z_d/a/p to be matrices.")
    batch = z_d.shape[0]
    zero = _connected_zero(z_d)
    empty_diagnostics = {
        "mean_positive_weight": zero.detach(),
        "mean_negative_weight": zero.detach(),
        "valid_contrastive_anchor_fraction": zero.detach(),
        "mean_same_prototype_cross_group_similarity": zero.detach(),
        "mean_same_prototype_same_group_similarity": zero.detach(),
        "mean_same_attribute_cross_disease_similarity": zero.detach(),
    }
    if batch < 2:
        return zero, empty_diagnostics
    y = y.reshape(-1).long()
    self_mask = torch.eye(batch, dtype=torch.bool, device=z_d.device)
    same_disease = y[:, None].eq(y[None, :]) & ~self_mask
    different_disease = y[:, None].ne(y[None, :]) & ~self_mask
    delta = a[:, None, :].ne(a[None, :, :]).float().mean(dim=2)
    if use_prototype_weights:
        learned_match = p.float() @ p.float().T
        r = (1.0 - rho) + rho * learned_match
    else:
        r = torch.ones_like(delta)
    w_positive = same_disease.float() * (r + eps).pow(gamma)
    w_positive = w_positive * (1.0 + alpha * delta)
    negative_kernel = (
        torch.exp(-delta / sigma)
        if use_attribute_kernel
        else torch.ones_like(delta)
    )
    w_negative = different_disease.float() * negative_kernel
    w = w_positive + w_negative

    normalized = F.normalize(z_d.float(), dim=1, eps=1e-12)
    similarities = normalized @ normalized.T
    q = similarities / tau
    q = q.masked_fill(self_mask, -torch.inf)
    weighted_logits = q + torch.log(w.clamp_min(eps))
    weighted_logits = weighted_logits.masked_fill(w <= 0, -torch.inf)
    log_b = torch.logsumexp(weighted_logits.float(), dim=1)
    positive_sum = w_positive.sum(dim=1)
    valid = (positive_sum > eps) & torch.isfinite(log_b)
    if valid.any():
        log_probability = q.float() - log_b[:, None]
        weighted_log_probability = torch.where(
            w_positive > 0,
            w_positive * log_probability,
            torch.zeros_like(log_probability),
        )
        anchor_loss = -weighted_log_probability.sum(dim=1) / positive_sum.clamp_min(eps)
        loss = anchor_loss[valid].mean()
    else:
        loss = zero

    def masked_mean(values, mask):
        return values[mask].mean().detach() if mask.any() else zero.detach()

    same_group = delta == 0
    # A pair is considered same-prototype for diagnostics when p agree most.
    hard_assignment = p.argmax(dim=1)
    same_prototype = hard_assignment[:, None].eq(hard_assignment[None, :])
    diagnostics = {
        "mean_positive_weight": masked_mean(w_positive, same_disease),
        "mean_negative_weight": masked_mean(w_negative, different_disease),
        "valid_contrastive_anchor_fraction": valid.float().mean().detach(),
        "mean_same_prototype_cross_group_similarity": masked_mean(
            similarities, same_disease & same_prototype & ~same_group
        ),
        "mean_same_prototype_same_group_similarity": masked_mean(
            similarities, same_disease & same_prototype & same_group & ~self_mask
        ),
        "mean_same_attribute_cross_disease_similarity": masked_mean(
            similarities, different_disease & same_group
        ),
    }
    return loss, diagnostics


def _multi_rbf(left: torch.Tensor, right: torch.Tensor, bandwidths: torch.Tensor):
    squared = torch.cdist(left.float(), right.float()).square()
    kernels = [torch.exp(-squared / (2.0 * bandwidth.square())) for bandwidth in bandwidths]
    return torch.stack(kernels).mean(dim=0)


def _weighted_mmd(
    left: torch.Tensor,
    right: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    bandwidths: torch.Tensor,
) -> torch.Tensor:
    lw = left_weight.float() / left_weight.float().sum().clamp_min(1e-12)
    rw = right_weight.float() / right_weight.float().sum().clamp_min(1e-12)
    k_ll = _multi_rbf(left, left, bandwidths)
    k_rr = _multi_rbf(right, right, bandwidths)
    k_lr = _multi_rbf(left, right, bandwidths)
    value = lw @ k_ll @ lw + rw @ k_rr @ rw - 2.0 * (lw @ k_lr @ rw)
    return value.clamp_min(0.0)


def prototype_conditional_mmd_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    attributes: torch.Tensor,
    assignments: torch.Tensor,
    prototype_mask: torch.Tensor,
    min_effective_weight: float = 0.5,
    bandwidth_floor: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align binary attributes within each disease/soft-prototype condition."""
    if min_effective_weight < 0 or bandwidth_floor <= 0:
        raise ValueError("Invalid MMD effective-weight or bandwidth floor.")
    zero = _connected_zero(embeddings)
    if embeddings.shape[0] < 2:
        return zero, zero.detach()
    with torch.no_grad():
        distances = torch.pdist(embeddings.detach().float())
        positive_distances = distances[distances > 0]
        median = (
            positive_distances.median()
            if positive_distances.numel()
            else embeddings.new_tensor(bandwidth_floor, dtype=torch.float32)
        ).clamp_min(bandwidth_floor)
        bandwidths = median * embeddings.new_tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    labels = labels.reshape(-1).long()
    terms = []
    for disease_class in (0, 1):
        class_mask = labels == disease_class
        for prototype_index in range(assignments.shape[1]):
            if not bool(prototype_mask[disease_class, prototype_index]):
                continue
            prototype_weight = assignments[:, prototype_index].float()
            for attribute_index in range(attributes.shape[1]):
                left_mask = class_mask & (attributes[:, attribute_index] == 0)
                right_mask = class_mask & (attributes[:, attribute_index] == 1)
                if int(left_mask.sum()) < 2 or int(right_mask.sum()) < 2:
                    continue
                left_weight = prototype_weight[left_mask]
                right_weight = prototype_weight[right_mask]
                if (
                    float(left_weight.sum()) < min_effective_weight
                    or float(right_weight.sum()) < min_effective_weight
                ):
                    continue
                terms.append(
                    _weighted_mmd(
                        embeddings[left_mask], embeddings[right_mask],
                        left_weight, right_weight, bandwidths,
                    )
                )
    if not terms:
        return zero, zero.detach()
    return torch.stack(terms).mean(), embeddings.new_tensor(float(len(terms))).detach()


def ema_prototype_consistency_loss(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor | None,
) -> torch.Tensor:
    """KL consistency from an EMA weak-view teacher to the strong-view student."""
    if teacher_probabilities is None:
        return _connected_zero(student_probabilities)
    if student_probabilities.shape != teacher_probabilities.shape:
        raise ValueError("Student and EMA prototype probabilities must have equal shape.")
    teacher = teacher_probabilities.detach().float()
    student = student_probabilities.float()
    valid = teacher.sum(dim=1) > 0
    if not valid.any():
        return _connected_zero(student_probabilities)
    per_sample = (
        teacher
        * (
            teacher.clamp_min(1e-12).log()
            - student.clamp_min(1e-12).log()
        )
    ).sum(dim=1)
    return per_sample[valid].mean()


def prototype_demographic_independence_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    attributes: torch.Tensor,
    prototype_mask: torch.Tensor,
) -> torch.Tensor:
    """Weakly penalize disease-conditional prototype/demographic correlation."""
    labels = labels.reshape(-1).long()
    terms = []
    for disease_class in (0, 1):
        sample_mask = labels == disease_class
        count = int(sample_mask.sum())
        prototype_count = int(prototype_mask[disease_class].sum())
        if count < 2 or prototype_count == 0:
            continue
        p = probabilities[sample_mask, :prototype_count].float()
        a = attributes[sample_mask].float()
        p = p - p.mean(dim=0, keepdim=True)
        a = a - a.mean(dim=0, keepdim=True)
        p_std = p.square().mean(dim=0).sqrt().clamp_min(1e-4)
        a_std = a.square().mean(dim=0).sqrt().clamp_min(1e-4)
        correlation = (p.T @ a / count) / (p_std[:, None] * a_std[None, :])
        terms.append(correlation.square().mean())
    return (
        torch.stack(terms).mean()
        if terms
        else _connected_zero(probabilities)
    )


def prototype_diagnostics(
    probabilities: torch.Tensor,
    sinkhorn_targets: torch.Tensor,
    labels: torch.Tensor,
    prototype_mask: torch.Tensor,
) -> dict[str, torch.Tensor | list[float]]:
    """Report natural, hard-natural, and balanced-target prototype usage."""
    labels = labels.reshape(-1).long()
    result: dict[str, torch.Tensor | list[float]] = {}
    for disease_class in (0, 1):
        sample_mask = labels == disease_class
        count = int(prototype_mask[disease_class].sum())
        if sample_mask.any():
            natural_usage = probabilities[sample_mask, :count].float().mean(dim=0)
            sinkhorn_usage = sinkhorn_targets[sample_mask, :count].float().mean(dim=0)
            hard_indices = probabilities[sample_mask, :count].argmax(dim=1)
            hard_usage = F.one_hot(hard_indices, num_classes=count).float().mean(dim=0)
            entropy = -(natural_usage * natural_usage.clamp_min(1e-12).log()).sum()
        else:
            natural_usage = probabilities.new_zeros(count)
            sinkhorn_usage = probabilities.new_zeros(count)
            hard_usage = probabilities.new_zeros(count)
            entropy = probabilities.new_zeros(())
        natural_values = [float(x) for x in natural_usage.detach().cpu()]
        result[f"prototype_usage_y{disease_class}"] = natural_values
        result[f"natural_probability_usage_y{disease_class}"] = natural_values
        result[f"sinkhorn_target_usage_y{disease_class}"] = [
            float(x) for x in sinkhorn_usage.detach().cpu()
        ]
        result[f"hard_probability_assignment_usage_y{disease_class}"] = [
            float(x) for x in hard_usage.detach().cpu()
        ]
        result[f"prototype_entropy_y{disease_class}"] = entropy.detach()
        p = probabilities[sample_mask, :count].float()
        sample_entropy = (-(p * p.clamp_min(1e-12).log()).sum(1).mean()
                          if sample_mask.any() else entropy)
        result[f"prototype_sample_entropy_y{disease_class}"] = sample_entropy.detach()
        result[f"prototype_information_y{disease_class}"] = (entropy - sample_entropy).detach()
        result[f"prototype_max_probability_y{disease_class}"] = (
            p.max(1).values.mean().detach() if sample_mask.any() else entropy.detach())
        result[f"prototype_hard_max_usage_y{disease_class}"] = hard_usage.max().detach()
    return result


def alignment_losses(
    outputs: dict[str, torch.Tensor],
    disease_labels: torch.Tensor,
    attributes: torch.Tensor,
    lambda_d_con: float = 1.0,
    lambda_inv: float = 0.1,
    lambda_decorr: float = 1.0,
    tau: float = 0.1,
    alpha: float = 1.0,
    contrastive_scale: float = 1.0,
    constraint_scale: float = 1.0,
    pos_weight: torch.Tensor | None = None,
    *,
    contrastive_mode: str = "cross_group_baseline",
    lambda_a: float = 1.0,
    lambda_alignment: float = 1.0,
    lambda_sinkhorn: float = 0.1,
    lambda_pc_mmd: float = 0.1,
    lambda_ema_consistency: float = 0.1,
    lambda_prototype_demographic: float = 0.01,
    prototype_scale: float = 1.0,
    alignment_scale: float = 1.0,
    ema_consistency_scale: float = 1.0,
    demographic_scale: float = 1.0,
    tau_p: float = 0.2,
    assignment_temperature: float | None = None,
    sinkhorn_iterations: int = 3,
    gamma: float = 1.0,
    sigma: float = 0.5,
    mmd_min_effective_weight: float = 0.5,
    rho: float = 1.0,
    use_prototype_weights: bool = True,
    use_attribute_kernel: bool = True,
    ema_teacher_probabilities: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compute cross-group alignment and optional experimental regularizers."""
    disease = F.binary_cross_entropy_with_logits(
        outputs["disease_logits"], disease_labels.float(), pos_weight=pos_weight
    )
    attribute = torch.stack(
        [
            F.cross_entropy(outputs["attribute_logits"][:, index], attributes[:, index].long())
            for index in range(attributes.shape[1])
        ]
    ).mean()
    zero = _connected_zero(outputs["z_d"])
    diagnostics: dict[str, object] = {}

    if contrastive_mode == "cross_group_baseline":
        current = attribute_aware_cross_group_supcon_loss(
            outputs["z_d"], disease_labels, attributes, tau, alpha
        )
        invariance = attribute_invariance_loss(outputs["z_d"], disease_labels, attributes)
        decorrelation = normalized_conditional_cross_covariance_loss(
            outputs["z_a"], outputs["z_d"], disease_labels
        )
        weighted_current = contrastive_scale * lambda_d_con * current
        weighted_inv = constraint_scale * lambda_inv * invariance
        weighted_decorr = constraint_scale * lambda_decorr * decorrelation
        return {
            "total": disease + lambda_a * attribute + weighted_current + weighted_inv + weighted_decorr,
            "disease": disease,
            "attribute": attribute,
            "disease_contrastive": current,
            "invariance": invariance,
            "decorrelation": decorrelation,
            "weighted_disease_contrastive": weighted_current,
            "weighted_invariance": weighted_inv,
            "weighted_decorrelation": weighted_decorr,
            "d_con": zero, "sinkhorn": zero, "pc_mmd": zero,
            "ema_consistency": zero, "prototype_demographic": zero,
            "weighted_d_con": zero, "weighted_sinkhorn": zero, "weighted_pc_mmd": zero,
            "weighted_ema_consistency": zero,
            "weighted_prototype_demographic": zero,
            "diagnostics": diagnostics,
        }
    if contrastive_mode != "prototype_guided":
        raise ValueError("contrastive_mode must be 'cross_group_baseline' or 'prototype_guided'.")

    probabilities, sinkhorn_targets, prototype = prototype_assignments(
        outputs["z_d"], disease_labels, outputs["normalized_prototypes"],
        outputs["prototype_mask"], tau_p, sinkhorn_iterations,
        assignment_temperature=assignment_temperature,
    )
    prototype_probabilities = probabilities.detach()
    ema_consistency = ema_prototype_consistency_loss(
        probabilities, ema_teacher_probabilities
    )
    prototype_demographic = prototype_demographic_independence_loss(
        probabilities,
        disease_labels,
        attributes,
        outputs["prototype_mask"],
    )
    d_con, contrastive_diagnostics = prototype_guided_contrastive_loss(
        outputs["z_d"], disease_labels, attributes, prototype_probabilities, tau,
        gamma, alpha, sigma,
        rho, use_prototype_weights, use_attribute_kernel,
    )
    pc_mmd, valid_mmd_terms = prototype_conditional_mmd_loss(
        outputs["z_d"], disease_labels, attributes, prototype_probabilities,
        outputs["prototype_mask"], mmd_min_effective_weight,
    )
    weighted_d_con = contrastive_scale * lambda_alignment * d_con
    weighted_sinkhorn = prototype_scale * lambda_sinkhorn * prototype
    weighted_pc_mmd = alignment_scale * lambda_pc_mmd * pc_mmd
    weighted_ema_consistency = (
        ema_consistency_scale * lambda_ema_consistency * ema_consistency
    )
    weighted_prototype_demographic = (
        demographic_scale * lambda_prototype_demographic * prototype_demographic
    )
    diagnostics.update(contrastive_diagnostics)
    diagnostics.update(
        prototype_diagnostics(
            prototype_probabilities,
            sinkhorn_targets,
            disease_labels,
            outputs["prototype_mask"],
        )
    )
    diagnostics["valid_mmd_term_count"] = valid_mmd_terms
    diagnostics["assignments"] = prototype_probabilities
    diagnostics["assignment_probabilities"] = prototype_probabilities
    diagnostics["sinkhorn_targets"] = sinkhorn_targets
    return {
        "total": (
            disease
            + lambda_a * attribute
            + weighted_d_con
            + weighted_sinkhorn
            + weighted_pc_mmd
            + weighted_ema_consistency
            + weighted_prototype_demographic
        ),
        "disease": disease,
        "attribute": attribute,
        "disease_contrastive": zero,
        "invariance": zero,
        "decorrelation": zero,
        "weighted_disease_contrastive": zero,
        "weighted_invariance": zero,
        "weighted_decorrelation": zero,
        "d_con": d_con,
        "sinkhorn": prototype,
        "pc_mmd": pc_mmd,
        "ema_consistency": ema_consistency,
        "prototype_demographic": prototype_demographic,
        "weighted_d_con": weighted_d_con,
        "weighted_sinkhorn": weighted_sinkhorn,
        "weighted_pc_mmd": weighted_pc_mmd,
        "weighted_ema_consistency": weighted_ema_consistency,
        "weighted_prototype_demographic": weighted_prototype_demographic,
        "diagnostics": diagnostics,
    }


def semantic_losses(outputs, disease_labels, attributes, lambda_a, pos_weight=None):
    """Binary disease cross-entropy and mean attribute cross-entropy."""
    disease = F.binary_cross_entropy_with_logits(
        outputs["disease_logits"], disease_labels.float(), pos_weight=pos_weight
    )
    attribute = torch.stack(
        [
            F.cross_entropy(
                outputs["attribute_logits"][:, index], attributes[:, index].long()
            )
            for index in range(attributes.shape[1])
        ]
    ).mean()
    return {
        "disease": disease,
        "attribute": attribute,
        "semantic": disease + float(lambda_a) * attribute,
    }


def feature_decorrelation_loss(z_d, z_a, disease_labels):
    """Feature-level decorrelation with a detached attribute space (Eqs. 6-7)."""
    labels = disease_labels.reshape(-1).long()
    z_d = z_d.float()
    z_a = z_a.detach().float()
    loss = z_d.new_zeros(())
    valid = 0
    for disease_class in (0, 1):
        mask = labels == disease_class
        if mask.sum() < 2:
            continue
        disease = z_d[mask] - z_d[mask].mean(dim=0, keepdim=True)
        attribute = z_a[mask] - z_a[mask].mean(dim=0, keepdim=True)
        loss = loss + (disease.T @ attribute).square().mean()
        valid += 1
    return loss / valid if valid else z_d.sum() * 0.0


def relation_decorrelation_loss(
    z_d: torch.Tensor,
    z_a: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Biased squared distance correlation with a detached attribute space.

    Both inputs contain the same batch of samples as ``[B, D]`` matrices. The
    attribute representation defines a fixed reference geometry: gradients
    flow through ``z_d`` only. Pairwise Euclidean distance
    matrices are double-centered before their normalized Frobenius inner
    product is computed.
    """
    if z_d.ndim != 2 or z_a.ndim != 2:
        raise ValueError("Expected z_d and z_a to be matrices.")
    if z_d.shape[0] != z_a.shape[0]:
        raise ValueError("Disease and attribute batch dimensions do not match.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if z_d.shape[0] < 2:
        return _connected_zero(z_d)

    disease = z_d.float()
    attribute = z_a.detach().float()
    D_d = torch.cdist(disease, disease, p=2)
    D_a = torch.cdist(attribute, attribute, p=2)

    def double_center(distances: torch.Tensor) -> torch.Tensor:
        return (
            distances
            - distances.mean(dim=1, keepdim=True)
            - distances.mean(dim=0, keepdim=True)
            + distances.mean()
        )

    A_d = double_center(D_d)
    A_a = double_center(D_a)
    V_da = (
        A_d * A_a
    ).mean().clamp_min(0.0)
    V_dd = A_d.square().mean()
    V_aa = A_a.square().mean()
    denominator = torch.sqrt(
        V_dd * V_aa
    )
    return V_da / (denominator + eps)


def conditional_demographic_distance_correlation_loss(
    disease_features: torch.Tensor,
    disease_labels: torch.Tensor,
    demographic_labels: torch.Tensor,
    min_samples_per_class: int = 4,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Average demographic-label dCor within each disease class.

    ``demographic_labels`` must contain the three binary columns Age, Sex, and
    Ethnicity. Each of the six attribute-by-disease-class terms is computed
    independently. Terms with fewer than ``min_samples_per_class`` samples are
    skipped, and the remaining valid terms are averaged. The demographic
    labels are treated as detached scalar 0/1 coordinates; they are neither
    one-hot encoded nor concatenated into a joint demographic vector.
    """
    if disease_features.ndim != 2:
        raise ValueError("Expected disease_features to be a [B,D] matrix.")
    labels = disease_labels.reshape(-1)
    if labels.shape[0] != disease_features.shape[0]:
        raise ValueError("Disease labels and features have different batch sizes.")
    if demographic_labels.ndim != 2 or demographic_labels.shape != (
        disease_features.shape[0], 3
    ):
        raise ValueError(
            "Expected demographic_labels to have shape [B,3] in "
            "Age/Sex/Ethnicity order."
        )
    if min_samples_per_class < 2:
        raise ValueError("min_samples_per_class must be at least 2.")

    terms = []
    labels = labels.long()
    for attribute_index in range(3):
        for disease_class in (0, 1):
            mask = labels == disease_class
            if int(mask.sum().item()) < min_samples_per_class:
                continue
            attribute = demographic_labels[mask, attribute_index : attribute_index + 1]
            # dCor is defined as zero when the scalar label has no variation.
            # Returning an explicitly connected zero also avoids the undefined
            # derivative of sqrt(0) in the generic normalized implementation.
            if torch.unique(attribute.detach()).numel() < 2:
                terms.append(_connected_zero(disease_features[mask]))
                continue
            terms.append(
                relation_decorrelation_loss(disease_features[mask], attribute, eps=eps)
            )
    if not terms:
        return _connected_zero(disease_features)
    return torch.stack(terms).mean()
