#!/usr/bin/env python3
"""Train prototype-guided cross-group alignment and dual-level decorrelation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler

from .data import (
    ATTRIBUTE_NAMES,
    TASKS,
    build_transform,
    make_disease_loaders,
)
from evaluation.metrics import (
    best_f1_threshold,
    save_detailed_fairness_metrics,
    save_result_tables,
    validation_metrics,
)

from .losses import (
    conditional_demographic_distance_correlation_loss,
    feature_decorrelation_loss,
    alignment_losses,
    relation_decorrelation_loss,
    prototype_probabilities,
    semantic_losses,
)
from .model import DiseaseAttributeModel
from .disease_pcgrad import disease_priority_multistate_step
from .multistate_adamw import MultiStateDiseasePriorityAdamW


CONFIG_ALIASES = {
    'lambda_attr': 'lambda_a',
    'temperature': 'tau',
    'cross_group_alpha': 'alpha',
    'ppicd_ema_delay_epochs': 'ema_delay_epochs',
    'protect_decoupling_budget': 'protect_decorrelation_budget',
    'prototype_temperature': 'tau_p',
    'phenotype_gamma': 'gamma',
    'attribute_kernel_sigma': 'sigma',
    'lambda_ppicd': 'lambda_alignment',
    'lambda_prototype': 'lambda_sinkhorn',
    'ppicd_semantic_warmup_epochs': 'alignment_semantic_warmup_epochs',
    'ppicd_prototype_warmup_epochs': 'prototype_warmup_epochs',
    'ppicd_contrastive_ramp_epochs': 'alignment_ramp_epochs',
    'ppicd_alignment_ramp_epochs': 'mmd_ramp_epochs',
    'use_phenotype_weights': 'use_prototype_weights',
    'lambda_row': 'lambda_decorr',
    'lambda_rel': 'lambda_dcor',
}


def normalize_config_keys(config):
    """Translate earlier option names; explicit current names take precedence."""
    result = dict(config)
    for old, new in CONFIG_ALIASES.items():
        if old in result:
            result.setdefault(new, result.pop(old))
    return result


EVALUATION_CHECKPOINTS = (
    "best_val_loss",
    "best_val_auc",
    "best_val_delta_eo",
    "best_val_delta_auc",
)
LOSS_NAMES = (
    "total", "semantic", "disease", "attribute", "weighted_attribute",
    "d_con", "sinkhorn", "pc_mmd", "ema_consistency",
    "prototype_demographic", "weighted_d_con", "weighted_sinkhorn",
    "weighted_pc_mmd", "weighted_ema_consistency",
    "weighted_prototype_demographic", "weighted_alignment", "decorr",
    "weighted_decorr", "dcor", "weighted_dcor", "decorrelation",
)
PROTOTYPE_METRIC_NAMES = tuple(
    f"prototype_{name}_y{y}" for y in (0, 1)
    for name in ("sample_entropy", "information", "max_probability", "hard_max_usage")
)
LOSS_NAMES = (*LOSS_NAMES, *PROTOTYPE_METRIC_NAMES)
GRADIENT_NAMES = (
    "backbone_contrastive_disease_norm_ratio", "disease_projector_contrastive_disease_norm_ratio",
    "backbone_reserved_auxiliary_disease_norm_ratio", "disease_projector_reserved_auxiliary_disease_norm_ratio",
    "backbone_combined_auxiliary_disease_norm_ratio", "disease_projector_combined_auxiliary_disease_norm_ratio",
    "backbone_contrastive_budget_scale", "disease_projector_contrastive_budget_scale",
    "row_budgeted_delta_norm", "relation_budgeted_delta_norm",
    "disease_gradient_norm", "attribute_gradient_norm", "contrastive_gradient_norm", "row_gradient_norm",
    "relation_gradient_norm", "disease_gradient_share",
    "attribute_gradient_share", "contrastive_gradient_share", "row_gradient_share", "relation_gradient_share",
    "attribute_cosine_before", "attribute_cosine_after", "attribute_gradient_conflict",
    "contrastive_cosine_before", "contrastive_cosine_after", "contrastive_gradient_conflict",
    "row_cosine_before", "row_cosine_after", "row_gradient_conflict",
    "relation_cosine_before", "relation_cosine_after", "relation_gradient_conflict",
    "disease_gradient_clip_scale", "attribute_gradient_clip_scale", "contrastive_gradient_clip_scale",
    "row_gradient_clip_scale", "relation_gradient_clip_scale",
    "backbone_auxiliary_budget_scale", "disease_projector_auxiliary_budget_scale",
    "attribute_displacement_dot_before", "attribute_displacement_dot_after",
    "attribute_displacement_harmful", "contrastive_displacement_dot_before",
    "contrastive_displacement_dot_after", "contrastive_displacement_harmful",
    "row_displacement_dot_before",
    "row_displacement_dot_after", "row_displacement_harmful",
    "relation_displacement_dot_before", "relation_displacement_dot_after",
    "relation_displacement_harmful", "disease_dot_actual_delta",
    "disease_dot_before_safety_projection", "disease_safety_projection_applied",
    "disease_gradient_nonzero", "disease_actual_descent", "actual_delta_norm",
    "all_gradients_finite",
)


@dataclass
class TrainConfig:
    target: str
    output_dir: str
    metadata_csv: str = ""
    image_root: str | None = None
    normalization_stats: str | None = None
    epochs: int = 40
    semantic_warmup_epochs: int = 5
    relation_ramp_epochs: int = 10
    batch_size: int = 128
    probe_feature_batch_size: int = 256
    num_workers: int = 0
    image_size: int = 224
    dropout: float = 0.2
    lambda_a: float = 0.25
    lambda_d_con: float = 1.0
    tau: float = 0.1
    alpha: float = 1.0
    supcon_positive_weighting: str = "difference_count"
    contrastive_initial_scale: float = 0.1
    contrastive_warmup_epochs: int = 5
    contrastive_candidate_scale: float = 0.05
    negative_prototypes: int = 3
    positive_prototypes: int = 3
    assignment_temperature: float | None = None
    prototype_candidate_scale: float | None = None
    ema_delay_epochs: int = 0
    protect_decorrelation_budget: bool = False
    independent_contrastive_budget_ratio: float | None = None
    tau_p: float = 0.2
    sinkhorn_iterations: int = 3
    gamma: float = 1.0
    sigma: float = 0.5
    lambda_alignment: float = 1.0
    lambda_sinkhorn: float = 0.1
    lambda_pc_mmd: float = 0.1
    lambda_ema_consistency: float = 0.1
    lambda_prototype_demographic: float = 0.01
    ema_decay: float = 0.996
    mmd_min_effective_weight: float = 0.5
    alignment_semantic_warmup_epochs: int = 5
    prototype_warmup_epochs: int = 2
    alignment_ramp_epochs: int = 5
    mmd_ramp_epochs: int = 10
    use_prototype_weights: bool = True
    use_attribute_kernel: bool = True
    lambda_decorr: float = 500.0
    lambda_dcor: float = 1.0
    relation_mode: str = "feature_spaces"
    relation_min_class_samples: int = 4
    auxiliary_update_budget_ratio: float = 1.0
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False
    train_fraction: float = 1.0
    subsample_seed: int = 0
    balanced_batches: bool = False
    utility_checkpoint_tolerance: float = 0.005
    independent_fairness_checkpoints: bool = True
    test_every_epoch: bool = False
    evaluation_only: bool = False
    save_last_checkpoint: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return dist.get_rank() if distributed_enabled() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if distributed_enabled() else 1


def initialize_distributed(device_name: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return torch.device(device_name), 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA/NCCL.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()


class RankShardSampler(Sampler[int]):
    """Exact non-padding rank shard for deterministic evaluation/basis passes."""

    def __init__(self, dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = max(len(self.dataset) - self.rank, 0)
        return (remaining + self.world_size - 1) // self.world_size


class DistributedBalancedBatchSampler:
    """Shard complete balanced batches while keeping equal DDP step counts.

    The underlying sampler independently produces the same deterministic batch
    sequence on every rank.  Whole balanced batches are assigned round-robin;
    at most ``world_size - 1`` already-balanced batches are repeated at the end
    so every rank executes the same number of optimizer steps.
    """

    def __init__(self, batch_sampler, rank: int, world_size: int) -> None:
        self.batch_sampler = batch_sampler
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        batches = list(iter(self.batch_sampler))
        if not batches:
            return iter(())
        remainder = len(batches) % self.world_size
        if remainder:
            batches.extend(batches[: self.world_size - remainder])
        return iter(batches[self.rank :: self.world_size])

    def __len__(self) -> int:
        return math.ceil(len(self.batch_sampler) / self.world_size)


def shard_loaders(loaders, cfg: TrainConfig, rank: int, world_size: int):
    if world_size == 1:
        return loaders
    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.num_workers > 0,
    }
    train = loaders["train"]
    if cfg.balanced_batches:
        train_loader = DataLoader(
            train.dataset,
            batch_sampler=DistributedBalancedBatchSampler(
                train.batch_sampler, rank, world_size
            ),
            **common,
        )
    else:
        train_loader = DataLoader(
            train.dataset,
            batch_size=cfg.batch_size,
            sampler=RankShardSampler(train.dataset, rank, world_size),
            **common,
        )
    result = {"train": train_loader}
    for split in ("val", "test"):
        dataset = loaders[split].dataset
        result[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=RankShardSampler(dataset, rank, world_size),
            **common,
        )
    return result


def broadcast_model_buffers(model) -> None:
    if distributed_enabled():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


def broadcast_model_state(model) -> None:
    if distributed_enabled():
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        broadcast_model_buffers(model)


def gather_numpy_arrays(arrays, device: torch.device):
    if not distributed_enabled():
        return arrays
    gathered_arrays = []
    world_size = distributed_world_size()
    for array in arrays:
        local = torch.as_tensor(array, device=device).contiguous()
        local_count = torch.tensor([local.shape[0]], dtype=torch.long, device=device)
        rank_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(rank_counts, local_count)
        counts = [int(value.item()) for value in rank_counts]
        max_count = max(counts)
        padded = torch.zeros(
            (max_count, *local.shape[1:]), dtype=local.dtype, device=device
        )
        padded[: local.shape[0]] = local
        rank_values = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(rank_values, padded)
        gathered_arrays.append(
            torch.cat(
                [value[:count] for value, count in zip(rank_values, counts)], dim=0
            ).cpu().numpy()
        )
    return tuple(gathered_arrays)


def amp_context(device: torch.device, enabled: bool):
    """Use BF16 on supported CUDA devices; gradients remain unscaled fp32."""
    active = enabled and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=active)


def decorrelation_scale(epoch: int, cfg: TrainConfig) -> float:
    """Zero through warmup, then a smooth zero-to-one cosine ramp."""
    stage_epoch = epoch - cfg.semantic_warmup_epochs
    if stage_epoch <= 0:
        return 0.0
    if cfg.relation_ramp_epochs <= 0 or stage_epoch >= cfg.relation_ramp_epochs:
        return 1.0
    progress = stage_epoch / cfg.relation_ramp_epochs
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def contrastive_scale(epoch: int, cfg: TrainConfig) -> float:
    """Linear ramp for the cross-group supervised contrastive baseline."""
    if cfg.contrastive_warmup_epochs <= 1 or epoch >= cfg.contrastive_warmup_epochs:
        return 1.0
    progress = (epoch - 1) / (cfg.contrastive_warmup_epochs - 1)
    return cfg.contrastive_initial_scale + (1.0 - cfg.contrastive_initial_scale) * progress


def _ramp(epoch_in_stage: int, length: int, initial: float = 0.1) -> float:
    if epoch_in_stage <= 0:
        return 0.0
    if length <= 1 or epoch_in_stage >= length:
        return 1.0
    progress = (epoch_in_stage - 1) / (length - 1)
    return initial + (1.0 - initial) * (0.5 - 0.5 * math.cos(math.pi * progress))


def _activation_ramp(epoch_in_stage: int, length: int) -> float:
    if epoch_in_stage <= 0:
        return 0.0
    if length <= 0:
        return 1.0
    progress = min(epoch_in_stage / length, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def _alignment_schedule(epoch: int, cfg: TrainConfig) -> dict[str, float]:
    """Warm up prototype assignments, then ramp contrastive alignment and MMD."""
    zero = {
        "contrastive": 0.0, "prototype": 0.0, "prototype_weight": 0.0,
        "alignment": 0.0, "ema": 0.0, "demographic": 0.0,
    }
    if epoch <= cfg.alignment_semantic_warmup_epochs:
        return zero
    prototype_epoch = epoch - cfg.alignment_semantic_warmup_epochs
    prototype = _activation_ramp(prototype_epoch, cfg.prototype_warmup_epochs)
    prototype_epoch = prototype_epoch - cfg.prototype_warmup_epochs
    if prototype_epoch <= 0:
        return {**zero, "prototype": prototype, "ema": prototype, "demographic": prototype}
    prototype_weight = _ramp(
        prototype_epoch, cfg.alignment_ramp_epochs, initial=0.0
    )
    alignment_epoch = prototype_epoch - cfg.alignment_ramp_epochs
    return {
        "contrastive": _activation_ramp(
            prototype_epoch, cfg.alignment_ramp_epochs
        ),
        "prototype": 1.0,
        "prototype_weight": prototype_weight,
        "alignment": _ramp(alignment_epoch, cfg.mmd_ramp_epochs),
        "ema": 1.0,
        "demographic": 1.0,
    }


def alignment_scales(epoch: int, cfg: TrainConfig) -> dict[str, float]:
    scales = _alignment_schedule(epoch, cfg)
    if cfg.ema_delay_epochs:
        scales["ema"] = _alignment_schedule(epoch - cfg.ema_delay_epochs, cfg)["ema"]
    return scales


def make_ema_model(model):
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay: float) -> None:
    student_state = model.state_dict()
    for name, ema_value in ema_model.state_dict().items():
        student_value = student_state[name].detach()
        if ema_value.is_floating_point():
            ema_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(student_value)


def build_optimizer(model: DiseaseAttributeModel, cfg: TrainConfig) -> MultiStateDiseasePriorityAdamW:
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    non_backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups = [
        {"params": non_backbone, "lr": cfg.learning_rate},
        {"params": backbone, "lr": cfg.backbone_learning_rate},
    ]
    if cfg.prototype_candidate_scale is not None:
        prototypes = list(model.class_prototypes.parameters())
        prototype_ids = {id(p) for p in prototypes}
        groups[0]["params"] = [p for p in non_backbone if id(p) not in prototype_ids]
        groups.append({"params": prototypes, "lr": cfg.learning_rate,
                       "contrastive_candidate_scale": cfg.prototype_candidate_scale})
    optimizer = MultiStateDiseasePriorityAdamW(
        groups,
        weight_decay=cfg.weight_decay,
    )
    optimized = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len({id(parameter) for parameter in optimized}) != len(optimized):
        raise RuntimeError("Optimizer contains duplicate parameters.")
    if {id(parameter) for parameter in optimized} != {id(parameter) for parameter in expected}:
        raise RuntimeError("Optimizer parameter coverage is incomplete.")
    return optimizer


def training_pos_weight(loader, device, balanced_batches: bool):
    if balanced_batches:
        return torch.ones((), device=device)
    labels = loader.dataset.frame[loader.dataset.target].to_numpy(dtype=float)
    positives = labels.sum()
    value = (len(labels) - positives) / positives if positives > 0 else 1.0
    return torch.tensor(float(value), device=device)


def make_probe_train_loader(train_loader, cfg: TrainConfig):
    """Each training sample once with validation preprocessing and no shuffle."""
    dataset = copy.copy(train_loader.dataset)
    dataset.dual_views = False
    statistics = dataset.normalization_statistics
    dataset.transform = build_transform(
        False, statistics["mean"], statistics["std"], cfg.image_size
    )
    return DataLoader(
        dataset,
        batch_size=cfg.probe_feature_batch_size,
        sampler=(
            RankShardSampler(dataset, distributed_rank(), distributed_world_size())
            if distributed_enabled()
            else None
        ),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )








def run_epoch(
    model, loader, cfg, device, pos_weight, scale, alignment_schedule,
    optimizer=None, ema_model=None,
):
    training = optimizer is not None
    model.train(training)
    sums = {name: 0.0 for name in (*LOSS_NAMES, *GRADIENT_NAMES)}
    labels_all, scores_all, attributes_all = [], [], []
    context = torch.enable_grad if training else torch.no_grad
    broadcast_model_buffers(model)
    with context():
        for batch in loader:
            if len(batch) == 4:
                teacher_images, images, labels, attributes = batch
                teacher_images = teacher_images.to(device, non_blocking=True)
            else:
                images, labels, attributes = batch
                teacher_images = images.to(device, non_blocking=True)
            if training:
                broadcast_model_buffers(model)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attributes = attributes.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp_context(device, cfg.amp):
                ema_teacher_probabilities = None
                if ema_model is not None and alignment_schedule["ema"] > 0:
                    with torch.no_grad():
                        teacher_outputs = ema_model(teacher_images)
                        ema_teacher_probabilities = prototype_probabilities(
                            teacher_outputs["z_d"], labels,
                            teacher_outputs["normalized_prototypes"],
                            teacher_outputs["prototype_mask"],
                            cfg.tau_p,
                        ).detach()
                outputs = model(images)
                semantic = semantic_losses(
                    outputs, labels, attributes, cfg.lambda_a, pos_weight
                )
            # Decorrelation runs in FP32. The conditional-label ablation uses
            # demographic labels in place of the attribute representation.
            if cfg.relation_mode == "conditional_demographics":
                dcor = conditional_demographic_distance_correlation_loss(
                    outputs["z_d"], labels, attributes,
                    min_samples_per_class=cfg.relation_min_class_samples,
                )
            else:
                dcor = relation_decorrelation_loss(outputs["z_d"], outputs["z_a"])
            decorr = feature_decorrelation_loss(outputs["z_d"], outputs["z_a"], labels)
            alignment_terms = alignment_losses(
                outputs, labels, attributes,
                tau=cfg.tau,
                alpha=cfg.alpha,
                contrastive_scale=alignment_schedule["contrastive"],
                pos_weight=pos_weight,
                contrastive_mode="prototype_guided",
                lambda_a=cfg.lambda_a,
                lambda_alignment=cfg.lambda_alignment,
                lambda_sinkhorn=cfg.lambda_sinkhorn,
                lambda_pc_mmd=cfg.lambda_pc_mmd,
                lambda_ema_consistency=cfg.lambda_ema_consistency,
                lambda_prototype_demographic=cfg.lambda_prototype_demographic,
                prototype_scale=alignment_schedule["prototype"],
                alignment_scale=alignment_schedule["alignment"],
                ema_consistency_scale=alignment_schedule["ema"] if training else 0.0,
                demographic_scale=alignment_schedule["demographic"],
                tau_p=cfg.tau_p,
                assignment_temperature=cfg.assignment_temperature,
                sinkhorn_iterations=cfg.sinkhorn_iterations,
                gamma=cfg.gamma,
                sigma=cfg.sigma,
                mmd_min_effective_weight=cfg.mmd_min_effective_weight,
                rho=alignment_schedule["prototype_weight"],
                use_prototype_weights=cfg.use_prototype_weights,
                use_attribute_kernel=cfg.use_attribute_kernel,
                ema_teacher_probabilities=ema_teacher_probabilities,
            )
            weighted_attribute = float(cfg.lambda_a) * semantic["attribute"]
            weighted_contrastive = sum(
                alignment_terms[name]
                for name in (
                    "weighted_d_con", "weighted_sinkhorn", "weighted_pc_mmd",
                    "weighted_ema_consistency", "weighted_prototype_demographic",
                )
            )
            weighted_decorr = float(scale) * cfg.lambda_decorr * decorr
            weighted_relation = float(scale) * cfg.lambda_dcor * dcor
            total = (
                semantic["disease"] + weighted_attribute + weighted_contrastive
                + weighted_decorr + weighted_relation
            )
            losses = {
                "total": total,
                **semantic,
                "weighted_attribute": weighted_attribute,
                **{name: alignment_terms[name] for name in (
                    "d_con", "sinkhorn", "pc_mmd", "ema_consistency",
                    "prototype_demographic", "weighted_d_con", "weighted_sinkhorn",
                    "weighted_pc_mmd", "weighted_ema_consistency",
                    "weighted_prototype_demographic",
                )},
                "weighted_alignment": weighted_contrastive,
                "decorr": decorr,
                "weighted_decorr": weighted_decorr,
                "dcor": dcor,
                "weighted_dcor": weighted_relation,
                "decorrelation": weighted_decorr + weighted_relation,
            }
            losses.update({name: alignment_terms["diagnostics"][name]
                           for name in PROTOTYPE_METRIC_NAMES})
            diagnostics = {name: total.detach().new_zeros(()) for name in GRADIENT_NAMES}
            if training:
                disease_parameters = (
                    list(model.backbone.parameters())
                    + list(model.disease_projector.parameters())
                    + list(model.disease_head.parameters())
                )
                diagnostics = disease_priority_multistate_step(
                    semantic["disease"],
                    weighted_attribute,
                    weighted_contrastive,
                    weighted_decorr,
                    weighted_relation,
                    optimizer,
                    disease_parameters,
                    {
                        "backbone": list(model.backbone.parameters()),
                        "disease_projector": list(model.disease_projector.parameters()),
                    },
                    auxiliary_update_budget_ratio=cfg.auxiliary_update_budget_ratio,
                    protect_decorrelation_budget=cfg.protect_decorrelation_budget,
                    independent_contrastive_budget_ratio=cfg.independent_contrastive_budget_ratio,
                    max_component_grad_norm=cfg.max_grad_norm,
                    decorrelation_active=scale > 0.0,
                    contrastive_candidate_scale=(
                        cfg.contrastive_candidate_scale
                        * max(alignment_schedule.values())
                    ),
                )
                if ema_model is not None:
                    update_ema_model(ema_model, model, cfg.ema_decay)

            batch_size = images.shape[0]
            for name in LOSS_NAMES:
                sums[name] += float(losses[name].detach()) * batch_size
            for name in GRADIENT_NAMES:
                sums[name] += float(diagnostics[name].detach()) * batch_size
            labels_all.append(labels.detach().cpu().numpy())
            scores_all.append(
                torch.sigmoid(outputs["disease_logits"].detach().float()).cpu().numpy()
            )
            attributes_all.append(attributes.detach().cpu().numpy())

    local_count = sum(len(values) for values in labels_all)
    arrays = gather_numpy_arrays((
        np.concatenate(labels_all), np.concatenate(scores_all), np.concatenate(attributes_all)
    ), device)
    totals = torch.tensor(
        [sums[name] for name in (*LOSS_NAMES, *GRADIENT_NAMES)] + [local_count],
        dtype=torch.float64,
        device=device,
    )
    if distributed_enabled():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = float(totals[-1].item())
    sums = {
        name: float(totals[index].item())
        for index, name in enumerate((*LOSS_NAMES, *GRADIENT_NAMES))
    }
    result = {name: value / count for name, value in sums.items()}
    result.update(validation_metrics(*arrays))
    return result, arrays


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, cfg, ema_model=None):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": asdict(cfg),
            "method": (
                "prototype_alignment_conditional_demographic_dcor_ema"
                if cfg.relation_mode == "conditional_demographics"
                else "prototype_alignment_dual_decorrelation_ema"
            ),
            "optimizer_type": "MultiStateDiseasePriorityAdamW",
            "optimizer_streams": ["disease", "attribute", "contrastive", "row", "relation"],
            "final_disease_descent_enforced_before_write": True,
        },
        path,
    )


def append_csv(path: Path, row: dict, first: bool):
    with path.open("w" if first else "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), extrasaction="ignore")
        if first:
            writer.writeheader()
        writer.writerow(row)


def load_model(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = DiseaseAttributeModel(
        num_attributes=len(ATTRIBUTE_NAMES),
        negative_prototypes=int(config.get("negative_prototypes", 3)),
        positive_prototypes=int(config.get("positive_prototypes", 3)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def validate_config(cfg):
    if cfg.independent_contrastive_budget_ratio is not None:
        if not cfg.protect_decorrelation_budget:
            raise ValueError("Independent contrastive budget requires protect_decorrelation_budget")
        if (not math.isfinite(cfg.independent_contrastive_budget_ratio)
                or cfg.independent_contrastive_budget_ratio < 0):
            raise ValueError("Independent contrastive budget must be finite and nonnegative")
    if cfg.assignment_temperature is not None and (not math.isfinite(cfg.assignment_temperature) or cfg.assignment_temperature <= 0):
        raise ValueError("assignment_temperature must be finite and positive")
    if cfg.prototype_candidate_scale is not None and (not math.isfinite(cfg.prototype_candidate_scale) or cfg.prototype_candidate_scale < 0):
        raise ValueError("prototype_candidate_scale must be finite and nonnegative")
    if cfg.ema_delay_epochs < 0:
        raise ValueError("ema_delay_epochs must be nonnegative")
    if cfg.target not in TASKS:
        raise ValueError(f"Unknown target {cfg.target!r}.")
    if cfg.epochs <= 0 or not 0 <= cfg.semantic_warmup_epochs < cfg.epochs:
        raise ValueError("Require epochs > 0 and 0 <= semantic_warmup_epochs < epochs.")
    if (
        cfg.relation_ramp_epochs < 0
        or cfg.lambda_decorr < 0
        or cfg.lambda_dcor < 0
        or cfg.lambda_a < 0
        or cfg.lambda_d_con < 0
    ):
        raise ValueError("Loss weights and ramp length must be non-negative.")
    if cfg.relation_mode not in {"feature_spaces", "conditional_demographics"}:
        raise ValueError(f"Unknown relation_mode={cfg.relation_mode!r}.")
    if cfg.relation_min_class_samples < 2:
        raise ValueError("relation_min_class_samples must be at least 2.")
    if cfg.relation_mode == "conditional_demographics" and tuple(ATTRIBUTE_NAMES) != (
        "age", "sex", "ethnicity"
    ):
        raise ValueError(
            "conditional_demographics requires Age, Sex, and Ethnicity in that order."
        )
    if cfg.tau <= 0 or cfg.alpha < 0:
        raise ValueError("tau must be positive and alpha non-negative.")
    if not 0 <= cfg.contrastive_initial_scale <= 1:
        raise ValueError("contrastive_initial_scale must be in [0,1].")
    if cfg.contrastive_warmup_epochs <= 0 or cfg.contrastive_candidate_scale < 0:
        raise ValueError("Contrastive warmup must be positive and candidate scale non-negative.")
    if cfg.supcon_positive_weighting not in {"difference_count", "uniform"}:
        raise ValueError("Unknown SupCon positive weighting mode.")
    if cfg.negative_prototypes <= 0 or cfg.positive_prototypes <= 0:
        raise ValueError("Each disease class must have at least one prototype.")
    if cfg.tau_p <= 0 or cfg.sinkhorn_iterations <= 0:
        raise ValueError("Prototype tau and Sinkhorn iterations must be positive.")
    if cfg.sigma <= 0 or cfg.gamma < 0:
        raise ValueError("Invalid prototype-guided alignment kernel sigma or prototype gamma.")
    if min(
        cfg.lambda_alignment, cfg.lambda_sinkhorn, cfg.lambda_pc_mmd,
        cfg.lambda_ema_consistency, cfg.lambda_prototype_demographic,
    ) < 0:
        raise ValueError("Alignment loss weights must be non-negative.")
    if not 0 <= cfg.ema_decay < 1:
        raise ValueError("ema_decay must be in [0,1).")
    if min(
        cfg.alignment_semantic_warmup_epochs, cfg.prototype_warmup_epochs,
        cfg.alignment_ramp_epochs, cfg.mmd_ramp_epochs,
    ) < 0:
        raise ValueError("Alignment schedule lengths must be non-negative.")
    if cfg.auxiliary_update_budget_ratio <= 0:
        raise ValueError("auxiliary_update_budget_ratio must be positive.")
    if not 0 < cfg.train_fraction <= 1:
        raise ValueError("train_fraction must be in (0,1].")


def train(cfg: TrainConfig):
    validate_config(cfg)
    device, rank, world_size = initialize_distributed(cfg.device)
    if cfg.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp now requires a BF16-capable CUDA GPU (A40/A800 supported).")
    is_main = rank == 0
    set_seed(cfg.seed + rank)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    if is_main:
        for directory in (checkpoints_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
    if is_main and not cfg.evaluation_only:
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    **asdict(cfg),
                    "rank": 3,
                    "gradient_control": "multistate_disease_priority_gradient_and_displacement_surgery",
                    "optimizer": "MultiStateDiseasePriorityAdamW",
                    "optimizer_streams": ["disease", "attribute", "contrastive", "row", "relation"],
                    "shared_parameter_writes_per_batch": 1,
                    "weight_decay_applications_per_parameter_per_batch": 1,
                    "final_disease_descent_enforced_before_write": True,
                    "precision": "cuda_bfloat16_autocast_no_grad_scaler" if cfg.amp else "float32",
                    "distributed_world_size": world_size,
                    "per_gpu_batch_size": cfg.batch_size,
                    "effective_global_batch_size": cfg.batch_size * world_size,
                },
                indent=2,
            )
            + "\n"
        )
    if distributed_enabled():
        dist.barrier()

    loaders = make_disease_loaders(
        cfg.metadata_csv, cfg.target, cfg.image_root, cfg.image_size,
        cfg.batch_size, cfg.num_workers, cfg.train_fraction, cfg.subsample_seed,
        cfg.balanced_batches, cfg.normalization_stats,
        dual_view_train=cfg.lambda_ema_consistency > 0,
    )
    loaders = shard_loaders(loaders, cfg, rank, world_size)
    model = DiseaseAttributeModel(
        cfg.dropout, len(ATTRIBUTE_NAMES),
        cfg.negative_prototypes, cfg.positive_prototypes,
    ).to(device)
    broadcast_model_state(model)
    ema_model = make_ema_model(model)
    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    pos_weight = training_pos_weight(loaders["train"], device, cfg.balanced_batches)

    data_summary = {
        split: {
            "subset": len(loader.dataset),
            "full": loader.dataset.full_split_size,
            "metadata_csv": str(loader.dataset.metadata_csv),
            "normalization_statistics": loader.dataset.normalization_statistics,
        }
        for split, loader in loaders.items()
    }
    data_summary["training"] = {
        "bce_pos_weight": float(pos_weight),
        "balanced_metadata": "metadata_defined_splits",
        "additional_batch_balancing": cfg.balanced_batches,
        "decoupling_objective": (
            "lambda_decorr * decorr + lambda_dcor * mean_attribute_label_dcor_conditioned_on_y"
            if cfg.relation_mode == "conditional_demographics"
            else "lambda_decorr * decorr + lambda_dcor * squared_distance_correlation"
        ),
        "relation_mode": cfg.relation_mode,
        "relation_min_class_samples": cfg.relation_min_class_samples,
        "relation_uses_attribute_representation": cfg.relation_mode == "feature_spaces",
        "distance_correlation_attribute_stop_gradient": True,
        "precision": "cuda_bfloat16_autocast_no_grad_scaler" if cfg.amp else "float32",
        "optimizer": "MultiStateDiseasePriorityAdamW",
        "optimizer_streams": ["disease", "attribute", "contrastive", "row", "relation"],
        "contrastive_parameter_scope": ["backbone", "disease_projector"],
        "contrastive_method": "prototype_guided_alignment_ema",
        "prototype_counts": [cfg.negative_prototypes, cfg.positive_prototypes],
        "ema_decay": cfg.ema_decay,
        "dual_view_train": cfg.lambda_ema_consistency > 0,
        "gradient_synchronization": "per_component_before_projection",
        "auxiliary_budget_groups": ["backbone", "disease_projector"],
        "final_disease_descent_enforced_before_write": True,
    }
    data_summary["training"].update(
        {
            "distributed_world_size": world_size,
            "per_gpu_batch_size": cfg.batch_size,
            "effective_global_batch_size": cfg.batch_size * world_size,
            "balanced_batches_are_sharded_whole": cfg.balanced_batches and world_size > 1,
        }
    )
    if is_main:
        (logs_dir / "data_summary.json").write_text(
            json.dumps(data_summary, indent=2) + "\n"
        )

    best_auc, best_seen_auc, best_loss = -math.inf, -math.inf, math.inf
    best_fairness = {"delta_eo": math.inf, "delta_auc": math.inf}
    best_fairness_auc = {"delta_eo": -math.inf, "delta_auc": -math.inf}
    epoch_range = range(0) if cfg.evaluation_only else range(1, cfg.epochs + 1)
    for epoch in epoch_range:
        scale = decorrelation_scale(epoch, cfg)
        alignment_schedule = alignment_scales(epoch, cfg)
        train_result, _ = run_epoch(
            model, loaders["train"], cfg, device, pos_weight, scale, alignment_schedule,
            optimizer, ema_model,
        )
        val_result, val_arrays = run_epoch(
            model, loaders["val"], cfg, device, pos_weight, scale, alignment_schedule
        )
        val_f1_threshold, val_f1_at_selected_threshold = best_f1_threshold(
            val_arrays[0], val_arrays[1]
        )
        epoch_test_result = None
        epoch_test_selected_result = None
        if cfg.test_every_epoch:
            epoch_test_result, epoch_test_arrays = run_epoch(
                model, loaders["test"], cfg, device, pos_weight, scale, alignment_schedule
            )
            epoch_test_selected_result = validation_metrics(
                *epoch_test_arrays, threshold=val_f1_threshold
            )
            if is_main:
                append_csv(
                    logs_dir / "per_epoch_test_metrics.csv",
                    {
                        "epoch": epoch,
                        "AUC": epoch_test_result["auc"],
                        "ACC": epoch_test_result["acc"],
                        "F1": epoch_test_result["f1"],
                        "delta_EO": epoch_test_result["delta_eo"],
                        "delta_AUC": epoch_test_result["delta_auc"],
                        "val_best_f1_threshold": val_f1_threshold,
                        "val_f1_at_selected_threshold": val_f1_at_selected_threshold,
                        "selected_ACC": epoch_test_selected_result["acc"],
                        "selected_AUC": epoch_test_selected_result["auc"],
                        "selected_F1": epoch_test_selected_result["f1"],
                        "selected_delta_EO": epoch_test_selected_result["delta_eo"],
                        "selected_delta_AUC": epoch_test_selected_result["delta_auc"],
                    },
                    epoch == 1,
                )

        row = {
            "epoch": epoch,
            "stage": "semantic_warmup" if scale == 0 else "alignment_and_decorrelation",
            "decorrelation_scale": scale,
            **{f"alignment_{name}_scale": value for name, value in alignment_schedule.items()},
            "effective_contrastive_candidate_scale": (
                cfg.contrastive_candidate_scale * max(alignment_schedule.values())
            ),
            "effective_lambda_decorr": scale * cfg.lambda_decorr,
            "effective_lambda_dcor": scale * cfg.lambda_dcor,
            "learning_rate_heads": optimizer.param_groups[0]["lr"],
            "learning_rate_backbone": optimizer.param_groups[1]["lr"],
            "val_best_f1_threshold": val_f1_threshold,
            "val_f1_at_selected_threshold": val_f1_at_selected_threshold,
            **{f"train_{key}": value for key, value in train_result.items()},
            **{f"val_{key}": value for key, value in val_result.items()},
        }
        if epoch_test_result is not None:
            row.update(
                {
                    "test_auc": epoch_test_result["auc"],
                    "test_acc": epoch_test_result["acc"],
                    "test_f1": epoch_test_result["f1"],
                    "test_delta_eo": epoch_test_result["delta_eo"],
                    "test_delta_auc": epoch_test_result["delta_auc"],
                    "test_val_best_f1_acc": epoch_test_selected_result["acc"],
                    "test_val_best_f1_auc": epoch_test_selected_result["auc"],
                    "test_val_best_f1_f1": epoch_test_selected_result["f1"],
                    "test_val_best_f1_delta_eo": epoch_test_selected_result["delta_eo"],
                    "test_val_best_f1_delta_auc": epoch_test_selected_result["delta_auc"],
                }
            )
        if is_main:
            append_csv(logs_dir / "training_log.csv", row, epoch == 1)
        scheduler.step()
        if is_main and cfg.save_last_checkpoint:
            save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, scheduler, epoch, row, cfg, ema_model)

        if np.isfinite(row["val_auc"]):
            best_seen_auc = max(best_seen_auc, row["val_auc"])
        # Select loss checkpoint using semantic validation loss so the changing
        # relation ramp cannot make epochs incomparable.
        if np.isfinite(row["val_semantic"]) and row["val_semantic"] < best_loss:
            best_loss = row["val_semantic"]
            if is_main:
                save_checkpoint(checkpoints_dir / "best_val_loss.pt", model, optimizer, scheduler, epoch, row, cfg, ema_model)
        if np.isfinite(row["val_auc"]) and row["val_auc"] > best_auc:
            best_auc = row["val_auc"]
            if is_main:
                save_checkpoint(checkpoints_dir / "best_val_auc.pt", model, optimizer, scheduler, epoch, row, cfg, ema_model)
        for metric in ("delta_eo", "delta_auc"):
            value = row[f"val_{metric}"]
            utility_ok = cfg.independent_fairness_checkpoints or (
                np.isfinite(row["val_auc"])
                and row["val_auc"] >= best_seen_auc - cfg.utility_checkpoint_tolerance
            )
            if utility_ok and np.isfinite(value) and value < best_fairness[metric]:
                best_fairness[metric] = value
                best_fairness_auc[metric] = row["val_auc"]
                if is_main:
                    save_checkpoint(checkpoints_dir / f"best_val_{metric}.pt", model, optimizer, scheduler, epoch, row, cfg, ema_model)

        if is_main:
            print(
                f"[epoch {epoch:03d}/{cfg.epochs}] val_auc={row['val_auc']:.5f} "
                f"semantic={row['val_semantic']:.5f} decorr={row['train_decorr']:.5f} "
                f"dcor={row['train_dcor']:.5f} "
                f"gshare=D:{row['train_disease_gradient_share']:.3f}/"
                f"A:{row['train_attribute_gradient_share']:.3f}/"
                f"Con:{row['train_contrastive_gradient_share']:.3f}/"
                f"R:{row['train_row_gradient_share']:.3f}/"
                f"Rel:{row['train_relation_gradient_share']:.3f} "
                f"decorrelation_scale={scale:.3f} alignment_schedule={alignment_schedule['contrastive']:.3f} "
                f"conflicts=A:{row['train_attribute_gradient_conflict']:.3f}/"
                f"Con:{row['train_contrastive_gradient_conflict']:.3f}/"
                f"R:{row['train_row_gradient_conflict']:.3f}/"
                f"Rel:{row['train_relation_gradient_conflict']:.3f}",
                flush=True,
            )

    if is_main and not cfg.save_last_checkpoint:
        last_checkpoint = checkpoints_dir / "last.pt"
        if last_checkpoint.exists():
            last_checkpoint.unlink()

    if distributed_enabled():
        dist.barrier()

    evaluation = {}
    comparison = []
    evaluation_alignment_schedule = alignment_scales(cfg.epochs, cfg)
    for name in EVALUATION_CHECKPOINTS:
        path = checkpoints_dir / f"{name}.pt"
        if not path.exists():
            raise RuntimeError(f"Required checkpoint missing: {path}")
        checkpoint_model, checkpoint = load_model(path, device)
        _, val_arrays = run_epoch(
            checkpoint_model, loaders["val"], cfg, device, pos_weight, 1.0,
            evaluation_alignment_schedule,
        )
        selected_threshold, selected_val_f1 = best_f1_threshold(
            val_arrays[0], val_arrays[1]
        )
        _, arrays = run_epoch(
            checkpoint_model, loaders["test"], cfg, device, pos_weight, 1.0,
            evaluation_alignment_schedule,
        )
        labels, scores, attributes = arrays
        fixed_metrics = validation_metrics(labels, scores, attributes, threshold=0.5)
        selected_metrics = validation_metrics(
            labels, scores, attributes, threshold=selected_threshold
        )
        target_dir = output_dir / "evaluation" / name
        payload = {
            "checkpoint": str(path), "epoch": checkpoint["epoch"],
            "validation_metrics": checkpoint["metrics"],
            "thresholds": {
                "fixed_0p5": 0.5,
                "val_best_f1": selected_threshold,
                "val_f1_at_selected_threshold": selected_val_f1,
            },
            # Backward-compatible alias: legacy consumers use fixed threshold 0.5.
            "test_metrics": fixed_metrics,
            "test_metrics_fixed_0p5": fixed_metrics,
            "test_metrics_val_best_f1": selected_metrics,
            "rank": 3,
        }
        if is_main:
            # Preserve the legacy files as the fixed-0.5 evaluation.
            save_result_tables(target_dir, labels, scores, attributes, cfg.target, threshold=0.5)
            save_result_tables(
                target_dir / "fixed_0p5",
                labels, scores, attributes, cfg.target, threshold=0.5,
            )
            save_result_tables(
                target_dir / "val_best_f1",
                labels, scores, attributes, cfg.target, threshold=selected_threshold,
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "checkpoint_evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
        evaluation[name] = payload
        comparison.append({
            "task": cfg.target, "checkpoint": name, "epoch": checkpoint["epoch"],
            "threshold_mode": "fixed_0p5", "threshold": 0.5,
            "ACC": fixed_metrics["acc"], "AUC": fixed_metrics["auc"],
            "F1": fixed_metrics["f1"], "delta_EO": fixed_metrics["delta_eo"],
            "delta_AUC": fixed_metrics["delta_auc"],
        })
        comparison.append({
            "task": cfg.target, "checkpoint": name, "epoch": checkpoint["epoch"],
            "threshold_mode": "val_best_f1", "threshold": selected_threshold,
            "ACC": selected_metrics["acc"], "AUC": selected_metrics["auc"],
            "F1": selected_metrics["f1"], "delta_EO": selected_metrics["delta_eo"],
            "delta_AUC": selected_metrics["delta_auc"],
        })
        if is_main and name == "best_val_auc":
            save_result_tables(output_dir, labels, scores, attributes, cfg.target, threshold=0.5)
            save_detailed_fairness_metrics(
                output_dir,
                labels,
                scores,
                attributes,
                cfg.target,
                checkpoint=name,
                epoch=checkpoint["epoch"],
                threshold=0.5,
            )
            save_detailed_fairness_metrics(
                output_dir / "fixed_0p5",
                labels, scores, attributes, cfg.target,
                checkpoint=name, epoch=checkpoint["epoch"], threshold=0.5,
            )
            save_detailed_fairness_metrics(
                output_dir / "val_best_f1",
                labels, scores, attributes, cfg.target,
                checkpoint=name, epoch=checkpoint["epoch"],
                threshold=selected_threshold,
            )
            (output_dir / "test_fairness_summary.json").write_text(
                json.dumps(
                    {
                        "thresholds": payload["thresholds"],
                        "fixed_0p5": fixed_metrics,
                        "val_best_f1": selected_metrics,
                    },
                    indent=2,
                ) + "\n"
            )

    if is_main:
        fixed_comparison = [
            {
                key: value for key, value in row.items()
                if key not in ("threshold_mode", "threshold")
            }
            for row in comparison if row["threshold_mode"] == "fixed_0p5"
        ]
        (output_dir / "checkpoint_selection.json").write_text(json.dumps(evaluation, indent=2) + "\n")
        (output_dir / "four_checkpoint_test_summary.json").write_text(
            json.dumps(fixed_comparison, indent=2) + "\n"
        )
        with (output_dir / "four_checkpoint_test_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fixed_comparison[0]))
            writer.writeheader()
            writer.writerows(fixed_comparison)
        (output_dir / "four_checkpoint_test_summary_by_threshold.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        with (output_dir / "four_checkpoint_test_summary_by_threshold.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        (output_dir / "run_complete.json").write_text(
            json.dumps(
                {
                    "task": cfg.target,
                    "epochs": cfg.epochs,
                    "rank": 3,
                    "distributed_world_size": world_size,
                    "status": "complete",
                },
                indent=2,
            )
            + "\n"
        )
    if distributed_enabled():
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TASKS)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="CSV containing image paths, data splits, task labels, and sensitive attributes.",
    )
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--normalization_stats", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--semantic_warmup_epochs", type=int, default=5)
    parser.add_argument("--relation_ramp_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--probe_feature_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda_a", type=float, default=0.25)
    parser.add_argument("--lambda_d_con", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--supcon_positive_weighting",
        choices=("difference_count", "uniform"),
        default="difference_count",
    )
    parser.add_argument("--contrastive_initial_scale", type=float, default=0.1)
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--contrastive_candidate_scale", type=float, default=0.05,
        help="Post-Adam scale for the contrastive candidate displacement.",
    )
    parser.add_argument("--negative_prototypes", type=int, default=3)
    parser.add_argument("--positive_prototypes", type=int, default=3)
    parser.add_argument("--assignment_temperature", type=float, default=None)
    parser.add_argument("--prototype_candidate_scale", type=float, default=None)
    parser.add_argument("--ema_delay_epochs", type=int, default=0)
    parser.add_argument("--protect_decorrelation_budget", action="store_true")
    parser.add_argument(
        "--independent_contrastive_budget_ratio", type=float, default=None,
        help="Separate contrastive cap relative to the disease step in each protected module; requires --protect_decorrelation_budget.",
    )
    parser.add_argument("--tau_p", type=float, default=0.2)
    parser.add_argument("--sinkhorn_iterations", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--lambda_alignment", type=float, default=1.0)
    parser.add_argument("--lambda_sinkhorn", type=float, default=0.1)
    parser.add_argument("--lambda_pc_mmd", type=float, default=0.1)
    parser.add_argument("--lambda_ema_consistency", type=float, default=0.1)
    parser.add_argument("--lambda_prototype_demographic", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--mmd_min_effective_weight", type=float, default=0.5)
    parser.add_argument("--alignment_semantic_warmup_epochs", type=int, default=5)
    parser.add_argument("--prototype_warmup_epochs", type=int, default=2)
    parser.add_argument("--alignment_ramp_epochs", type=int, default=5)
    parser.add_argument("--mmd_ramp_epochs", type=int, default=10)
    parser.add_argument(
        "--disable_prototype_weights", dest="use_prototype_weights",
        action="store_false",
    )
    parser.add_argument(
        "--disable_attribute_kernel", dest="use_attribute_kernel",
        action="store_false",
    )
    parser.set_defaults(use_prototype_weights=True, use_attribute_kernel=True)
    parser.add_argument("--lambda_decorr", type=float, default=500.0)
    parser.add_argument("--lambda_dcor", type=float, default=1.0)
    parser.add_argument(
        "--relation_mode",
        choices=("feature_spaces", "conditional_demographics"),
        default="feature_spaces",
        help=(
            "feature_spaces compares disease and attribute representations; conditional_demographics "
            "and averages Age/Sex/Ethnicity label dCor separately within y=0 and y=1."
        ),
    )
    parser.add_argument("--relation_min_class_samples", type=int, default=4)
    parser.add_argument(
        "--auxiliary_update_budget_ratio",
        "--auxiliary_gradient_max_ratio",
        "--relation_gradient_max_ratio",
        dest="auxiliary_update_budget_ratio", type=float, default=1.0,
        help=(
            "Cap the combined auxiliary AdamW candidate displacement relative to the "
            "disease candidate, separately on backbone and disease projector."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--amp", action="store_true",
        help="Use CUDA BF16 autocast without GradScaler (A40/A800).",
    )
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--subsample_seed", type=int, default=0)
    parser.add_argument("--balanced_batches", action="store_true")
    parser.add_argument("--utility_checkpoint_tolerance", type=float, default=0.005)
    parser.add_argument("--independent_fairness_checkpoints", action="store_true")
    parser.add_argument("--test_every_epoch", action="store_true")
    parser.add_argument(
        "--evaluation_only",
        action="store_true",
        help="Skip training and evaluate the four saved checkpoints in output_dir.",
    )
    parser.add_argument(
        "--no_save_last_checkpoint",
        dest="save_last_checkpoint",
        action="store_false",
    )
    parser.set_defaults(save_last_checkpoint=True)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
