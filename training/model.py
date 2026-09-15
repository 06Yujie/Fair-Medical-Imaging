"""Shared DenseNet-121 encoder, disease and attribute MLPs, and class prototypes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DiseaseAttributeModel(nn.Module):
    """Shared DenseNet with independent, normalized 128-D semantic spaces."""

    feature_dim = 1024
    representation_dim = 128
    subspace_rank = 3

    def __init__(
        self,
        dropout: float = 0.2,
        num_attributes: int = 3,
        negative_prototypes: int = 3,
        positive_prototypes: int = 3,
    ) -> None:
        super().__init__()
        backbone = models.densenet121(weights=None)
        if backbone.classifier.in_features != self.feature_dim:
            raise RuntimeError("Unexpected DenseNet121 feature dimension.")
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.attribute_projector = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.representation_dim),
        )
        self.disease_projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.representation_dim),
        )
        self.attribute_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(self.representation_dim, 2)
                )
                for _ in range(num_attributes)
            ]
        )
        self.disease_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.representation_dim, 1)
        )
        if negative_prototypes <= 0 or positive_prototypes <= 0:
            raise ValueError("Each disease class must have at least one prototype.")
        self.prototype_counts = (negative_prototypes, positive_prototypes)
        self.class_prototypes = nn.ParameterList(
            [
                nn.Parameter(torch.randn(negative_prototypes, self.representation_dim)),
                nn.Parameter(torch.randn(positive_prototypes, self.representation_dim)),
            ]
        )

        # Buffers make the detached, one-epoch-lagged bases checkpoint-safe.
        self.register_buffer(
            "attribute_basis_y0",
            torch.zeros(self.representation_dim, self.subspace_rank),
        )
        self.register_buffer(
            "attribute_basis_y1",
            torch.zeros(self.representation_dim, self.subspace_rank),
        )
        self.register_buffer("basis_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("basis_source_epoch", torch.tensor(0, dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in list(state_dict):
            old_prefix = prefix + "phenotype_prototypes."
            if key.startswith(old_prefix):
                new_key = prefix + "class_prototypes." + key[len(old_prefix):]
                state_dict.setdefault(new_key, state_dict.pop(key))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def encode(self, images: torch.Tensor):
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(f"Expected [B,1024], got {tuple(features.shape)}.")
        z_a = F.normalize(
            self.attribute_projector(features).float(), dim=1, eps=1e-12
        )
        z_d = F.normalize(
            self.disease_projector(features).float(), dim=1, eps=1e-12
        )
        return features, z_a, z_d

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features, z_a, z_d = self.encode(images)
        result = {
            "features": features,
            "z_a": z_a,
            "z_d": z_d,
            "attribute_logits": torch.stack(
                [head(z_a) for head in self.attribute_heads], dim=1
            ),
            "disease_logits": self.disease_head(z_d).squeeze(-1),
        }
        result["normalized_prototypes"] = self.normalized_prototypes()
        result["prototype_mask"] = self.prototype_mask(z_d.device)
        return result

    def normalized_prototypes(self) -> torch.Tensor:
        normalized = [
            F.normalize(value.float(), dim=1, eps=1e-12)
            for value in self.class_prototypes
        ]
        maximum = max(self.prototype_counts)
        padded = normalized[0].new_zeros(
            (2, maximum, self.representation_dim)
        )
        for disease_class, value in enumerate(normalized):
            padded[disease_class, : value.shape[0]] = value
        return padded

    def prototype_mask(self, device: torch.device | None = None) -> torch.Tensor:
        maximum = max(self.prototype_counts)
        target_device = device or next(self.parameters()).device
        indices = torch.arange(maximum, device=target_device)
        counts = torch.tensor(self.prototype_counts, device=target_device)
        return indices[None, :] < counts[:, None]

    def bases(self) -> dict[int, torch.Tensor]:
        return {0: self.attribute_basis_y0, 1: self.attribute_basis_y1}

    @torch.no_grad()
    def set_bases(self, bases: dict[int, torch.Tensor], source_epoch: int) -> None:
        expected = (self.representation_dim, self.subspace_rank)
        for disease_class, destination in (
            (0, self.attribute_basis_y0),
            (1, self.attribute_basis_y1),
        ):
            value = bases[disease_class]
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Basis y={disease_class} must be {expected}, got {tuple(value.shape)}."
                )
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        self.basis_source_epoch.fill_(int(source_epoch))
        self.basis_ready.fill_(True)
