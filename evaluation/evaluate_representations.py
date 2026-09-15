#!/usr/bin/env python3
"""Four-direction representation evaluation for one trained checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from training.data import ATTRIBUTE_NAMES, make_disease_loaders
from .metrics import METRICS, task_metrics

from training.model import DiseaseAttributeModel
from training.train import TrainConfig, make_probe_train_loader, normalize_config_keys


class BinaryProbe(nn.Module):
    """Two-layer MLP probe for a frozen representation."""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden = min(256, max(64, input_dim // 2))
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def binary_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)),
    }


def load_config(task_dir: Path) -> TrainConfig:
    payload = normalize_config_keys(json.loads((task_dir / "config.json").read_text()))
    known = {field.name for field in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in payload.items() if key in known})


def load_model(checkpoint_path: Path, cfg: TrainConfig, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DiseaseAttributeModel(cfg.dropout, len(ATTRIBUTE_NAMES), cfg.negative_prototypes, cfg.positive_prototypes).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, checkpoint


@torch.inference_mode()
def extract(model, loader, device, use_bf16):
    disease_labels, attribute_labels = [], []
    z_attribute, z_disease = [], []
    disease_scores, attribute_head_scores = [], []
    for images, disease, attributes in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            outputs = model(images)
        disease_labels.append(disease.numpy())
        attribute_labels.append(attributes.numpy())
        z_attribute.append(outputs["z_a"].float().cpu().numpy())
        z_disease.append(outputs["z_d"].float().cpu().numpy())
        disease_scores.append(
            torch.sigmoid(outputs["disease_logits"].float()).cpu().numpy()
        )
        attribute_head_scores.append(
            torch.softmax(outputs["attribute_logits"].float(), dim=-1)[..., 1]
            .cpu()
            .numpy()
        )
    return {
        "disease": np.concatenate(disease_labels).astype(np.int64),
        "attributes": np.concatenate(attribute_labels).astype(np.int64),
        "z_a": np.concatenate(z_attribute).astype(np.float32),
        "z_d": np.concatenate(z_disease).astype(np.float32),
        "disease_head_score": np.concatenate(disease_scores).astype(np.float32),
        "attribute_head_scores": np.concatenate(attribute_head_scores).astype(np.float32),
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict_probe(probe, features, device, batch_size):
    outputs = []
    probe.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        outputs.append(probe(batch).cpu())
    return torch.cat(outputs).numpy()


def fit_probe(train_x, train_y, val_x, val_y, args, device, seed):
    """Match AUC.evaluate_balanced_representation_probes.train_probe exactly."""
    seed_everything(seed)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    train_x = ((train_x - mean) / std).astype(np.float32)
    val_x = ((val_x - mean) / std).astype(np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    val_y = np.asarray(val_y, dtype=np.float32)

    probe = BinaryProbe(train_x.shape[1]).to(device)
    optimizer = AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=generator,
    )
    val_features = torch.from_numpy(val_x)
    val_labels = torch.from_numpy(val_y)
    best_loss, best_epoch, best_state = math.inf, None, None
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features.to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_labels.to(device)
            )
            loss.backward()
            optimizer.step()
        val_logits = predict_probe(
            probe, val_features.numpy(), device, args.probe_batch_size
        )
        val_loss = float(
            F.binary_cross_entropy_with_logits(
                torch.from_numpy(val_logits), val_labels
            )
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(probe.state_dict())
    if best_state is None:
        raise RuntimeError("Probe training did not produce a checkpoint.")
    probe.load_state_dict(best_state)
    return probe, mean, std, {"probe_epoch": best_epoch, "best_val_loss": best_loss}


def probe_direction(
    train, val, test, representation, target, target_name, args, device, seed
):
    train_y = train[target] if target == "disease" else train["attributes"][:, target]
    val_y = val[target] if target == "disease" else val["attributes"][:, target]
    test_y = test[target] if target == "disease" else test["attributes"][:, target]
    probe, mean, std, selection = fit_probe(
        train[representation], train_y, val[representation], val_y,
        args, device, seed,
    )
    test_x = ((test[representation] - mean) / std).astype(np.float32)
    test_logits = predict_probe(probe, test_x, device, args.probe_batch_size)
    test_score = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -50, 50)))
    return {
        "target": target_name,
        **selection,
        "test": binary_metrics(test_y, test_score),
    }, test_score


def marginal_disease_fairness(labels, scores, attributes):
    result = {}
    subgroup_rows = []
    for index, attribute_name in enumerate(ATTRIBUTE_NAMES):
        groups = {}
        for group in (0, 1):
            mask = attributes[:, index] == group
            metrics = task_metrics(labels[mask], scores[mask], threshold=0.5)
            groups[str(group)] = {"n": int(mask.sum()), **metrics}
            subgroup_rows.append(
                {"attribute": attribute_name, "group": group, "n": int(mask.sum()), **metrics}
            )
        left, right = groups["0"], groups["1"]
        delta_tpr = abs((1.0 - left["fnr"]) - (1.0 - right["fnr"]))
        delta_fpr = abs(left["fpr"] - right["fpr"])
        result[attribute_name] = {
            "groups": groups,
            "gaps": {
                "delta_acc": abs(left["acc"] - right["acc"]),
                "delta_f1": abs(left["f1"] - right["f1"]),
                "delta_auc": abs(left["auc"] - right["auc"]),
                "delta_tpr": delta_tpr,
                "delta_fpr": delta_fpr,
                "delta_eo": max(delta_tpr, delta_fpr),
            },
        }
    return result, subgroup_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--normalization_stats", default=None)
    parser.add_argument("--checkpoint", default="best_val_auc")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    task_dir = Path(args.task_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else task_dir / "four_direction_evaluation" / args.checkpoint
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(task_dir)
    for name in ("metadata_csv", "image_root", "normalization_stats"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    cfg.num_workers = args.num_workers
    cfg.probe_feature_batch_size = args.batch_size
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_path = task_dir / "checkpoints" / f"{args.checkpoint}.pt"
    model, checkpoint = load_model(checkpoint_path, cfg, device)

    loaders = make_disease_loaders(
        cfg.metadata_csv,
        cfg.target,
        cfg.image_root,
        cfg.image_size,
        args.batch_size,
        args.num_workers,
        1.0,
        cfg.subsample_seed,
        False,
        cfg.normalization_stats,
    )
    # Probe training must use deterministic validation preprocessing, not augmentation.
    train_loader = make_probe_train_loader(loaders["train"], cfg)
    split_loaders = {"train": train_loader, "val": loaders["val"], "test": loaders["test"]}
    arrays = {}
    for split, loader in split_loaders.items():
        print(f"Extracting {split}: {len(loader.dataset)} samples", flush=True)
        arrays[split] = extract(model, loader, device, cfg.amp)
    np.savez_compressed(
        output_dir / "frozen_embeddings_and_scores.npz",
        **{
            f"{split}_{name}": value
            for split, values in arrays.items()
            for name, value in values.items()
        },
    )

    train, val, test = arrays["train"], arrays["val"], arrays["test"]
    za_attributes, za_attribute_scores = {}, {}
    zd_attributes, zd_attribute_scores = {}, {}
    for index, name in enumerate(ATTRIBUTE_NAMES):
        za_attributes[name], za_attribute_scores[name] = probe_direction(
            train, val, test, "z_a", index, name, args, device, args.seed + index
        )
        zd_attributes[name], zd_attribute_scores[name] = probe_direction(
            train, val, test, "z_d", index, name, args, device, args.seed + 4 + index
        )
    za_disease, za_disease_score = probe_direction(
        train, val, test, "z_a", "disease", "disease", args, device, args.seed + 3
    )
    zd_disease, zd_disease_score = probe_direction(
        train, val, test, "z_d", "disease", "disease", args, device, args.seed + 7
    )

    trained_attribute_heads = {
        name: binary_metrics(
            test["attributes"][:, index], test["attribute_head_scores"][:, index]
        )
        for index, name in enumerate(ATTRIBUTE_NAMES)
    }
    disease_head = binary_metrics(test["disease"], test["disease_head_score"])
    probe_marginal, probe_subgroup_rows = marginal_disease_fairness(
        test["disease"], zd_disease_score, test["attributes"]
    )
    native_marginal, native_subgroup_rows = marginal_disease_fairness(
        test["disease"], test["disease_head_score"], test["attributes"]
    )
    payload = {
        "task": cfg.target,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_path": str(checkpoint_path),
        "test_samples": int(len(test["disease"])),
        "protocol": {
            "representations": "frozen 128-D normalized z_a and z_d",
            "probe_train_split": "train with deterministic evaluation preprocessing",
            "probe": "LayerNorm -> Linear -> GELU -> Linear",
            "model_selection": "epoch selected on validation BCE",
            "test_usage": "test used once for final ACC/F1/AUC",
            "probe_epochs": args.probe_epochs,
            "probe_batch_size": args.probe_batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "attribute_space_to_attributes": {
            "mlp_probes": za_attributes,
            "trained_attribute_heads_reference": trained_attribute_heads,
        },
        "attribute_space_to_disease": za_disease,
        "disease_space_to_disease": {
            "mlp_probe": zd_disease,
            "mlp_probe_marginal_fairness": probe_marginal,
            "trained_disease_head_reference": disease_head,
            "trained_disease_head_marginal_fairness": native_marginal,
        },
        "disease_space_to_attributes": zd_attributes,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    summary_rows = []
    for direction, values in (
        ("z_a_to_attribute_mlp_probe", za_attributes),
        ("z_a_to_attribute_trained_head", {k: {"test": v} for k, v in trained_attribute_heads.items()}),
        ("z_d_to_attribute_mlp_probe", zd_attributes),
    ):
        for target, target_values in values.items():
            summary_rows.append({"direction": direction, "target": target, **target_values["test"]})
    summary_rows.append(
        {"direction": "z_a_to_disease_mlp_probe", "target": "disease", **za_disease["test"]}
    )
    summary_rows.append(
        {"direction": "z_d_to_disease_mlp_probe", "target": "disease", **zd_disease["test"]}
    )
    summary_rows.append(
        {"direction": "z_d_to_disease_trained_head_reference", "target": "disease", **disease_head}
    )
    with (output_dir / "direction_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("direction", "target", "acc", "f1", "auc"))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "disease_marginal_subgroup_metrics.csv").open("w", newline="") as handle:
        fieldnames = ("source", "attribute", "group", "n", *METRICS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {"source": "z_d_mlp_probe", **row} for row in probe_subgroup_rows
        )
        writer.writerows(
            {"source": "trained_disease_head_reference", **row}
            for row in native_subgroup_rows
        )

    prediction_columns = {
        "disease_label": test["disease"],
        "disease_head_score": test["disease_head_score"],
        "za_disease_probe_score": za_disease_score,
        "zd_disease_probe_score": zd_disease_score,
    }
    for index, name in enumerate(ATTRIBUTE_NAMES):
        prediction_columns[f"{name}_label"] = test["attributes"][:, index]
        prediction_columns[f"za_{name}_head_score"] = test["attribute_head_scores"][:, index]
        prediction_columns[f"za_{name}_probe_score"] = za_attribute_scores[name]
        prediction_columns[f"zd_{name}_probe_score"] = zd_attribute_scores[name]
    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(prediction_columns)
        writer.writerows(zip(*prediction_columns.values()))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
