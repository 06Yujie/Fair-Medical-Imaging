"""Classification, subgroup fairness, FATE, and DRAR metrics and CLI evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


METRICS = ("f1", "acc", "auc", "fpr", "fnr", "bacc", "macro_f1")
INTERSECTION_ATTRIBUTE = "sex_age_ethnicity_binary"
DETAILED_METRIC_COLUMNS = (
    "Task",
    "Checkpoint",
    "Epoch",
    "Test ACC",
    "Test AUC",
    "Test F1",
    "Age ΔEO",
    "Age ΔAUC",
    "Sex ΔEO",
    "Sex ΔAUC",
    "Ethnicity ΔEO",
    "Ethnicity ΔAUC",
    "交叉 ΔEO",
    "交叉 ΔAUC",
)


def safe_auc(labels, scores) -> float:
    labels = np.asarray(labels).astype(int)
    return (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )


def task_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "bacc": float(0.5 * (tp / (tp + fn) + tn / (tn + fp)))
        if tp + fn and tn + fp else float("nan"),
        "acc": float(accuracy_score(labels, predictions)),
        "auc": safe_auc(labels, scores),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "fnr": float(fn / (fn + tp)) if fn + tp else float("nan"),
    }


def best_f1_threshold(labels, scores) -> tuple[float, float]:
    """Select a score threshold that maximizes F1 on the supplied split.

    Scores tied at the threshold are handled together, matching the ``>=``
    convention in :func:`task_metrics`.  If several thresholds have the same
    best F1, prefer the one closest to 0.5 for deterministic, conservative
    tie-breaking.
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    tp = cumulative_tp[group_ends].astype(float)
    predicted_positive = (group_ends + 1).astype(float)
    fp = predicted_positive - tp
    fn = float((labels == 1).sum()) - tp
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )
    thresholds = sorted_scores[group_ends]
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0.0, atol=1e-12))
    chosen = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(thresholds[chosen]), float(f1[chosen])


def subgroup_name(attribute_row) -> str:
    values = tuple(int(value) for value in attribute_row)
    if len(values) == 1:
        return "age>=60" if values[0] else "age<60"
    age, sex, ethnicity = values
    return (
        f"{'male' if sex else 'female'}|{'age>=60' if age else 'age<60'}|"
        f"{'non-white' if ethnicity else 'white'}"
    )


def result_tables(labels, scores, attributes, target: str, threshold: float = 0.5):
    labels, scores = np.asarray(labels), np.asarray(scores)
    attributes = np.asarray(attributes).astype(int)
    overall = pd.DataFrame(
        [{"task": target, **task_metrics(labels, scores, threshold)}]
    )
    names = np.asarray([subgroup_name(row) for row in attributes], dtype=object)
    attribute_name = "age_binary" if attributes.shape[1] == 1 else INTERSECTION_ATTRIBUTE
    rows = []
    for name in sorted(np.unique(names)):
        mask = names == name
        rows.append(
            {
                "attribute": attribute_name,
                "subgroup": name,
                "task": target,
                "n": int(mask.sum()),
                **task_metrics(labels[mask], scores[mask], threshold),
            }
        )
    return overall, pd.DataFrame(rows)


def validation_metrics(labels, scores, attributes, threshold: float = 0.5):
    overall, subgroup = result_tables(labels, scores, attributes, "target", threshold)
    result = {key: float(overall.iloc[0][key]) for key in METRICS}
    finite_auc = subgroup["auc"].dropna()
    tpr = 1.0 - subgroup["fnr"]
    tpr_gap = (
        float(tpr.max() - tpr.min()) if tpr.notna().sum() >= 2 else float("nan")
    )
    fpr_gap = (
        float(subgroup["fpr"].max() - subgroup["fpr"].min())
        if subgroup["fpr"].notna().sum() >= 2
        else float("nan")
    )
    result["delta_auc"] = (
        float(finite_auc.max() - finite_auc.min())
        if len(finite_auc) >= 2
        else float("nan")
    )
    result["delta_tpr"] = tpr_gap
    result["delta_fpr"] = fpr_gap
    finite_eo = [value for value in (tpr_gap, fpr_gap) if np.isfinite(value)]
    result["delta_eo"] = max(finite_eo) if finite_eo else float("nan")
    return result


def _group_fairness_rows(labels, scores, groups, attribute: str, threshold: float):
    rows = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        rows.append(
            {
                "attribute": attribute,
                "subgroup": int(group),
                "n": int(mask.sum()),
                **task_metrics(labels[mask], scores[mask], threshold),
            }
        )
    return rows


def _fairness_gaps(rows) -> tuple[float, float]:
    auc_values = np.asarray([row["auc"] for row in rows], dtype=float)
    tpr_values = 1.0 - np.asarray([row["fnr"] for row in rows], dtype=float)
    fpr_values = np.asarray([row["fpr"] for row in rows], dtype=float)

    def finite_range(values) -> float:
        values = values[np.isfinite(values)]
        return float(values.max() - values.min()) if values.size >= 2 else float("nan")

    tpr_gap = finite_range(tpr_values)
    fpr_gap = finite_range(fpr_values)
    finite_eo = [value for value in (tpr_gap, fpr_gap) if np.isfinite(value)]
    delta_eo = max(finite_eo) if finite_eo else float("nan")
    return delta_eo, finite_range(auc_values)


def detailed_fairness_metrics(
    labels,
    scores,
    attributes,
    target: str,
    checkpoint: str,
    epoch: int,
    threshold: float = 0.5,
):
    """Return fixed-threshold marginal and intersectional fairness metrics."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    attributes = np.asarray(attributes).astype(int)
    if attributes.ndim != 2 or attributes.shape[1] != 3:
        raise ValueError(
            "Detailed fairness reporting requires attributes ordered as age, sex, ethnicity."
        )

    overall = task_metrics(labels, scores, threshold)
    row = {
        "Task": target,
        "Checkpoint": checkpoint,
        "Epoch": int(epoch),
        "Test ACC": overall["acc"],
        "Test AUC": overall["auc"],
        "Test F1": overall["f1"],
    }
    subgroup_rows = []
    for index, display_name in enumerate(("Age", "Sex", "Ethnicity")):
        rows = _group_fairness_rows(
            labels, scores, attributes[:, index], display_name, threshold
        )
        subgroup_rows.extend(rows)
        row[f"{display_name} ΔEO"], row[f"{display_name} ΔAUC"] = (
            _fairness_gaps(rows)
        )

    # Binary code in age-sex-ethnicity order; all observed intersectional groups
    # participate in the max-min gaps.
    intersection = (
        attributes[:, 0] * 4 + attributes[:, 1] * 2 + attributes[:, 2]
    )
    rows = _group_fairness_rows(labels, scores, intersection, "交叉", threshold)
    subgroup_rows.extend(rows)
    row["交叉 ΔEO"], row["交叉 ΔAUC"] = _fairness_gaps(rows)
    return row, pd.DataFrame(subgroup_rows)


def save_detailed_fairness_metrics(
    output_dir: str | Path,
    labels,
    scores,
    attributes,
    target: str,
    checkpoint: str,
    epoch: int,
    threshold: float = 0.5,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row, subgroup = detailed_fairness_metrics(
        labels, scores, attributes, target, checkpoint, epoch, threshold
    )
    pd.DataFrame([row], columns=DETAILED_METRIC_COLUMNS).to_csv(
        output_dir / "best_val_auc_test_metrics.csv", index=False
    )
    pd.Series(row).to_json(
        output_dir / "best_val_auc_test_metrics.json", indent=2, force_ascii=False
    )
    subgroup.to_csv(
        output_dir / "best_val_auc_test_subgroup_metrics.csv", index=False
    )


def save_result_tables(
    output_dir: str | Path,
    labels,
    scores,
    attributes,
    target: str,
    threshold: float = 0.5,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall, subgroup = result_tables(labels, scores, attributes, target, threshold)
    overall.to_csv(output_dir / "test_overall_metrics.csv", index=False)
    subgroup.to_csv(output_dir / "test_subgroup_metrics.csv", index=False)
    gaps = (
        subgroup.groupby(["attribute", "task"], as_index=False)[list(METRICS)]
        .agg(lambda values: values.max() - values.min())
        .rename(columns={name: f"{name}_gap" for name in METRICS})
    )
    gaps.to_csv(output_dir / "gap_table.csv", index=False)
    minus = subgroup[["attribute", "subgroup", "task", "n"]].copy()
    for metric in METRICS:
        means = subgroup.groupby(["attribute", "task"])[metric].transform("mean")
        minus[f"{metric}_minus_mean"] = subgroup[metric] - means
    minus.to_csv(output_dir / "minus_mean_table.csv", index=False)


def centered_linear_cka(z_d, z_a_ref) -> float:
    """Linear CKA of paired sample-row representations (Eq. 16)."""
    z_d = np.asarray(z_d, dtype=np.float64)
    z_a_ref = np.asarray(z_a_ref, dtype=np.float64)
    if z_d.ndim != 2 or z_a_ref.ndim != 2 or z_d.shape[0] != z_a_ref.shape[0]:
        raise ValueError("Representations must be matrices with matching sample counts.")
    if z_d.shape[0] < 2 or min(z_d.shape[1], z_a_ref.shape[1]) < 1:
        raise ValueError("CKA requires at least two samples and nonempty features.")
    if not np.isfinite(z_d).all() or not np.isfinite(z_a_ref).all():
        raise ValueError("Representations must contain only finite values.")
    z_d = z_d - z_d.mean(axis=0, keepdims=True)
    z_a_ref = z_a_ref - z_a_ref.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(z_d.T @ z_d) * np.linalg.norm(z_a_ref.T @ z_a_ref)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("CKA is undefined for a constant or numerically degenerate representation.")
    return float(np.square(z_d.T @ z_a_ref).sum() / denominator)


def demographic_representation_alignment_reduction(cka: float, erm_cka: float) -> float:
    """DRAR as a fraction (Eq. 17); multiply by 100 for CKA_Reduction_percent."""
    if not np.isfinite([cka, erm_cka]).all() or erm_cka <= 0:
        raise ValueError("DRAR requires finite alignments and positive ERM CKA.")
    return float((erm_cka - cka) / erm_cka)


def fairness_tradeoff(bacc: float, erm_bacc: float, gap: float, erm_gap: float) -> float:
    """Relative BACC gain minus relative disparity increase (Eqs. 14-15)."""
    if not np.isfinite([bacc, erm_bacc, gap, erm_gap]).all() or erm_bacc <= 0 or erm_gap <= 0:
        raise ValueError("FATE requires finite metrics and positive ERM BACC and disparity.")
    return float((bacc - erm_bacc) / erm_bacc - (gap - erm_gap) / erm_gap)


def load_export(path: Path, fields: tuple[str, ...]) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(fields) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        result = {name: archive[name] for name in fields}
    sample_ids = result["sample_ids"]
    if sample_ids.ndim != 1 or len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f"{path}: sample_ids must be a unique one-dimensional array")
    if any(values.shape[0] != len(sample_ids) for values in result.values()):
        raise ValueError(f"{path}: sample counts do not match")
    return result


def evaluate(method: dict, erm: dict, reference: dict, task: str) -> dict:
    """Use the supplied evaluation script's fixed 0.5 operating threshold."""
    for name, other in (("ERM", erm), ("attribute reference", reference)):
        if not np.array_equal(method["sample_ids"], other["sample_ids"]):
            raise ValueError(f"Sample order differs between method and {name}")
    for field in ("labels", "attributes"):
        if not np.array_equal(method[field], erm[field]):
            raise ValueError(f"Method and ERM {field} do not match")
    metrics = []
    for export in (method, erm):
        labels, scores, attributes = (export[key] for key in ("labels", "scores", "attributes"))
        if labels.ndim != 1 or scores.shape != labels.shape or attributes.shape != (len(labels), 3):
            raise ValueError("Expected labels/scores [n] and age/sex/ethnicity attributes [n,3]")
        if not np.isin(labels, [0, 1]).all() or not np.isin(attributes, [0, 1]).all():
            raise ValueError("Disease and demographic labels must be binary")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Scores must be finite probabilities in [0,1]")
        _, groups = result_tables(labels, scores, attributes, task, threshold=0.5)
        if len(groups) != 8 or groups[["auc", "fpr", "fnr"]].isna().any().any():
            raise ValueError("Paper evaluation requires both disease classes in all eight intersections")
        metrics.append(validation_metrics(labels, scores, attributes, threshold=0.5))
    current, baseline = metrics
    cka = centered_linear_cka(method["z_d"], reference["z_a_ref"])
    erm_cka = centered_linear_cka(erm["z_d"], reference["z_a_ref"])
    drar = demographic_representation_alignment_reduction(cka, erm_cka)
    return {
        "Task": task,
        "N": int(len(method["labels"])),
        "Threshold": 0.5,
        "BACC": current["bacc"],
        "AUC": current["auc"],
        "Macro_F1": current["macro_f1"],
        "Intersection_delta_EO": current["delta_eo"],
        "Intersection_delta_AUC": current["delta_auc"],
        "FATEEO_BACC": fairness_tradeoff(current["bacc"], baseline["bacc"], current["delta_eo"], baseline["delta_eo"]),
        "FATEAUC_BACC": fairness_tradeoff(current["bacc"], baseline["bacc"], current["delta_auc"], baseline["delta_auc"]),
        "CKA_D_vs_A": cka,
        "CKA_ERM_D_vs_A": erm_cka,
        "DRAR": drar,
        "DRAR_percent": 100.0 * drar,
        "CKA_Reduction_percent": 100.0 * drar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=Path, required=True, help="NPZ with method predictions and z_d")
    parser.add_argument("--erm", type=Path, required=True, help="NPZ with ERM predictions and z_d")
    parser.add_argument("--attribute-reference", type=Path, required=True, help="NPZ from a separately supervised attribute encoder")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    fields = ("sample_ids", "labels", "attributes", "scores", "z_d")
    row = evaluate(
        load_export(args.method, fields),
        load_export(args.erm, fields),
        load_export(args.attribute_reference, ("sample_ids", "z_a_ref")),
        args.task,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output / "paper_metrics.csv", index=False)
    (args.output / "paper_metrics.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    print(json.dumps(row, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
