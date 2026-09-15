#!/usr/bin/env python3
"""Convert raw CheXpert tables into balanced train/val and natural test CSVs.

Expected inputs under ``--chexpert-root``:

* ``train.csv`` and optionally ``valid.csv`` from CheXpert;
* a patient-demographics CSV/XLSX supplied through ``--demographics``. It must
  contain patient ID and race/ethnicity; common CheXpert column names are
  detected automatically.

The output contains only relative image paths:

    <output>/no_finding.csv             # combined train, val, and test rows
    <output>/train/no_finding.csv       # balanced train and validation rows
    <output>/test/no_finding.csv        # natural-distribution test rows

The same pair is produced for Cardiomegaly, Effusion, and Pneumothorax. Patients
are assigned deterministically to 70% train, 15% validation, and 15% test. Only
train/validation are downsampled; test retains every valid patient-disjoint row.

Example:

    python prepare_chexpert_data.py \
      --chexpert-root data/CheXpert-v1.0 \
      --demographics data/chexpert_demographics.xlsx \
      --output data/processed

The label policy is U-Zero: missing and uncertain (-1) labels become 0.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


TASKS = {
    "No Finding": "no_finding",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "effusion",
    "Pneumothorax": "pneumothorax",
}
RAW_TASK_COLUMNS = {
    "No Finding": ("No Finding",),
    "Cardiomegaly": ("Cardiomegaly",),
    "Effusion": ("Effusion", "Pleural Effusion"),
    "Pneumothorax": ("Pneumothorax",),
}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
OUTPUT_COLUMNS = (
    "Path",
    "subject_id",
    "split",
    "original_source_split",
    "Age",
    "sex",
    "ethnicity",
    *TASKS,
)


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(f"Cannot find {label}; tried columns {candidates}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("--demographics must be a CSV, XLSX, or XLS file")


def extract_patient_key(path: object) -> str | None:
    match = re.search(r"patient\d+", str(path), flags=re.IGNORECASE)
    return match.group(0).lower() if match else None


def numeric_subject_id(patient_key: object) -> str | None:
    match = re.search(r"\d+", str(patient_key))
    return str(int(match.group(0))) if match else None


def relative_image_path(path: object) -> str:
    """Return a portable path relative to the CheXpert image root."""
    normalized = str(path).strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Encountered an empty image path")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError(f"Image path must not contain parent traversal: {normalized}")
    lowered = [part.lower() for part in parts]
    for split_name in ("train", "valid", "test"):
        if split_name in lowered:
            index = lowered.index(split_name)
            return PurePosixPath(*parts[index:]).as_posix()
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise ValueError(f"Cannot convert image path to a relative path: {normalized}")
    return PurePosixPath(normalized).as_posix()


def encode_sex(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"female", "f", "0", "0.0"}:
        return 0
    if text in {"male", "m", "1", "1.0"}:
        return 1
    return None


def encode_ethnicity(value: object) -> int | None:
    """Encode White as 0 and all other/unknown race categories as 1."""
    if pd.isna(value):
        return 1
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return int(number != 0.0) if math.isfinite(number) else None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return 1
    return 0 if "white" in text and "non-white" not in text else 1


def canonicalize_raw_data(
    chexpert_root: Path, demographics_path: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_frames = []
    for filename, source_split in (("train.csv", "train"), ("valid.csv", "valid")):
        path = chexpert_root / filename
        if not path.exists():
            if filename == "valid.csv":
                continue
            raise FileNotFoundError(f"Missing required CheXpert table: {path}")
        frame = pd.read_csv(path)
        frame["original_source_split"] = source_split
        raw_frames.append(frame)
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)

    path_column = find_column(raw, ("Path", "path"), "image path")
    age_column = find_column(raw, ("Age", "age"), "age")
    sex_column = find_column(raw, ("Sex", "sex", "Gender", "gender"), "sex")
    raw["_patient_key"] = raw[path_column].map(extract_patient_key)
    # The original experiment hashed the integer-like ID ("1"), not the
    # display key ("patient00001"). Keeping that convention reproduces its
    # deterministic patient partitions.
    raw["subject_id"] = raw["_patient_key"].map(numeric_subject_id)
    raw["Path"] = raw[path_column].map(relative_image_path)
    raw["Age"] = pd.to_numeric(raw[age_column], errors="coerce")
    raw["sex"] = raw[sex_column].map(encode_sex)

    demographics = read_table(demographics_path)
    patient_column = find_column(
        demographics,
        ("PATIENT", "patient", "subject_id", "patient_id"),
        "demographic patient ID",
    )
    race_column = find_column(
        demographics,
        ("PRIMARY_RACE", "primary_race", "race", "ethnicity", "ETHNICITY"),
        "race/ethnicity",
    )
    demo = demographics[[patient_column, race_column]].copy()
    # Match patient00001, integer IDs, and spreadsheet numeric IDs consistently.
    demo["_patient_key"] = demo[patient_column].map(numeric_subject_id)
    demo["ethnicity"] = demo[race_column].map(encode_ethnicity)
    demo = demo.dropna(subset=["_patient_key", "ethnicity"])
    conflicting = demo.groupby("_patient_key")["ethnicity"].nunique()
    if (conflicting > 1).any():
        examples = conflicting[conflicting > 1].index[:5].tolist()
        raise ValueError(f"Conflicting demographic rows for patients: {examples}")
    demo["_demo_present"] = True
    demo = demo.drop_duplicates("_patient_key", keep="first")[
        ["_patient_key", "ethnicity", "_demo_present"]
    ]
    frame = raw.merge(
        demo.rename(columns={"_patient_key": "subject_id"}),
        on="subject_id", how="left", validate="many_to_one",
    )
    frame["ethnicity"] = frame["ethnicity"].fillna(1).astype(np.int8)
    # Filled later by the deterministic patient-level splitter.
    frame["split"] = "unassigned"

    for target, candidates in RAW_TASK_COLUMNS.items():
        source = find_column(raw, candidates, target)
        values = pd.to_numeric(frame[source], errors="coerce").fillna(0.0)
        frame[target] = (values > 0.5).astype(np.int8)

    # Patients absent from the supplied demographic table cannot participate in
    # intersectional balancing, even though unknown race *within* that table is
    # retained in the non-White/other group to reproduce the original protocol.
    invalid = frame[list(OUTPUT_COLUMNS)].isna().any(axis=1) | frame["_demo_present"].isna()
    duplicate = frame["Path"].duplicated(keep="first")
    audit = {
        "raw_rows": int(len(frame)),
        "rows_missing_required_demographics": int(invalid.sum()),
        "duplicate_paths_removed": int((duplicate & ~invalid).sum()),
    }
    frame = frame.loc[~invalid & ~duplicate, list(OUTPUT_COLUMNS)].copy()
    frame["Age"] = frame["Age"].astype(int)
    frame["sex"] = frame["sex"].astype(np.int8)
    frame["ethnicity"] = frame["ethnicity"].astype(np.int8)
    audit["canonical_rows"] = int(len(frame))
    audit["canonical_patients"] = int(frame["subject_id"].nunique())
    return frame, audit


def patient_split(patient_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < SPLIT_RATIOS["train"]:
        return "train"
    if value < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def pattern(row: pd.Series, target: str) -> tuple[int, int, int, int]:
    return (
        int(row[target]),
        int(row["Age"] >= 60),
        int(row["sex"]),
        int(row["ethnicity"]),
    )


def maximum_balanced_quotas(
    counts: Counter[tuple[int, int, int, int]],
    margin_tolerance: float,
    intersection_tolerance: float,
) -> dict[tuple[int, int, int, int], int]:
    """Maximize retained rows under the original approximate balance constraints."""
    patterns = sorted(counts)
    if not patterns:
        raise ValueError("Cannot balance an empty split")
    capacities = np.asarray([counts[item] for item in patterns], dtype=float)
    rows, lower, upper = [], [], []
    low, high = 0.5 - margin_tolerance, 0.5 + margin_tolerance

    for column in range(4):
        positive = np.asarray([item[column] for item in patterns], dtype=float)
        total = np.ones(len(patterns), dtype=float)
        rows.extend((positive - low * total, positive - high * total))
        lower.extend((0.0, -np.inf))
        upper.extend((np.inf, 0.0))

    for attribute_column in range(1, 4):
        for value in (0, 1):
            group = np.asarray(
                [item[attribute_column] == value for item in patterns], dtype=float
            )
            disease_positive = np.asarray(
                [item[attribute_column] == value and item[0] == 1 for item in patterns],
                dtype=float,
            )
            rows.extend((disease_positive - low * group, disease_positive - high * group))
            lower.extend((0.0, -np.inf))
            upper.extend((np.inf, 0.0))

    cell_low = 0.5 - intersection_tolerance
    cell_high = 0.5 + intersection_tolerance
    for attributes in itertools.product((0, 1), repeat=3):
        if not all((label, *attributes) in counts for label in (0, 1)):
            raise RuntimeError(
                f"Cannot balance intersection {attributes}: one disease class is absent"
            )
        group = np.asarray([item[1:] == attributes for item in patterns], dtype=float)
        disease_positive = np.asarray(
            [item[1:] == attributes and item[0] == 1 for item in patterns], dtype=float
        )
        rows.extend(
            (disease_positive - cell_low * group, disease_positive - cell_high * group)
        )
        lower.extend((0.0, -np.inf))
        upper.extend((np.inf, 0.0))

    result = milp(
        c=-np.ones(len(patterns), dtype=float),
        integrality=np.ones(len(patterns), dtype=int),
        bounds=Bounds(np.ones(len(patterns)), capacities),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Balance optimization failed: {result.message}")
    selected = np.rint(result.x).astype(int)
    return {item: int(selected[index]) for index, item in enumerate(patterns)}


def balanced_subset(
    frame: pd.DataFrame,
    target: str,
    seed: int,
    margin_tolerance: float,
    intersection_tolerance: float,
) -> pd.DataFrame:
    cells: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        cells[pattern(row, target)].append(index)
    counts = Counter({key: len(indices) for key, indices in cells.items()})
    quotas = maximum_balanced_quotas(counts, margin_tolerance, intersection_tolerance)
    selected = []
    for cell, quota in sorted(quotas.items()):
        indices = cells[cell].copy()
        cell_seed = seed + sum(bit << position for position, bit in enumerate(cell))
        random.Random(cell_seed).shuffle(indices)
        selected.extend(indices[:quota])
    return frame.loc[sorted(selected)].copy()


def summarize(frame: pd.DataFrame, target: str) -> dict[str, float | int]:
    total = len(frame)
    if total == 0:
        return {"rows": 0}
    return {
        "rows": total,
        "patients": int(frame["subject_id"].nunique()),
        "target_positive_rate": float(frame[target].mean()),
        "age_ge_60_rate": float((frame["Age"] >= 60).mean()),
        "male_rate": float(frame["sex"].mean()),
        "non_white_rate": float(frame["ethnicity"].mean()),
    }


def assert_patient_disjoint(frame: pd.DataFrame) -> None:
    patients = {
        split: set(frame.loc[frame["split"] == split, "subject_id"])
        for split in SPLIT_RATIOS
    }
    for left, right in itertools.combinations(SPLIT_RATIOS, 2):
        overlap = patients[left] & patients[right]
        if overlap:
            raise RuntimeError(f"Patient leakage between {left} and {right}: {len(overlap)}")


def write_outputs(
    frame: pd.DataFrame,
    output: Path,
    seed: int,
    margin_tolerance: float,
    intersection_tolerance: float,
    source_audit: dict[str, int],
) -> None:
    frame = frame.copy()
    frame["split"] = frame["subject_id"].map(lambda value: patient_split(value, seed))
    assert_patient_disjoint(frame)
    train_dir, test_dir = output / "train", output / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = {
        "seed": seed,
        "patient_split": SPLIT_RATIOS,
        "balance_margin_tolerance": margin_tolerance,
        "intersection_tolerance": intersection_tolerance,
        "source": source_audit,
        "tasks": {},
    }
    test_manifest = {
        "seed": seed,
        "selection": "All valid rows from the deterministic patient-level test split",
        "source": source_audit,
        "tasks": {},
    }

    natural_test = frame.loc[frame["split"] == "test", list(OUTPUT_COLUMNS)].copy()
    for task_index, (target, slug) in enumerate(TASKS.items()):
        balanced_parts = []
        task_stats = {}
        for split_index, split in enumerate(("train", "val")):
            candidates = frame.loc[frame["split"] == split]
            selected = balanced_subset(
                candidates,
                target,
                seed + task_index * 10_000 + split_index * 1_000,
                margin_tolerance,
                intersection_tolerance,
            )
            balanced_parts.append(selected)
            task_stats[split] = summarize(selected, target)
        balanced = pd.concat(balanced_parts, ignore_index=True)[list(OUTPUT_COLUMNS)]
        assert_patient_disjoint(pd.concat([balanced, natural_test], ignore_index=True))
        balanced.to_csv(train_dir / f"{slug}.csv", index=False)
        natural_test.to_csv(test_dir / f"{slug}.csv", index=False)
        # The training loader reads all three splits from a single task CSV.
        pd.concat([balanced, natural_test], ignore_index=True).to_csv(
            output / f"{slug}.csv", index=False
        )
        train_manifest["tasks"][target] = task_stats
        test_manifest["tasks"][target] = summarize(natural_test, target)
        print(
            f"{target}: train={len(balanced[balanced['split'] == 'train'])}, "
            f"val={len(balanced[balanced['split'] == 'val'])}, test={len(natural_test)}"
        )

    (train_dir / "manifest.json").write_text(
        json.dumps(train_manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (test_dir / "manifest.json").write_text(
        json.dumps(test_manifest, indent=2, ensure_ascii=False) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chexpert-root",
        type=Path,
        default=Path("data/CheXpert-v1.0"),
        help="Directory containing train.csv, valid.csv, and the image folders.",
    )
    parser.add_argument(
        "--demographics",
        type=Path,
        required=True,
        help="Patient demographics CSV/XLSX containing patient ID and race/ethnicity.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--margin-tolerance", type=float, default=0.05)
    parser.add_argument("--intersection-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    for name in ("margin_tolerance", "intersection_tolerance"):
        value = getattr(args, name)
        if not 0.0 <= value < 0.5:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 0.5)")
    return args


def main() -> None:
    args = parse_args()
    canonical, audit = canonicalize_raw_data(args.chexpert_root, args.demographics)
    write_outputs(
        canonical,
        args.output,
        args.seed,
        args.margin_tolerance,
        args.intersection_tolerance,
        audit,
    )


if __name__ == "__main__":
    main()
