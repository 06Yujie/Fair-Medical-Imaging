"""CheXpert metadata interface and contrastive-friendly batch sampling."""

from __future__ import annotations

import math
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision import transforms


TASKS = ("No Finding", "Cardiomegaly", "Effusion", "Pneumothorax")
_ALL_ATTRIBUTE_NAMES = ("age", "sex", "ethnicity")
_ALL_ATTRIBUTE_COLUMNS = (
    "_age_binary",
    "_sex_binary",
    "_ethnicity_binary",
)
ATTRIBUTE_NAMES = _ALL_ATTRIBUTE_NAMES
ATTRIBUTE_COLUMNS = _ALL_ATTRIBUTE_COLUMNS

def normalize_split(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"0", "tr", "train"}:
        return "train"
    if value in {"1", "va", "val", "valid", "validation"}:
        return "val"
    if value in {"2", "te", "test"}:
        return "test"
    return value


def detect_column(columns, candidates: tuple[str, ...]) -> str:
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for candidate in candidates:
        matches = [c for c in columns if str(c).lower() == candidate.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous metadata column {candidate!r}: {matches}")
    raise ValueError(f"None of {candidates} exists in metadata columns.")


def detect_raw_age_column(columns) -> str:
    """Use the raw numeric `Age` field, never the derived lowercase `age`."""
    columns = list(columns)
    if "Age" not in columns:
        raise ValueError(
            "Required raw age column 'Age' is missing. The lowercase 'age' "
            "column is a derived group label and must not be used as raw age."
        )
    return "Age"


def stratified_subset(
    frame: pd.DataFrame, fraction: float, seed: int, strata_cols: tuple[str, ...]
) -> pd.DataFrame:
    if fraction >= 1.0:
        return frame.copy()
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    target_n = max(1, int(round(len(frame) * fraction)))
    groups = []
    for key, indices in frame.groupby(
        list(strata_cols), dropna=False, sort=True
    ).indices.items():
        indices = np.asarray(indices, dtype=int)
        exact = len(indices) * fraction
        groups.append(
            {
                "key": key,
                "indices": indices,
                "exact": exact,
                "quota": int(np.floor(exact)),
            }
        )
    if target_n >= len(groups):
        for group in groups:
            group["quota"] = max(group["quota"], 1)
    current = sum(group["quota"] for group in groups)
    if current < target_n:
        order = sorted(
            range(len(groups)),
            key=lambda i: (
                groups[i]["exact"] - np.floor(groups[i]["exact"]),
                len(groups[i]["indices"]),
            ),
            reverse=True,
        )
        for index in order:
            if current >= target_n:
                break
            if groups[index]["quota"] < len(groups[index]["indices"]):
                groups[index]["quota"] += 1
                current += 1
    elif current > target_n:
        order = sorted(
            range(len(groups)),
            key=lambda i: (
                groups[i]["exact"] - np.floor(groups[i]["exact"]),
                len(groups[i]["indices"]),
            ),
        )
        for index in order:
            if current <= target_n:
                break
            minimum = 1 if target_n >= len(groups) else 0
            if groups[index]["quota"] > minimum:
                groups[index]["quota"] -= 1
                current -= 1
    rng = np.random.RandomState(seed)
    chosen = []
    for group in groups:
        if group["quota"]:
            chosen.extend(
                rng.permutation(group["indices"])[: group["quota"]].tolist()
            )
    return frame.iloc[sorted(chosen)].copy()


class SquareCenterCrop:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        return image.crop((left, top, left + side, top + side))


def build_transform(
    train: bool,
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int = 224,
    strong: bool = False,
):
    steps = [
        SquareCenterCrop(),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5
            ),
        ]
        if strong:
            steps += [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.25
                ),
            ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


class _PixelStatisticsDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, path_col: str, image_root: Path, image_size: int
    ) -> None:
        self.paths = frame[path_col].astype(str).tolist()
        self.image_root = image_root
        self.transform = transforms.Compose(
            [
                SquareCenterCrop(),
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = Path(self.paths[index])
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def _compute_pixel_statistics(
    dataset: "DiseaseDataset", image_size: int, num_workers: int
) -> tuple[list[float], list[float]]:
    statistics_dataset = _PixelStatisticsDataset(
        dataset.frame, dataset.path_col, dataset.image_root, image_size
    )
    loader = DataLoader(
        statistics_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0
    for images in loader:
        images = images.to(dtype=torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += images.square().sum(dim=(0, 2, 3))
        total_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = channel_sum / total_pixels
    variance = (channel_sum_sq / total_pixels - mean.square()).clamp_min(1e-12)
    return mean.tolist(), variance.sqrt().tolist()


def resolve_train_statistics(
    dataset: "DiseaseDataset",
    image_size: int,
    num_workers: int,
    cache_path: str | Path | None,
) -> tuple[list[float], list[float], dict]:
    """Compute train-only RGB statistics once and safely share the cache."""
    expected = {
        "metadata_sha256": hashlib.sha256(dataset.metadata_csv.read_bytes()).hexdigest(),
        "image_size": image_size,
        "train_samples": len(dataset),
        "preprocessing": "square_center_crop_resize_to_tensor",
        "value_range": "[0,1]",
    }
    if cache_path is None:
        mean, std = _compute_pixel_statistics(dataset, image_size, num_workers)
        return mean, std, {**expected, "mean": mean, "std": std}
    cache_path = Path(cache_path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def read_cache():
        payload = json.loads(cache_path.read_text())
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Normalization cache {cache_path} has {key}={payload.get(key)!r}; "
                    f"expected {value!r}."
                )
        return payload["mean"], payload["std"], payload

    if cache_path.exists():
        return read_cache()
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    owns_lock = False
    while not owns_lock:
        try:
            lock_path.mkdir()
            owns_lock = True
        except FileExistsError:
            if cache_path.exists():
                return read_cache()
            if time.time() - lock_path.stat().st_mtime > 6 * 60 * 60:
                lock_path.rmdir()
                continue
            print(f"[data] waiting for normalization cache: {cache_path}", flush=True)
            time.sleep(10)
    try:
        if cache_path.exists():
            return read_cache()
        print(
            f"[data] computing train RGB mean/std from {len(dataset)} images...",
            flush=True,
        )
        mean, std = _compute_pixel_statistics(dataset, image_size, num_workers)
        payload = {**expected, "mean": mean, "std": std}
        temporary = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, cache_path)
        print(f"[data] normalization mean={mean}, std={std}", flush=True)
        return mean, std, payload
    finally:
        if lock_path.exists():
            lock_path.rmdir()


class DiseaseDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        split: str,
        target: str,
        image_root: str | Path | None = None,
        image_size: int = 224,
        fraction: float = 0.1,
        subsample_seed: int = 0,
        normalization_mean: Sequence[float] = (0.0, 0.0, 0.0),
        normalization_std: Sequence[float] = (1.0, 1.0, 1.0),
        dual_views: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r}.")
        metadata_csv = Path(metadata_csv).expanduser().resolve()
        self.metadata_csv = metadata_csv
        frame = pd.read_csv(metadata_csv)
        self.path_col = detect_column(
            frame.columns,
            (
                "filename",
                "path",
                "image_path",
                "img_path",
                "filepath",
                "relative_path",
                "file",
            ),
        )
        split_col = detect_column(frame.columns, ("split", "set", "subset", "partition"))
        if target not in frame.columns:
            raise ValueError(f"Disease target {target!r} does not exist in metadata.")
        resolved = {}
        if "age" in ATTRIBUTE_NAMES:
            resolved["age"] = detect_raw_age_column(frame.columns)
        if "sex" in ATTRIBUTE_NAMES:
            resolved["sex"] = detect_column(frame.columns, ("sex", "gender"))
        if "ethnicity" in ATTRIBUTE_NAMES:
            resolved["ethnicity"] = detect_column(frame.columns, ("ethnicity", "race"))
        self.resolved_sensitive_columns = resolved
        frame["_split"] = frame[split_col].map(normalize_split)
        for task in TASKS:
            if task not in frame.columns:
                raise ValueError(f"Required disease column {task!r} is missing.")
            frame[task] = pd.to_numeric(frame[task], errors="coerce")
        raw_columns = {}
        for name, source in resolved.items():
            raw = f"_{name}"
            frame[raw] = pd.to_numeric(frame[source], errors="coerce")
            raw_columns[name] = raw
        frame = frame.loc[frame["_split"] == split].dropna(
            subset=[*TASKS, *raw_columns.values()]
        )
        for task in TASKS:
            frame[task] = (frame[task] > 0.5).astype(int)
        for name, column in zip(ATTRIBUTE_NAMES, ATTRIBUTE_COLUMNS):
            values = frame[raw_columns[name]]
            frame[column] = ((values >= 60) if name == "age" else (values != 0)).astype(int)
        self.full_split_size = len(frame)
        frame = stratified_subset(
            frame,
            fraction,
            subsample_seed + {"train": 0, "val": 1, "test": 2}[split],
            (*TASKS, *ATTRIBUTE_COLUMNS),
        )
        self.frame = frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No valid samples for split={split!r}, target={target!r}.")
        for name, column in zip(ATTRIBUTE_NAMES, ATTRIBUTE_COLUMNS):
            observed = set(self.frame[column].unique().tolist())
            if observed != {0, 1}:
                raise ValueError(
                    f"Sensitive attribute {name!r} has groups {sorted(observed)}."
                )
        self.target = target
        self.image_root = (
            Path(image_root).expanduser().resolve()
            if image_root
            else metadata_csv.parent
        )
        self.transform = build_transform(
            split == "train", normalization_mean, normalization_std, image_size
        )
        self.strong_transform = build_transform(
            split == "train",
            normalization_mean,
            normalization_std,
            image_size,
            strong=True,
        )
        self.dual_views = bool(dual_views and split == "train")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(str(row[self.path_col]))
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
            strong_tensor = self.strong_transform(image) if self.dual_views else None
        label = torch.tensor(float(row[self.target]), dtype=torch.float32)
        attributes = torch.tensor(
            [int(row[column]) for column in ATTRIBUTE_COLUMNS], dtype=torch.long
        )
        if self.dual_views:
            return image_tensor, strong_tensor, label, attributes
        return image_tensor, label, attributes


class DiseaseAttributeBatchSampler(BatchSampler):
    """Queue-based disease/intersectional-attribute stratified batches.

    Every epoch independently shuffles each (disease, attribute-code) queue.
    A fixed number of slots is assigned to each disease class. Within each
    disease class, queued (not-yet-seen) samples are always preferred, and two
    different attribute groups are seeded whenever possible. Replacement from
    a group is allowed only after that group's shuffled queue is exhausted.

    The epoch length is chosen so both disease classes have enough assigned
    slots to consume every original sample at least once.
    """

    def __init__(
        self, dataset: DiseaseDataset, batch_size: int, seed: int, drop_last: bool = False
    ) -> None:
        if batch_size < 4:
            raise ValueError("The balanced contrastive sampler requires batch_size >= 4.")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.iteration = 0
        self.groups: dict[tuple[int, int], np.ndarray] = {}
        for index, row in dataset.frame.iterrows():
            disease = int(row[dataset.target])
            code = sum(int(row[column]) << bit for bit, column in enumerate(ATTRIBUTE_COLUMNS))
            self.groups.setdefault((disease, code), []).append(index)
        self.groups = {
            key: np.asarray(indices, dtype=int) for key, indices in self.groups.items()
        }
        self.codes = {
            disease: sorted(code for group_disease, code in self.groups
                            if group_disease == disease)
            for disease in (0, 1)
        }
        if any(not self.codes[disease] for disease in (0, 1)):
            raise ValueError("Queue sampler requires both disease classes.")
        self.disease_slots = {
            0: batch_size // 2,
            1: batch_size - batch_size // 2,
        }
        disease_counts = {
            disease: sum(len(self.groups[(disease, code)])
                         for code in self.codes[disease])
            for disease in (0, 1)
        }
        # One slot per disease may be needed for replacement to preserve a
        # cross-group pair after rare groups are exhausted. Therefore at least
        # (slots - 1) positions per batch are guaranteed to consume queued
        # originals until that disease class is fully covered.
        self.num_batches = max(
            math.ceil(
                disease_counts[disease]
                / max(self.disease_slots[disease] - 1, 1)
            )
            for disease in (0, 1)
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + self.iteration)
        self.iteration += 1
        queues = {
            key: rng.permutation(indices).tolist()
            for key, indices in self.groups.items()
        }
        positions = {key: 0 for key in self.groups}
        group_orders = {
            disease: rng.permutation(self.codes[disease]).tolist()
            for disease in (0, 1)
        }
        group_cursors = {0: 0, 1: 0}

        def queued(code: int, disease: int) -> bool:
            key = (disease, code)
            return positions[key] < len(queues[key])

        def take(disease: int, code: int) -> int:
            key = (disease, code)
            position = positions[key]
            if position < len(queues[key]):
                positions[key] += 1
                return int(queues[key][position])
            # This group's full shuffled queue has already been consumed.
            return int(rng.choice(self.groups[key]))

        def next_active_code(disease: int, excluded: set[int] | None = None):
            excluded = excluded or set()
            order = group_orders[disease]
            for offset in range(len(order)):
                index = (group_cursors[disease] + offset) % len(order)
                code = int(order[index])
                if code not in excluded and queued(code, disease):
                    group_cursors[disease] = (index + 1) % len(order)
                    return code
            return None

        def replacement_code(disease: int, excluded: set[int] | None = None) -> int:
            excluded = excluded or set()
            candidates = [
                code for code in self.codes[disease]
                if code not in excluded and not queued(code, disease)
            ]
            if not candidates:
                candidates = [
                    code for code in self.codes[disease] if code not in excluded
                ]
            if not candidates:
                candidates = self.codes[disease]
            return int(rng.choice(candidates))

        for _ in range(self.num_batches):
            batch: list[int] = []
            for disease in (0, 1):
                slots = self.disease_slots[disease]
                disease_batch: list[int] = []

                # Seed cross-group positives for this disease class.
                if slots >= 2 and len(self.codes[disease]) >= 2:
                    first = next_active_code(disease)
                    if first is None:
                        first = replacement_code(disease)
                    disease_batch.append(take(disease, first))
                    second = next_active_code(disease, {first})
                    if second is None:
                        second = replacement_code(disease, {first})
                    disease_batch.append(take(disease, second))

                while len(disease_batch) < slots:
                    code = next_active_code(disease)
                    if code is None:
                        code = replacement_code(disease)
                    disease_batch.append(take(disease, code))

                batch.extend(disease_batch)
            rng.shuffle(batch)
            yield batch


def make_disease_loaders(
    metadata_csv: str | Path,
    target: str,
    image_root: str | Path | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    fraction: float = 0.1,
    subsample_seed: int = 0,
    balanced_batches: bool = True,
    normalization_stats: str | Path | None = None,
    dual_view_train: bool = False,
) -> dict[str, DataLoader]:
    train_dataset = DiseaseDataset(
        metadata_csv,
        "train",
        target,
        image_root,
        image_size,
        fraction,
        subsample_seed,
        dual_views=dual_view_train,
    )
    mean, std, statistics = resolve_train_statistics(
        train_dataset, image_size, num_workers, normalization_stats
    )
    train_dataset.transform = build_transform(True, mean, std, image_size)
    train_dataset.strong_transform = build_transform(
        True, mean, std, image_size, strong=True
    )
    train_dataset.normalization_statistics = statistics
    datasets = {"train": train_dataset}
    for split in ("val", "test"):
        dataset = DiseaseDataset(
            metadata_csv,
            split,
            target,
            image_root,
            image_size,
            fraction,
            subsample_seed,
            mean,
            std,
        )
        dataset.normalization_statistics = statistics
        datasets[split] = dataset
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    loaders = {}
    for split, dataset in datasets.items():
        if split == "train" and balanced_batches:
            sampler = DiseaseAttributeBatchSampler(
                dataset, batch_size=batch_size, seed=subsample_seed
            )
            loaders[split] = DataLoader(dataset, batch_sampler=sampler, **common)
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                **common,
            )
    return loaders
