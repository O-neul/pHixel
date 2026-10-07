#!/usr/bin/env python
"""Smoke-train four robustness methods on one direct-capture LOPO split.

Methods
-------
1. baseline
2. colorjitter
3. nuisance
4. nuisance_consistency

The nuisance methods use the train-only decomposition created by script 03.
They move an image from its current phone/light nuisance state to a sampled
training-domain state. Background delta Lab is applied to the whole resized
image and contrast (pad - background) delta Lab is applied inside the
LabelMe reaction-pad ROI with a feathered boundary.

This script is intended for a one-split smoke test, not final reporting.
Validation and test images are never augmented. Final train metrics are also
computed with an unaugmented evaluation loader.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision.models as tv_models
    import torchvision.transforms.functional as TF
    from torchvision.transforms import ColorJitter
except Exception as exc:  # pragma: no cover
    print("ERROR: PyTorch/torchvision import failed.")
    print("Install torch and torchvision in this environment, then rerun.")
    print(f"Original error: {exc}")
    raise

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
except Exception as exc:  # pragma: no cover
    print("ERROR: scikit-learn import failed. Install scikit-learn, then rerun.")
    print(f"Original error: {exc}")
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_ORDER = ["acidic", "neutral", "alkaline"]
ALL_METHODS = ["baseline", "colorjitter", "nuisance", "nuisance_consistency"]
COMPONENT_NAMES = ["phone_effect", "light_effect", "interaction_effect", "residual"]


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class AugmentationSettings:
    probability: float
    phone_scale: float
    light_scale: float
    interaction_scale: float
    residual_scale: float
    overall_scale: float
    feather_radius_original_px: float
    sampler: str
    allow_same_condition: bool


# ---------------------------------------------------------------------------
# Arguments and general utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-split smoke training for baseline and nuisance methods."
    )
    parser.add_argument("--manifest", default="data/processed/capture_manifest.csv")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--decomposition-dir", required=True)
    parser.add_argument("--annotation-dir", default="data/raw/captures/annotations")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help=(
            "Comma-separated subset of: " + ", ".join(ALL_METHODS)
        ),
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenet_v3_small", "tiny_cnn"],
        default="resnet18",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--roi-label", default="roi")

    # Nuisance augmentation settings validated in preview 05.
    parser.add_argument("--nuisance-probability", type=float, default=0.5)
    parser.add_argument(
        "--sampler",
        choices=["structured", "independent", "swap_row"],
        default="structured",
    )
    parser.add_argument("--phone-scale", type=float, default=1.0)
    parser.add_argument("--light-scale", type=float, default=1.0)
    parser.add_argument("--interaction-scale", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--overall-scale", type=float, default=1.0)
    parser.add_argument(
        "--feather-radius",
        type=float,
        default=24.0,
        help="Feather radius in original annotation-image pixels.",
    )
    parser.add_argument("--allow-same-condition", action="store_true")

    # ColorJitter comparator.
    parser.add_argument("--jitter-brightness", type=float, default=0.20)
    parser.add_argument("--jitter-contrast", type=float, default=0.20)
    parser.add_argument("--jitter-saturation", type=float, default=0.20)
    parser.add_argument("--jitter-hue", type=float, default=0.05)

    # Consistency method.
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument(
        "--consistency-aug-probability",
        type=float,
        default=1.0,
        help="Probability of generating a nuisance view for consistency training.",
    )

    # Optional quick debugging limits. Keep 0 for the intended 104/26/65 split.
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks required columns: {missing}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_methods(text: str) -> List[str]:
    methods = [item.strip() for item in text.split(",") if item.strip()]
    if not methods:
        raise ValueError("No methods were selected")
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {ALL_METHODS}")
    # Preserve order and remove duplicates.
    return list(dict.fromkeys(methods))


def assign_class(ph: float) -> str:
    if np.isclose(float(ph), 7.0):
        return "neutral"
    return "acidic" if float(ph) < 7.0 else "alkaline"


# ---------------------------------------------------------------------------
# Data loading and leakage checks
# ---------------------------------------------------------------------------


def load_manifest_and_split(manifest_path: Path, split_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "image_id": str,
            "image_path": str,
            "phone_id": str,
            "light_id": str,
            "label_class": str,
        },
    )
    split = pd.read_csv(
        split_path,
        dtype={"image_id": str, "phone_id": str, "light_id": str, "split": str},
    )
    require_columns(
        manifest,
        ["image_id", "image_path", "phone_id", "light_id", "ph_nominal"],
        manifest_path,
    )
    require_columns(split, ["image_id", "split"], split_path)

    split_keep = split[["image_id", "split"]].drop_duplicates("image_id")
    merged = manifest.merge(split_keep, on="image_id", how="inner")
    merged = merged[merged["split"].isin(["train", "val", "test"])].copy()
    merged["ph_nominal"] = pd.to_numeric(merged["ph_nominal"], errors="raise")
    if "label_class" not in merged.columns:
        merged["label_class"] = [assign_class(value) for value in merged["ph_nominal"]]
    else:
        missing_label = ~merged["label_class"].isin(LABEL_ORDER)
        if missing_label.any():
            merged.loc[missing_label, "label_class"] = [
                assign_class(value) for value in merged.loc[missing_label, "ph_nominal"]
            ]

    if merged.empty:
        raise ValueError("No rows matched between manifest and split file")
    duplicated = merged["image_id"].duplicated(keep=False)
    if duplicated.any():
        examples = merged.loc[duplicated, "image_id"].head(5).tolist()
        raise ValueError(f"Duplicate image_id rows after merge: {examples}")
    return merged.reset_index(drop=True)


def limit_frame(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame.copy().reset_index(drop=True)
    # Keep class coverage when possible.
    chunks: List[pd.DataFrame] = []
    per_class = max(1, limit // len(LABEL_ORDER))
    for label in LABEL_ORDER:
        chunks.append(frame[frame["label_class"].eq(label)].head(per_class))
    selected = pd.concat(chunks, ignore_index=True).drop_duplicates("image_id")
    if len(selected) < limit:
        remaining = frame[~frame["image_id"].isin(selected["image_id"])]
        selected = pd.concat([selected, remaining.head(limit - len(selected))], ignore_index=True)
    return selected.head(limit).reset_index(drop=True)


def verify_split_structure(full_df: pd.DataFrame) -> dict:
    train = full_df[full_df["split"].eq("train")]
    val = full_df[full_df["split"].eq("val")]
    test = full_df[full_df["split"].eq("test")]
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Need non-empty train/val/test; got {len(train)}/{len(val)}/{len(test)}"
        )

    train_phones = set(train["phone_id"])
    test_phones = set(test["phone_id"])
    overlap_phone = sorted(train_phones & test_phones)
    if overlap_phone:
        raise ValueError(f"LOPO leakage: train and test share phones {overlap_phone}")

    train_lights = set(train["light_id"])
    val_lights = set(val["light_id"])
    overlap_light = sorted(train_lights & val_lights)
    if overlap_light:
        raise ValueError(
            "Unseen-light validation leakage: train and val share lights "
            f"{overlap_light}"
        )

    return {
        "rows": {name: int((full_df["split"] == name).sum()) for name in ["train", "val", "test"]},
        "train_phones": sorted(train_phones),
        "validation_phones": sorted(set(val["phone_id"])),
        "test_phones": sorted(test_phones),
        "train_lights": sorted(train_lights),
        "validation_lights": sorted(val_lights),
        "test_lights": sorted(set(test["light_id"])),
    }


# ---------------------------------------------------------------------------
# CIE Lab conversion and ROI masks
# ---------------------------------------------------------------------------


def srgb_to_lab(rgb_01: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_01, dtype=np.float64)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    scaled = xyz / white
    delta = 6.0 / 29.0
    transformed = np.where(
        scaled > delta**3,
        np.cbrt(scaled),
        scaled / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        [
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ],
        axis=-1,
    )


def lab_to_srgb_unclipped(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    delta = 6.0 / 29.0

    def inverse_f(value: np.ndarray) -> np.ndarray:
        return np.where(
            value > delta,
            value**3,
            3.0 * delta**2 * (value - 4.0 / 29.0),
        )

    white = np.array([0.95047, 1.0, 1.08883])
    xyz = np.stack([inverse_f(fx), inverse_f(fy), inverse_f(fz)], axis=-1) * white
    inverse_matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    linear = xyz @ inverse_matrix.T
    positive = np.maximum(linear, 0.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * positive ** (1.0 / 2.4) - 0.055,
    )


def load_labelme_feather_mask(
    annotation_path: Path,
    output_size: tuple[int, int],
    roi_label: str,
    feather_radius_original_px: float,
) -> np.ndarray:
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    candidates = [
        shape
        for shape in annotation.get("shapes", [])
        if str(shape.get("label", "")) == roi_label
    ]
    if not candidates:
        raise ValueError(f"No LabelMe shape labelled {roi_label!r}: {annotation_path}")
    points = candidates[0].get("points", [])
    if len(points) < 2:
        raise ValueError(f"ROI has fewer than two points: {annotation_path}")

    annotation_width = float(annotation.get("imageWidth") or output_size[0])
    annotation_height = float(annotation.get("imageHeight") or output_size[1])
    scale_x = output_size[0] / annotation_width
    scale_y = output_size[1] / annotation_height

    xs = [float(point[0]) * scale_x for point in points]
    ys = [float(point[1]) * scale_y for point in points]
    box = [min(xs), min(ys), max(xs), max(ys)]

    mask = Image.new("L", output_size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    scaled_radius = float(feather_radius_original_px) * min(scale_x, scale_y)
    if scaled_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=scaled_radius))
    return np.asarray(mask, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Train-only nuisance library and transform
# ---------------------------------------------------------------------------


def vector_from_row(row: pd.Series, component: str) -> np.ndarray:
    return np.array(
        [row[f"{component}_L"], row[f"{component}_a"], row[f"{component}_b"]],
        dtype=np.float64,
    )


class NuisanceRecombination:
    def __init__(
        self,
        library_path: Path,
        annotation_dir: Path,
        image_size: int,
        roi_label: str,
        settings: AugmentationSettings,
        seed: int,
    ) -> None:
        self.annotation_dir = annotation_dir
        self.image_size = int(image_size)
        self.roi_label = roi_label
        self.settings = settings
        self.rng = random.Random(seed)
        self.calls = 0
        self.applied = 0
        self.target_phone_counts: Counter[str] = Counter()
        self.target_light_counts: Counter[str] = Counter()
        self.delta_e_values: List[float] = []

        library = pd.read_csv(
            library_path,
            dtype={"image_id": str, "phone_id": str, "light_id": str, "region": str},
        )
        required = ["image_id", "phone_id", "light_id", "region"]
        for component in COMPONENT_NAMES:
            required.extend(
                [f"{component}_L", f"{component}_a", f"{component}_b"]
            )
        require_columns(library, required, library_path)
        missing_regions = {"bg", "contrast"} - set(library["region"].astype(str))
        if missing_regions:
            raise ValueError(f"Nuisance library lacks regions: {sorted(missing_regions)}")

        self.library = library.copy()
        self.image_ids = sorted(library["image_id"].astype(str).unique().tolist())
        self.phones = sorted(library["phone_id"].astype(str).unique().tolist())
        self.lights = sorted(library["light_id"].astype(str).unique().tolist())
        self.interaction_cells = sorted(
            set(zip(library["phone_id"].astype(str), library["light_id"].astype(str)))
        )

        self.current_rows: Dict[tuple[str, str], pd.Series] = {}
        self.phone_rows: Dict[tuple[str, str], pd.Series] = {}
        self.light_rows: Dict[tuple[str, str], pd.Series] = {}
        self.interaction_rows: Dict[tuple[str, str, str], pd.Series] = {}
        self.residual_rows: Dict[tuple[str, str], pd.Series] = {}
        for region in ["bg", "contrast"]:
            region_df = library[library["region"].eq(region)]
            for image_id, group in region_df.groupby("image_id", sort=False):
                self.current_rows[(region, str(image_id))] = group.iloc[0]
                self.residual_rows[(region, str(image_id))] = group.iloc[0]
            for phone, group in region_df.groupby("phone_id", sort=False):
                self.phone_rows[(region, str(phone))] = group.iloc[0]
            for light, group in region_df.groupby("light_id", sort=False):
                self.light_rows[(region, str(light))] = group.iloc[0]
            for (phone, light), group in region_df.groupby(
                ["phone_id", "light_id"], sort=False
            ):
                self.interaction_rows[(region, str(phone), str(light))] = group.iloc[0]

    @property
    def library_image_ids(self) -> set[str]:
        return set(self.image_ids)

    def _sample_target(self, source_phone: str, source_light: str) -> dict:
        candidate: dict = {}
        for _ in range(100):
            if self.settings.sampler == "swap_row":
                target_image_id = self.rng.choice(self.image_ids)
                target_row = self.current_rows[("bg", target_image_id)]
                target_phone = str(target_row["phone_id"])
                target_light = str(target_row["light_id"])
                interaction_phone = target_phone
                interaction_light = target_light
            elif self.settings.sampler == "structured":
                target_phone = self.rng.choice(self.phones)
                target_light = self.rng.choice(self.lights)
                if (target_phone, target_light) not in self.interaction_cells:
                    continue
                interaction_phone = target_phone
                interaction_light = target_light
                target_image_id = self.rng.choice(self.image_ids)
            else:
                target_phone = self.rng.choice(self.phones)
                target_light = self.rng.choice(self.lights)
                interaction_phone, interaction_light = self.rng.choice(
                    self.interaction_cells
                )
                target_image_id = self.rng.choice(self.image_ids)

            candidate = {
                "target_phone": target_phone,
                "target_light": target_light,
                "interaction_phone": interaction_phone,
                "interaction_light": interaction_light,
                "target_residual_image_id": target_image_id,
            }
            same = target_phone == source_phone and target_light == source_light
            if self.settings.allow_same_condition or not same:
                return candidate
        raise RuntimeError(f"Could not sample target condition; last candidate={candidate}")

    def _region_delta(self, region: str, image_id: str, target: dict) -> np.ndarray:
        current_row = self.current_rows[(region, image_id)]
        if self.settings.sampler == "swap_row":
            target_row = self.current_rows[(region, target["target_residual_image_id"])]
            target_vectors = {
                component: vector_from_row(target_row, component)
                for component in COMPONENT_NAMES
            }
        else:
            target_vectors = {
                "phone_effect": vector_from_row(
                    self.phone_rows[(region, target["target_phone"])],
                    "phone_effect",
                ),
                "light_effect": vector_from_row(
                    self.light_rows[(region, target["target_light"])],
                    "light_effect",
                ),
                "interaction_effect": vector_from_row(
                    self.interaction_rows[
                        (
                            region,
                            target["interaction_phone"],
                            target["interaction_light"],
                        )
                    ],
                    "interaction_effect",
                ),
                "residual": vector_from_row(
                    self.residual_rows[
                        (region, target["target_residual_image_id"])
                    ],
                    "residual",
                ),
            }

        scales = {
            "phone_effect": self.settings.phone_scale,
            "light_effect": self.settings.light_scale,
            "interaction_effect": self.settings.interaction_scale,
            "residual": self.settings.residual_scale,
        }
        total = np.zeros(3, dtype=np.float64)
        for component in COMPONENT_NAMES:
            current = vector_from_row(current_row, component)
            total += scales[component] * (target_vectors[component] - current)
        return self.settings.overall_scale * total

    def __call__(
        self,
        image: Image.Image,
        image_id: str,
        image_stem: str,
        source_phone: str,
        source_light: str,
        force_apply: bool = False,
    ) -> Image.Image:
        self.calls += 1
        apply_now = force_apply or self.rng.random() < self.settings.probability
        if not apply_now:
            return image

        target = self._sample_target(source_phone, source_light)
        bg_delta = self._region_delta("bg", image_id, target)
        contrast_delta = self._region_delta("contrast", image_id, target)

        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        lab = srgb_to_lab(rgb)
        annotation_path = self.annotation_dir / f"{image_stem}.json"
        mask = load_labelme_feather_mask(
            annotation_path=annotation_path,
            output_size=image.size,
            roi_label=self.roi_label,
            feather_radius_original_px=self.settings.feather_radius_original_px,
        )
        shifted_lab = (
            lab
            + bg_delta.reshape(1, 1, 3)
            + mask[..., None] * contrast_delta.reshape(1, 1, 3)
        )
        shifted_rgb = np.clip(lab_to_srgb_unclipped(shifted_lab), 0.0, 1.0)

        self.applied += 1
        self.target_phone_counts[target["target_phone"]] += 1
        self.target_light_counts[target["target_light"]] += 1
        pad_total_delta = bg_delta + contrast_delta
        self.delta_e_values.append(float(np.linalg.norm(pad_total_delta)))
        return Image.fromarray(
            np.round(shifted_rgb * 255.0).astype(np.uint8), mode="RGB"
        )

    def stats_dict(self) -> dict:
        values = np.asarray(self.delta_e_values, dtype=np.float64)
        return {
            "calls": int(self.calls),
            "applied": int(self.applied),
            "application_rate": float(self.applied / self.calls) if self.calls else 0.0,
            "target_phone_counts": dict(self.target_phone_counts),
            "target_light_counts": dict(self.target_light_counts),
            "pad_total_delta_e": {
                "n": int(values.size),
                "mean": float(values.mean()) if values.size else None,
                "q95": float(np.quantile(values, 0.95)) if values.size else None,
                "maximum": float(values.max()) if values.size else None,
            },
        }


# ---------------------------------------------------------------------------
# Dataset, normalization, and models
# ---------------------------------------------------------------------------


def open_resized_image(path: Path, image_size: int) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def compute_train_stats(train_df: pd.DataFrame, image_size: int) -> NormalizationStats:
    sums = np.zeros(3, dtype=np.float64)
    square_sums = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for _, row in train_df.iterrows():
        image_path = resolve(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = open_resized_image(image_path, image_size)
        array = np.asarray(image, dtype=np.float64) / 255.0
        flat = array.reshape(-1, 3)
        sums += flat.sum(axis=0)
        square_sums += (flat * flat).sum(axis=0)
        n_pixels += flat.shape[0]
    mean = sums / max(n_pixels, 1)
    variance = square_sums / max(n_pixels, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def pil_to_normalized_tensor(image: Image.Image, stats: NormalizationStats) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=stats.mean.tolist(), std=stats.std.tolist())


class CaptureDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_idx: Dict[str, int],
        stats: NormalizationStats,
        image_size: int,
        mode: str,
        nuisance: Optional[NuisanceRecombination] = None,
        color_jitter: Optional[ColorJitter] = None,
        consistency_probability: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.label_to_idx = label_to_idx
        self.stats = stats
        self.image_size = int(image_size)
        self.mode = mode
        self.nuisance = nuisance
        self.color_jitter = color_jitter
        self.consistency_probability = float(consistency_probability)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = resolve(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = open_resized_image(image_path, self.image_size)
        image_id = str(row["image_id"])
        label = torch.tensor(
            self.label_to_idx[str(row["label_class"])], dtype=torch.long
        )

        if self.mode == "nuisance_consistency":
            original_tensor = pil_to_normalized_tensor(image, self.stats)
            apply_aug = self.rng.random() < self.consistency_probability
            augmented = image
            if apply_aug:
                if self.nuisance is None:
                    raise RuntimeError("nuisance_consistency requires a nuisance transform")
                augmented = self.nuisance(
                    image=image,
                    image_id=image_id,
                    image_stem=Path(image_path).stem,
                    source_phone=str(row["phone_id"]),
                    source_light=str(row["light_id"]),
                    force_apply=True,
                )
            augmented_tensor = pil_to_normalized_tensor(augmented, self.stats)
            return original_tensor, augmented_tensor, label, image_id

        transformed = image
        if self.mode == "colorjitter":
            if self.color_jitter is None:
                raise RuntimeError("colorjitter mode requires ColorJitter")
            transformed = self.color_jitter(transformed)
        elif self.mode == "nuisance":
            if self.nuisance is None:
                raise RuntimeError("nuisance mode requires a nuisance transform")
            transformed = self.nuisance(
                image=transformed,
                image_id=image_id,
                image_stem=Path(image_path).stem,
                source_phone=str(row["phone_id"]),
                source_light=str(row["light_id"]),
            )
        elif self.mode not in {"baseline", "eval"}:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

        tensor = pil_to_normalized_tensor(transformed, self.stats)
        return tensor, label, image_id


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def make_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "tiny_cnn":
        return TinyCNN(num_classes)
    if model_name == "resnet18":
        if pretrained:
            model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        else:
            model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        if pretrained:
            model = tv_models.mobilenet_v3_small(
                weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
            )
        else:
            model = tv_models.mobilenet_v3_small(weights=None)
        final = model.classifier[-1]
        model.classifier[-1] = nn.Linear(final.in_features, num_classes)
        return model
    raise ValueError(model_name)


def class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["label_class"].value_counts().to_dict()
    raw = np.array(
        [0.0 if counts.get(label, 0) == 0 else 1.0 / counts[label] for label in LABEL_ORDER],
        dtype=np.float32,
    )
    if raw.sum() <= 0:
        raise ValueError("Cannot calculate class weights from empty class counts")
    raw = raw / raw.sum() * len(raw)
    return torch.tensor(raw, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        f"{prefix}_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_pa = F.log_softmax(logits_a, dim=1)
    log_pb = F.log_softmax(logits_b, dim=1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    kl_ab = F.kl_div(log_pa, pb, reduction="batchmean")
    kl_ba = F.kl_div(log_pb, pa, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def run_standard_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    image_ids: List[str] = []

    for x, y, ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
        batch = int(y.shape[0])
        total_loss += float(loss.item()) * batch
        total_count += batch
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        image_ids.extend(list(ids))

    return (
        total_loss / max(total_count, 1),
        np.asarray(all_true),
        np.asarray(all_pred),
        image_ids,
    )


def run_consistency_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    consistency_weight: float,
) -> Tuple[float, np.ndarray, np.ndarray, List[str], dict]:
    model.train(True)
    total_loss = 0.0
    total_ce = 0.0
    total_consistency = 0.0
    total_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    image_ids: List[str] = []

    for original, augmented, y, ids in loader:
        original = original.to(device, non_blocking=True)
        augmented = augmented.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits_original = model(original)
        logits_augmented = model(augmented)
        ce = 0.5 * (
            criterion(logits_original, y) + criterion(logits_augmented, y)
        )
        consistency = symmetric_kl(logits_original, logits_augmented)
        loss = ce + float(consistency_weight) * consistency
        loss.backward()
        optimizer.step()

        batch = int(y.shape[0])
        total_loss += float(loss.item()) * batch
        total_ce += float(ce.item()) * batch
        total_consistency += float(consistency.item()) * batch
        total_count += batch
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(
            torch.argmax(logits_original, dim=1).detach().cpu().numpy().tolist()
        )
        image_ids.extend(list(ids))

    details = {
        "ce_loss": total_ce / max(total_count, 1),
        "consistency_loss": total_consistency / max(total_count, 1),
    }
    return (
        total_loss / max(total_count, 1),
        np.asarray(all_true),
        np.asarray(all_pred),
        image_ids,
        details,
    )


def save_split_predictions(
    output_dir: Path,
    split_name: str,
    split_df: pd.DataFrame,
    image_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_label: Dict[int, str],
) -> None:
    predictions = pd.DataFrame(
        {
            "image_id": list(image_ids),
            "y_true_idx": y_true,
            "y_pred_idx": y_pred,
            "y_true_label": [idx_to_label[int(value)] for value in y_true],
            "y_pred_label": [idx_to_label[int(value)] for value in y_pred],
        }
    )
    metadata_columns = [
        column
        for column in [
            "image_id",
            "image_path",
            "phone_id",
            "light_id",
            "ph_nominal",
            "label_class",
        ]
        if column in split_df.columns
    ]
    predictions = predictions.merge(
        split_df[metadata_columns].drop_duplicates("image_id"),
        on="image_id",
        how="left",
    )
    predictions.to_csv(
        output_dir / f"predictions_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matrix = confusion_matrix(y_true, y_pred, labels=list(idx_to_label.keys()))
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{idx_to_label[index]}" for index in idx_to_label],
        columns=[f"pred_{idx_to_label[index]}" for index in idx_to_label],
    )
    matrix_df.index.name = "true_label"
    matrix_df.to_csv(
        output_dir / f"confusion_matrix_{split_name}.csv", encoding="utf-8-sig"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(idx_to_label.keys()),
        target_names=[idx_to_label[index] for index in idx_to_label],
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    report_df.index.name = "class"
    report_df.to_csv(
        output_dir / f"classification_report_{split_name}.csv",
        encoding="utf-8-sig",
    )


def build_loaders(
    method: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_to_idx: Dict[str, int],
    stats: NormalizationStats,
    args: argparse.Namespace,
    nuisance: Optional[NuisanceRecombination],
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    jitter = None
    if method == "colorjitter":
        jitter = ColorJitter(
            brightness=args.jitter_brightness,
            contrast=args.jitter_contrast,
            saturation=args.jitter_saturation,
            hue=args.jitter_hue,
        )

    train_dataset = CaptureDataset(
        train_df,
        label_to_idx,
        stats,
        args.image_size,
        mode=method,
        nuisance=nuisance,
        color_jitter=jitter,
        consistency_probability=args.consistency_aug_probability,
        seed=args.seed + 101,
    )
    train_eval_dataset = CaptureDataset(
        train_df,
        label_to_idx,
        stats,
        args.image_size,
        mode="eval",
        seed=args.seed + 102,
    )
    val_dataset = CaptureDataset(
        val_df,
        label_to_idx,
        stats,
        args.image_size,
        mode="eval",
        seed=args.seed + 103,
    )
    test_dataset = CaptureDataset(
        test_df,
        label_to_idx,
        stats,
        args.image_size,
        mode="eval",
        seed=args.seed + 104,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    train_eval_loader = DataLoader(train_eval_dataset, shuffle=False, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, train_eval_loader, val_loader, test_loader


def run_method(
    method: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stats: NormalizationStats,
    args: argparse.Namespace,
    device: torch.device,
    library_path: Path,
    annotation_dir: Path,
    output_root: Path,
) -> dict:
    # Reset every method so initialization, loader order, and optimizer state are comparable.
    set_seed(args.seed)
    method_dir = output_root / method
    method_dir.mkdir(parents=True, exist_ok=True)

    label_to_idx = {label: index for index, label in enumerate(LABEL_ORDER)}
    idx_to_label = {index: label for label, index in label_to_idx.items()}

    nuisance = None
    if method in {"nuisance", "nuisance_consistency"}:
        probability = (
            args.consistency_aug_probability
            if method == "nuisance_consistency"
            else args.nuisance_probability
        )
        settings = AugmentationSettings(
            probability=float(probability),
            phone_scale=float(args.phone_scale),
            light_scale=float(args.light_scale),
            interaction_scale=float(args.interaction_scale),
            residual_scale=float(args.residual_scale),
            overall_scale=float(args.overall_scale),
            feather_radius_original_px=float(args.feather_radius),
            sampler=str(args.sampler),
            allow_same_condition=bool(args.allow_same_condition),
        )
        nuisance = NuisanceRecombination(
            library_path=library_path,
            annotation_dir=annotation_dir,
            image_size=args.image_size,
            roi_label=args.roi_label,
            settings=settings,
            seed=args.seed + 201,
        )

    train_loader, train_eval_loader, val_loader, test_loader = build_loaders(
        method,
        train_df,
        val_df,
        test_df,
        label_to_idx,
        stats,
        args,
        nuisance,
    )

    model = make_model(args.model, len(LABEL_ORDER), args.pretrained).to(device)
    if args.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights(train_df).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = -math.inf
    best_epoch = -1
    no_improve = 0
    best_path = method_dir / "best_model.pt"
    logs: List[dict] = []
    start_time = time.time()

    print("\n" + "=" * 76)
    print(f"Method: {method}")
    print("=" * 76)
    for epoch in range(1, args.epochs + 1):
        if method == "nuisance_consistency":
            (
                train_loss,
                y_train,
                pred_train,
                _,
                consistency_details,
            ) = run_consistency_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args.consistency_weight,
            )
        else:
            train_loss, y_train, pred_train, _ = run_standard_epoch(
                model, train_loader, criterion, optimizer, device
            )
            consistency_details = {"ce_loss": train_loss, "consistency_loss": 0.0}

        val_loss, y_val, pred_val, _ = run_standard_epoch(
            model, val_loader, criterion, None, device
        )
        train_metrics = metrics_dict(y_train, pred_train, "train")
        val_metrics = metrics_dict(y_val, pred_val, "val")
        val_f1 = val_metrics["val_macro_f1"]
        scheduler.step(val_f1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce_loss": consistency_details["ce_loss"],
            "train_consistency_loss": consistency_details["consistency_loss"],
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
        }
        logs.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_f1={train_metrics['train_macro_f1']:.3f} "
            f"val_f1={val_f1:.3f} val_acc={val_metrics['val_accuracy']:.3f}"
        )

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "method": method,
                    "model_name": args.model,
                    "label_to_idx": label_to_idx,
                    "normalization_mean": stats.mean.tolist(),
                    "normalization_std": stats.std.tolist(),
                    "epoch": epoch,
                    "val_macro_f1": best_val_f1,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                    f"best val macro-F1={best_val_f1:.4f}"
                )
                break

    pd.DataFrame(logs).to_csv(
        method_dir / "train_log.csv", index=False, encoding="utf-8-sig"
    )
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metric_rows: List[dict] = []
    loaders = [
        ("train", train_eval_loader, train_df),
        ("val", val_loader, val_df),
        ("test", test_loader, test_df),
    ]
    for split_name, loader, frame in loaders:
        loss, y_true, y_pred, image_ids = run_standard_epoch(
            model, loader, criterion, None, device
        )
        split_metrics = metrics_dict(y_true, y_pred, split_name)
        metric_rows.append(
            {
                "method": method,
                "split": split_name,
                "loss": loss,
                "accuracy": split_metrics[f"{split_name}_accuracy"],
                "balanced_accuracy": split_metrics[
                    f"{split_name}_balanced_accuracy"
                ],
                "macro_f1": split_metrics[f"{split_name}_macro_f1"],
            }
        )
        save_split_predictions(
            method_dir,
            split_name,
            frame,
            image_ids,
            y_true,
            y_pred,
            idx_to_label,
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(
        method_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )

    augmentation_stats = nuisance.stats_dict() if nuisance is not None else None
    if augmentation_stats is not None:
        with (method_dir / "augmentation_stats.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(augmentation_stats, handle, ensure_ascii=False, indent=2)

    config = {
        "method": method,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "elapsed_seconds": time.time() - start_time,
        "rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "normalization": {"mean": stats.mean.tolist(), "std": stats.std.tolist()},
        "augmentation_stats": augmentation_stats,
        "args": vars(args),
    }
    with (method_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    test_row = metrics_frame[metrics_frame["split"].eq("test")].iloc[0].to_dict()
    test_row.update(
        {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "elapsed_seconds": config["elapsed_seconds"],
            "augmentation_application_rate": (
                augmentation_stats["application_rate"]
                if augmentation_stats is not None
                else None
            ),
        }
    )
    print("Final metrics:")
    print(metrics_frame.to_string(index=False))
    return test_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    if args.num_workers != 0:
        print(
            "WARNING: num_workers > 0 is allowed, but augmentation counters may not "
            "aggregate correctly across worker processes. Use 0 for the smoke test."
        )

    manifest_path = resolve(args.manifest)
    split_path = resolve(args.split_file)
    decomposition_dir = resolve(args.decomposition_dir)
    annotation_dir = resolve(args.annotation_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    library_path = decomposition_dir / "nuisance_vector_library.csv"
    if not library_path.exists():
        raise FileNotFoundError(library_path)

    full_df = load_manifest_and_split(manifest_path, split_path)
    split_audit = verify_split_structure(full_df)
    original_train = full_df[full_df["split"].eq("train")].copy().reset_index(drop=True)
    original_val = full_df[full_df["split"].eq("val")].copy().reset_index(drop=True)
    original_test = full_df[full_df["split"].eq("test")].copy().reset_index(drop=True)

    # Verify that decomposition uses exactly the unmodified train split before debug limits.
    library_ids = set(
        pd.read_csv(library_path, usecols=["image_id"], dtype={"image_id": str})[
            "image_id"
        ].astype(str)
    )
    train_ids = set(original_train["image_id"].astype(str))
    if library_ids != train_ids:
        raise ValueError(
            "Train/decomposition mismatch. The selected nuisance library must contain "
            "exactly the image IDs in this split's train partition. "
            f"Missing={len(train_ids - library_ids)}, extra={len(library_ids - train_ids)}"
        )

    config_path = decomposition_dir / "decomposition_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            decomposition_config = json.load(handle)
        configured_stem = Path(str(decomposition_config.get("split_file", ""))).stem
        if configured_stem and configured_stem != split_path.stem:
            raise ValueError(
                "The decomposition directory belongs to another split: "
                f"{configured_stem!r} != {split_path.stem!r}"
            )

    train_df = limit_frame(original_train, args.limit_train)
    val_df = limit_frame(original_val, args.limit_val)
    test_df = limit_frame(original_test, args.limit_test)

    print("Manifest:", manifest_path)
    print("Split:", split_path)
    print("Decomposition:", decomposition_dir)
    print("Output:", output_dir)
    print("Methods:", methods)
    print("Model:", args.model, "Pretrained:", args.pretrained)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("Full split audit:", split_audit)
    print(
        "Rows used:",
        {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    )
    print("Train label counts:")
    print(train_df["label_class"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int))

    set_seed(args.seed)
    stats = compute_train_stats(original_train, args.image_size)
    print("Normalization mean:", stats.mean.tolist())
    print("Normalization std:", stats.std.tolist())
    with (output_dir / "normalization_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"mean": stats.mean.tolist(), "std": stats.std.tolist()},
            handle,
            indent=2,
        )
    with (output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(split_audit, handle, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparison_rows: List[dict] = []
    for method in methods:
        comparison_rows.append(
            run_method(
                method=method,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                stats=stats,
                args=args,
                device=device,
                library_path=library_path,
                annotation_dir=annotation_dir,
                output_root=output_dir,
            )
        )

    comparison = pd.DataFrame(comparison_rows)
    preferred_columns = [
        "method",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "best_val_macro_f1",
        "best_epoch",
        "augmentation_application_rate",
        "elapsed_seconds",
    ]
    comparison = comparison[[column for column in preferred_columns if column in comparison.columns]]
    comparison.to_csv(
        output_dir / "smoke_comparison.csv", index=False, encoding="utf-8-sig"
    )
    print("\n" + "=" * 76)
    print("Smoke comparison (test split)")
    print("=" * 76)
    print(comparison.to_string(index=False))
    print("\nSaved to:", output_dir)
    print(
        "NOTE: This is a one-split, short-epoch smoke test. Use it to verify the "
        "pipeline, not as the final research conclusion."
    )


if __name__ == "__main__":
    main()


"""
python .\scripts\06_smoke_train_one_split.py `
  --split-file ".\data\splits\capture_lopo_holdout_galaxy_s24plus_vallight_L01.csv" `
  --decomposition-dir ".\outputs\decomposition\capture_lopo_holdout_galaxy_s24plus_vallight_L01" `
  --annotation-dir ".\data\raw\captures\annotations" `
  --output-dir ".\outputs\smoke\capture_lopo_holdout_galaxy_s24plus_vallight_L01_seed42" `
  --methods "baseline,colorjitter,nuisance,nuisance_consistency" `
  --model resnet18 `
  --epochs 5 `
  --patience 5 `
  --batch-size 8 `
  --image-size 224 `
  --num-workers 0 `
  --seed 42 `
  --use-class-weights `
  --nuisance-probability 0.5 `
  --phone-scale 1.0 `
  --light-scale 1.0 `
  --interaction-scale 0.5 `
  --residual-scale 0.0 `
  --feather-radius 24 `
  --consistency-weight 0.5
"""