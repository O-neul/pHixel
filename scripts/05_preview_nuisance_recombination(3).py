#!/usr/bin/env python
"""Create ROI-aware previews from train-only decomposed nuisance components.

The transform moves an image from its current structured nuisance state to a
sampled target state:

    delta = phone_scale * (target_phone - current_phone)
          + light_scale * (target_light - current_light)
          + interaction_scale * (target_interaction - current_interaction)
          + residual_scale * (target_residual - current_residual)

Background-region delta Lab is applied to the whole image. Contrast-region
(delta between reaction pad and background) is applied only inside the LabelMe
reaction-pad rectangle with a feathered edge. Therefore the reaction pad gets
background delta + contrast delta, while the rest of the image gets background
delta only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_NAMES = ["phone_effect", "light_effect", "interaction_effect", "residual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview ROI-aware nuisance recombination for one train-only LOPO fold."
    )
    parser.add_argument("--manifest", default="data/processed/capture_manifest.csv")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--decomposition-dir", required=True)
    parser.add_argument("--annotation-dir", default="data/raw/captures/annotations")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-label", default="roi")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--variants-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=["balanced_conditions", "sorted", "random"],
        default="balanced_conditions",
    )
    parser.add_argument(
        "--sampler",
        choices=["structured", "independent", "swap_row"],
        default="structured",
        help=(
            "structured: independently select target phone and light and use their matching interaction; "
            "independent: sample phone/light/interaction independently; "
            "swap_row: copy all target components from one sampled train row."
        ),
    )
    parser.add_argument("--phone-scale", type=float, default=1.0)
    parser.add_argument("--light-scale", type=float, default=1.0)
    parser.add_argument("--interaction-scale", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--overall-scale", type=float, default=1.0)
    parser.add_argument("--feather-radius", type=float, default=24.0)
    parser.add_argument(
        "--allow-same-condition",
        action="store_true",
        help="Permit sampled target phone/light to equal the source condition.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=360,
        help="Width of each tile in contact sheets; original output images keep full resolution.",
    )
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks required columns: {missing}")


def load_fold_data(manifest_path: Path, split_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        manifest_path,
        dtype={"image_id": str, "phone_id": str, "light_id": str, "image_path": str},
    )
    split = pd.read_csv(
        split_path,
        dtype={"image_id": str, "phone_id": str, "light_id": str},
    )
    require_columns(
        manifest,
        ["image_id", "image_path", "phone_id", "light_id", "ph_nominal"],
        manifest_path,
    )
    require_columns(split, ["image_id", "split"], split_path)
    merged = manifest.merge(
        split[["image_id", "split"]].drop_duplicates("image_id"),
        on="image_id",
        how="inner",
    )
    train = merged[merged["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No train rows were found after merging manifest and split")
    return train.reset_index(drop=True)


def choose_source_rows(frame: pd.DataFrame, count: int, mode: str, rng: random.Random) -> pd.DataFrame:
    if count <= 0 or count >= len(frame):
        return frame.copy().reset_index(drop=True)

    if mode == "sorted":
        return frame.sort_values(["phone_id", "light_id", "ph_nominal", "image_id"]).head(count).reset_index(drop=True)
    if mode == "random":
        indices = rng.sample(range(len(frame)), count)
        return frame.iloc[indices].reset_index(drop=True)

    # Round-robin over phone/light cells, preserving pH coverage inside each cell.
    groups = []
    for _, group in frame.sort_values(["ph_nominal", "image_id"]).groupby(
        ["phone_id", "light_id"], sort=True
    ):
        groups.append(group.reset_index(drop=True))

    selected_indices: list[int] = []
    positions = [0] * len(groups)
    while len(selected_indices) < count:
        progressed = False
        for group_index, group in enumerate(groups):
            position = positions[group_index]
            if position >= len(group):
                continue
            selected_indices.append(int(group.iloc[position].name))
            positions[group_index] += 1
            progressed = True
            if len(selected_indices) >= count:
                break
        if not progressed:
            break

    # The group row names were reset, so perform a stable merge by image_id instead.
    chosen_ids: list[str] = []
    positions = [0] * len(groups)
    while len(chosen_ids) < count:
        progressed = False
        for group_index, group in enumerate(groups):
            position = positions[group_index]
            if position >= len(group):
                continue
            chosen_ids.append(str(group.iloc[position]["image_id"]))
            positions[group_index] += 1
            progressed = True
            if len(chosen_ids) >= count:
                break
        if not progressed:
            break
    indexed = frame.set_index("image_id", drop=False)
    return indexed.loc[chosen_ids].reset_index(drop=True)


def load_library(decomposition_dir: Path) -> pd.DataFrame:
    library_path = decomposition_dir / "nuisance_vector_library.csv"
    if not library_path.exists():
        raise FileNotFoundError(library_path)
    library = pd.read_csv(
        library_path,
        dtype={"image_id": str, "phone_id": str, "light_id": str, "region": str},
    )
    required = ["image_id", "phone_id", "light_id", "ph_nominal", "region"]
    for component in COMPONENT_NAMES:
        required.extend([f"{component}_L", f"{component}_a", f"{component}_b"])
    require_columns(library, required, library_path)
    needed_regions = {"bg", "contrast"}
    missing_regions = needed_regions - set(library["region"].astype(str).unique())
    if missing_regions:
        raise ValueError(f"Vector library lacks required regions: {sorted(missing_regions)}")
    return library


def vector_from_row(row: pd.Series, component: str) -> np.ndarray:
    return np.array(
        [row[f"{component}_L"], row[f"{component}_a"], row[f"{component}_b"]],
        dtype=np.float64,
    )


def first_matching(frame: pd.DataFrame, **conditions: str) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= frame[column].astype(str).eq(str(value)).to_numpy()
    matched = frame.loc[mask]
    if matched.empty:
        raise KeyError(f"No vector row matched {conditions}")
    return matched.iloc[0]


def sample_target_spec(
    source: pd.Series,
    library: pd.DataFrame,
    sampler: str,
    rng: random.Random,
    allow_same_condition: bool,
) -> dict:
    base_region = library[library["region"].eq("bg")].copy()
    phones = sorted(base_region["phone_id"].astype(str).unique().tolist())
    lights = sorted(base_region["light_id"].astype(str).unique().tolist())
    interaction_cells = sorted(
        set(zip(base_region["phone_id"].astype(str), base_region["light_id"].astype(str)))
    )
    image_ids = sorted(base_region["image_id"].astype(str).unique().tolist())

    for _ in range(100):
        if sampler == "swap_row":
            target_image_id = rng.choice(image_ids)
            row = first_matching(base_region, image_id=target_image_id)
            target_phone = str(row["phone_id"])
            target_light = str(row["light_id"])
            interaction_phone = target_phone
            interaction_light = target_light
        elif sampler == "structured":
            target_phone = rng.choice(phones)
            target_light = rng.choice(lights)
            if (target_phone, target_light) not in interaction_cells:
                continue
            interaction_phone = target_phone
            interaction_light = target_light
            target_image_id = rng.choice(image_ids)
        else:
            target_phone = rng.choice(phones)
            target_light = rng.choice(lights)
            interaction_phone, interaction_light = rng.choice(interaction_cells)
            target_image_id = rng.choice(image_ids)

        same = (
            target_phone == str(source["phone_id"])
            and target_light == str(source["light_id"])
        )
        if allow_same_condition or not same:
            return {
                "target_phone": target_phone,
                "target_light": target_light,
                "interaction_phone": interaction_phone,
                "interaction_light": interaction_light,
                "target_residual_image_id": target_image_id,
            }

    raise RuntimeError("Could not sample a target condition different from the source")


def get_component_vectors(
    library_region: pd.DataFrame,
    source_image_id: str,
    target_spec: dict,
    sampler: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    current_row = first_matching(library_region, image_id=source_image_id)

    if sampler == "swap_row":
        target_row = first_matching(
            library_region,
            image_id=target_spec["target_residual_image_id"],
        )
        target = {component: vector_from_row(target_row, component) for component in COMPONENT_NAMES}
    else:
        phone_row = first_matching(
            library_region,
            phone_id=target_spec["target_phone"],
        )
        light_row = first_matching(
            library_region,
            light_id=target_spec["target_light"],
        )
        interaction_row = first_matching(
            library_region,
            phone_id=target_spec["interaction_phone"],
            light_id=target_spec["interaction_light"],
        )
        residual_row = first_matching(
            library_region,
            image_id=target_spec["target_residual_image_id"],
        )
        target = {
            "phone_effect": vector_from_row(phone_row, "phone_effect"),
            "light_effect": vector_from_row(light_row, "light_effect"),
            "interaction_effect": vector_from_row(interaction_row, "interaction_effect"),
            "residual": vector_from_row(residual_row, "residual"),
        }

    current = {component: vector_from_row(current_row, component) for component in COMPONENT_NAMES}
    return current, target


def combined_delta(
    current: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    scales: dict[str, float],
    overall_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    component_deltas: dict[str, np.ndarray] = {}
    total = np.zeros(3, dtype=np.float64)
    for component in COMPONENT_NAMES:
        delta = scales[component] * (target[component] - current[component])
        component_deltas[component] = overall_scale * delta
        total += component_deltas[component]
    return total, component_deltas


# ---- Color conversion (D65, CIE Lab) --------------------------------------

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
    f = np.where(
        scaled > delta**3,
        np.cbrt(scaled),
        scaled / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def lab_to_srgb_unclipped(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    delta = 6.0 / 29.0

    def inverse_f(value: np.ndarray) -> np.ndarray:
        return np.where(value > delta, value**3, 3.0 * delta**2 * (value - 4.0 / 29.0))

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
    # Do not raise negative linear values to a fractional power.
    positive = np.maximum(linear, 0.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * positive ** (1.0 / 2.4) - 0.055,
    )
    return srgb


def load_labelme_mask(
    annotation_path: Path,
    image_size: tuple[int, int],
    roi_label: str,
    feather_radius: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    shapes = annotation.get("shapes", [])
    candidates = [shape for shape in shapes if str(shape.get("label", "")) == roi_label]
    if not candidates:
        raise ValueError(f"No LabelMe shape with label {roi_label!r}: {annotation_path}")

    shape = candidates[0]
    points = shape.get("points", [])
    if len(points) < 2:
        raise ValueError(f"ROI shape has fewer than two points: {annotation_path}")

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    annotation_width = float(annotation.get("imageWidth") or image_size[0])
    annotation_height = float(annotation.get("imageHeight") or image_size[1])
    scale_x = image_size[0] / annotation_width
    scale_y = image_size[1] / annotation_height
    x1, x2 = x1 * scale_x, x2 * scale_x
    y1, y2 = y1 * scale_y, y2 * scale_y

    hard = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(hard)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    hard_mask = np.asarray(hard, dtype=np.float64) / 255.0

    feathered = hard
    if feather_radius > 0:
        feathered = hard.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    feather_mask = np.asarray(feathered, dtype=np.float64) / 255.0
    return hard_mask, feather_mask, (x1, y1, x2, y2)


def clipping_metrics(
    lab_raw: np.ndarray,
    rgb_raw: np.ndarray,
    hard_roi: np.ndarray,
) -> dict[str, float]:
    lab_clip = (lab_raw[..., 0] < 0.0) | (lab_raw[..., 0] > 100.0)
    rgb_clip = np.any((rgb_raw < 0.0) | (rgb_raw > 1.0), axis=-1)
    pad = hard_roi > 0.5
    outside = ~pad

    def ratio(mask: np.ndarray, region: np.ndarray) -> float:
        denominator = int(region.sum())
        if denominator == 0:
            return float("nan")
        return float(np.logical_and(mask, region).sum() / denominator)

    all_region = np.ones_like(pad, dtype=bool)
    return {
        "all_lab_clipped_ratio": ratio(lab_clip, all_region),
        "all_rgb_gamut_clipped_ratio": ratio(rgb_clip, all_region),
        "pad_lab_clipped_ratio": ratio(lab_clip, pad),
        "pad_rgb_gamut_clipped_ratio": ratio(rgb_clip, pad),
        "outside_lab_clipped_ratio": ratio(lab_clip, outside),
        "outside_rgb_gamut_clipped_ratio": ratio(rgb_clip, outside),
        "pad_pixel_ratio": float(pad.mean()),
    }


def annotate_tile(image: Image.Image, lines: Sequence[str], width: int) -> Image.Image:
    original = image.convert("RGB")
    ratio = width / original.width
    height = max(1, int(round(original.height * ratio)))
    resized = original.resize((width, height), Image.Resampling.LANCZOS)
    text_height = 18 * len(lines) + 12
    tile = Image.new("RGB", (width, height + text_height), "white")
    tile.paste(resized, (0, 0))
    draw = ImageDraw.Draw(tile)
    y = height + 4
    for line in lines:
        draw.text((6, y), str(line), fill="black")
        y += 18
    return tile


def make_contact_sheet(
    original: Image.Image,
    variants: list[tuple[Image.Image, list[str]]],
    output_path: Path,
    tile_width: int,
) -> None:
    tiles = [annotate_tile(original, ["original"], tile_width)]
    tiles.extend(annotate_tile(image, lines, tile_width) for image, lines in variants)
    max_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (tile_width * len(tiles), max_height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * tile_width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def safe_filename(text: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in text)


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    split_path = resolve(args.split_file)
    decomposition_dir = resolve(args.decomposition_dir)
    annotation_dir = resolve(args.annotation_dir)
    output_dir = resolve(args.output_dir)
    individual_dir = output_dir / "individual"
    contact_dir = output_dir / "contact_sheets"
    individual_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train = load_fold_data(manifest_path, split_path)
    library = load_library(decomposition_dir)

    config_path = decomposition_dir / "decomposition_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            decomposition_config = json.load(handle)
        configured_split = Path(str(decomposition_config.get("split_file", ""))).stem
        if configured_split and configured_split != split_path.stem:
            raise ValueError(
                "The decomposition directory was created from a different split. "
                f"Configured split={configured_split!r}, supplied split={split_path.stem!r}."
            )

    train_ids = set(train["image_id"].astype(str))
    library_ids = set(library["image_id"].astype(str))
    missing_library = sorted(train_ids - library_ids)
    extra_library = sorted(library_ids - train_ids)
    if missing_library or extra_library:
        raise ValueError(
            "The nuisance vector library and supplied train split do not contain exactly "
            "the same image IDs. This usually means the wrong decomposition directory was selected. "
            f"Missing from library={len(missing_library)}, extra in library={len(extra_library)}. "
            f"Examples missing={missing_library[:3]}, extra={extra_library[:3]}"
        )

    selected = choose_source_rows(train, args.max_images, args.selection, rng)
    scales = {
        "phone_effect": float(args.phone_scale),
        "light_effect": float(args.light_scale),
        "interaction_effect": float(args.interaction_scale),
        "residual": float(args.residual_scale),
    }

    metadata_rows: list[dict] = []
    generated = 0

    for source_index, source in selected.iterrows():
        source_image_id = str(source["image_id"])
        image_path = resolve(str(source["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        original = Image.open(image_path).convert("RGB")
        rgb = np.asarray(original, dtype=np.float64) / 255.0
        lab = srgb_to_lab(rgb)

        annotation_path = annotation_dir / f"{Path(image_path).stem}.json"
        hard_roi, feather_roi, bbox = load_labelme_mask(
            annotation_path,
            original.size,
            args.roi_label,
            args.feather_radius,
        )

        contact_variants: list[tuple[Image.Image, list[str]]] = []
        used_target_keys: set[tuple[str, ...]] = set()
        for variant_index in range(1, args.variants_per_image + 1):
            target_spec = None
            for _ in range(100):
                candidate = sample_target_spec(
                    source,
                    library,
                    args.sampler,
                    rng,
                    args.allow_same_condition,
                )
                if args.sampler == "swap_row":
                    target_key = (str(candidate["target_residual_image_id"]),)
                else:
                    target_key = (
                        str(candidate["target_phone"]),
                        str(candidate["target_light"]),
                        str(candidate["interaction_phone"]),
                        str(candidate["interaction_light"]),
                    )
                if target_key not in used_target_keys:
                    target_spec = candidate
                    used_target_keys.add(target_key)
                    break
            if target_spec is None:
                target_spec = candidate

            region_deltas: dict[str, np.ndarray] = {}
            component_deltas_by_region: dict[str, dict[str, np.ndarray]] = {}
            for region in ["bg", "contrast"]:
                region_library = library[library["region"].eq(region)].copy()
                current, target = get_component_vectors(
                    region_library,
                    source_image_id,
                    target_spec,
                    args.sampler,
                )
                total_delta, component_deltas = combined_delta(
                    current,
                    target,
                    scales,
                    args.overall_scale,
                )
                region_deltas[region] = total_delta
                component_deltas_by_region[region] = component_deltas

            bg_delta = region_deltas["bg"]
            contrast_delta = region_deltas["contrast"]
            lab_raw = (
                lab
                + bg_delta.reshape(1, 1, 3)
                + feather_roi[..., None] * contrast_delta.reshape(1, 1, 3)
            )
            rgb_raw = lab_to_srgb_unclipped(lab_raw)
            metrics = clipping_metrics(lab_raw, rgb_raw, hard_roi)
            rgb_output = np.clip(rgb_raw, 0.0, 1.0)
            output_image = Image.fromarray(np.round(rgb_output * 255.0).astype(np.uint8), mode="RGB")

            filename = (
                f"{source_index + 1:03d}_{safe_filename(source_image_id)}"
                f"__v{variant_index:02d}"
                f"__to_{safe_filename(target_spec['target_phone'])}_{safe_filename(target_spec['target_light'])}.png"
            )
            output_path = individual_dir / filename
            output_image.save(output_path)
            generated += 1

            metadata = {
                "source_image_id": source_image_id,
                "source_image_path": str(source["image_path"]),
                "source_phone": str(source["phone_id"]),
                "source_light": str(source["light_id"]),
                "source_ph": float(source["ph_nominal"]),
                "variant_index": variant_index,
                "sampler": args.sampler,
                **target_spec,
                "phone_scale": args.phone_scale,
                "light_scale": args.light_scale,
                "interaction_scale": args.interaction_scale,
                "residual_scale": args.residual_scale,
                "overall_scale": args.overall_scale,
                "bg_delta_L": float(bg_delta[0]),
                "bg_delta_a": float(bg_delta[1]),
                "bg_delta_b": float(bg_delta[2]),
                "bg_delta_E": float(np.linalg.norm(bg_delta)),
                "contrast_delta_L": float(contrast_delta[0]),
                "contrast_delta_a": float(contrast_delta[1]),
                "contrast_delta_b": float(contrast_delta[2]),
                "contrast_delta_E": float(np.linalg.norm(contrast_delta)),
                "pad_total_delta_L": float(bg_delta[0] + contrast_delta[0]),
                "pad_total_delta_a": float(bg_delta[1] + contrast_delta[1]),
                "pad_total_delta_b": float(bg_delta[2] + contrast_delta[2]),
                "pad_total_delta_E": float(np.linalg.norm(bg_delta + contrast_delta)),
                "roi_x1": bbox[0],
                "roi_y1": bbox[1],
                "roi_x2": bbox[2],
                "roi_y2": bbox[3],
                "output_image": output_path.relative_to(PROJECT_ROOT).as_posix()
                if PROJECT_ROOT in output_path.resolve().parents
                else str(output_path),
                **metrics,
            }

            for region, component_deltas in component_deltas_by_region.items():
                for component, delta in component_deltas.items():
                    short = component.replace("_effect", "")
                    metadata[f"{region}_{short}_delta_L"] = float(delta[0])
                    metadata[f"{region}_{short}_delta_a"] = float(delta[1])
                    metadata[f"{region}_{short}_delta_b"] = float(delta[2])
                    metadata[f"{region}_{short}_delta_E"] = float(np.linalg.norm(delta))

            metadata_rows.append(metadata)
            contact_variants.append(
                (
                    output_image,
                    [
                        f"v{variant_index}: {source['phone_id']}/{source['light_id']} -> {target_spec['target_phone']}/{target_spec['target_light']}",
                        f"bg ΔE={metadata['bg_delta_E']:.2f}, contrast ΔE={metadata['contrast_delta_E']:.2f}",
                        f"pad RGB clip={100.0 * metrics['pad_rgb_gamut_clipped_ratio']:.2f}%",
                    ],
                )
            )

        contact_path = contact_dir / f"{source_index + 1:03d}_{safe_filename(source_image_id)}.png"
        make_contact_sheet(original, contact_variants, contact_path, args.image_width)

    metadata_frame = pd.DataFrame(metadata_rows)
    metadata_path = output_dir / "preview_metadata.csv"
    metadata_frame.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    summary_rows: list[dict] = []
    for column in [
        "all_lab_clipped_ratio",
        "all_rgb_gamut_clipped_ratio",
        "pad_lab_clipped_ratio",
        "pad_rgb_gamut_clipped_ratio",
        "outside_lab_clipped_ratio",
        "outside_rgb_gamut_clipped_ratio",
        "bg_delta_E",
        "contrast_delta_E",
        "pad_total_delta_E",
    ]:
        values = pd.to_numeric(metadata_frame[column], errors="coerce").dropna().to_numpy(float)
        summary_rows.append(
            {
                "metric": column,
                "n": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else float("nan"),
                "median": float(np.median(values)) if len(values) else float("nan"),
                "q95": float(np.quantile(values, 0.95)) if len(values) else float("nan"),
                "maximum": float(np.max(values)) if len(values) else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "preview_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "manifest": str(manifest_path),
        "split_file": str(split_path),
        "decomposition_dir": str(decomposition_dir),
        "annotation_dir": str(annotation_dir),
        "train_rows_available": len(train),
        "selected_source_images": len(selected),
        "generated_variants": generated,
        "settings": vars(args),
        "important": (
            "Only train rows and the matching train-only nuisance vector library were used. "
            "No validation-light or held-out-phone image was used to generate previews."
        ),
    }
    with (output_dir / "preview_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("Selected train images:", len(selected))
    print("Generated variants:", generated)
    print("Contact sheets:", contact_dir)
    print("Metadata:", metadata_path)
    if not metadata_frame.empty:
        print(
            "Max pad RGB gamut-clipped ratio:",
            f"{100.0 * metadata_frame['pad_rgb_gamut_clipped_ratio'].max():.4f}%",
        )
        print(
            "Max outside RGB gamut-clipped ratio:",
            f"{100.0 * metadata_frame['outside_rgb_gamut_clipped_ratio'].max():.4f}%",
        )
        print(
            "Max pad total deltaE:",
            f"{metadata_frame['pad_total_delta_E'].max():.4f}",
        )


if __name__ == "__main__":
    main()


"""
python .\scripts\05_preview_nuisance_recombination.py `
  --split-file ".\data\splits\capture_lopo_holdout_galaxy_s24plus_vallight_L01.csv" `
  --decomposition-dir ".\outputs\decomposition\capture_lopo_holdout_galaxy_s24plus_vallight_L01" `
  --annotation-dir ".\data\raw\captures\annotations" `
  --output-dir ".\outputs\previews\capture_lopo_holdout_galaxy_s24plus_vallight_L01" `
  --max-images 12 `
  --variants-per-image 4 `
  --seed 42 `
  --sampler structured `
  --phone-scale 1.0 `
  --light-scale 1.0 `
  --interaction-scale 0.5 `
  --residual-scale 0.0 `
  --feather-radius 24
"""