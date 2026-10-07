#!/usr/bin/env python
"""
09_make_local_context_inputs.py

Create and audit local-context image inputs for the DIC nuisance-DG project.

Spatial design
--------------
The uploaded nuisance-preview script contains one spatial annotation:
the LabelMe ROI rectangle. Its background effect is applied to the entire image,
so there is no separate saved background rectangle to recover.

This script therefore constructs an explicit local background rectangle:
- same size as the LabelMe ROI
- placed to the left or right of the ROI
- separated by a configurable gap
- automatically falls back to the opposite side if the requested side does not fit

The final local-context crop is:
    union(ROI, local background rectangle) + configurable margin

Outputs
-------
- overlays/: original image with ROI, background, and final crop boxes
- crops_raw/: local-context crops at native resolution
- model_inputs/: letterboxed model inputs, default 224 x 224
- contact_sheets/: pH comparison sheets grouped by phone and light
- local_context_manifest.csv: paths and transformed region coordinates
- audit_summary.csv: clipping, area, and fallback diagnostics

No OCR or automatic text detection is performed. Contact sheets must be visually
checked to confirm that handwritten pH, phone names, and light labels are absent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ROI + adjacent-background local-context crops and "
            "visual audit sheets."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/capture_manifest.csv",
        help="Canonical capture manifest.",
    )
    parser.add_argument(
        "--annotation-dir",
        default="data/raw/captures/annotations",
        help="Directory containing LabelMe JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help=(
            "Optional split CSV. When supplied, its split labels are merged "
            "into the output manifest."
        ),
    )
    parser.add_argument(
        "--include-splits",
        default="train,val,test",
        help="Comma-separated split labels to process when --split-file is used.",
    )
    parser.add_argument(
        "--roi-label",
        default="roi",
        help="LabelMe shape label for the strip/reaction ROI.",
    )
    parser.add_argument(
        "--bg-side",
        choices=["left", "right", "auto"],
        default="left",
        help="Preferred side for the adjacent local background rectangle.",
    )
    parser.add_argument(
        "--bg-width-ratio",
        type=float,
        default=1.0,
        help="Background width divided by ROI width.",
    )
    parser.add_argument(
        "--bg-height-ratio",
        type=float,
        default=1.0,
        help="Background height divided by ROI height.",
    )
    parser.add_argument(
        "--bg-gap-ratio",
        type=float,
        default=0.25,
        help="Horizontal gap divided by ROI width.",
    )
    parser.add_argument(
        "--context-margin-ratio",
        type=float,
        default=0.15,
        help=(
            "Margin added around the ROI/background union, relative to the "
            "union width and height."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square model-input size.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["letterbox", "stretch"],
        default="letterbox",
        help="How local crops are converted to square model inputs.",
    )
    parser.add_argument(
        "--letterbox-fill",
        choices=["median", "mean", "white", "black"],
        default="median",
        help="Padding color for letterbox mode.",
    )
    parser.add_argument(
        "--selection",
        choices=["audit_grid", "all", "random"],
        default="audit_grid",
        help=(
            "audit_grid: requested pH values across every phone/light cell; "
            "all: every manifest row; random: random rows."
        ),
    )
    parser.add_argument(
        "--audit-ph",
        default="2,6,8,9,12",
        help="Comma-separated pH values used by audit_grid.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help=(
            "Optional maximum after selection. 0 means no limit. "
            "For random selection, a positive value is required."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--contact-tile-size",
        type=int,
        default=224,
        help="Displayed model-input tile size in contact sheets.",
    )
    parser.add_argument(
        "--save-original-size-overlay",
        action="store_true",
        help=(
            "Save overlays at original resolution. By default overlays are "
            "downscaled to at most 1400 pixels on the long side."
        ),
    )
    return parser.parse_args()


def resolve_existing(path_text: str | Path) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PROJECT_ROOT / path, Path.cwd() / path])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path not found: {path_text}")


def resolve_output(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
) -> str:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(
        f"Could not identify {description}. Tried {list(candidates)}. "
        f"Available columns: {list(frame.columns)}"
    )


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in text
    )


def parse_float_list(raw: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid numeric list: {raw}") from exc
    if not values:
        raise ValueError("At least one pH value is required.")
    return values


def parse_string_list(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return list(dict.fromkeys(values))


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "bg_width_ratio": args.bg_width_ratio,
        "bg_height_ratio": args.bg_height_ratio,
        "image_size": args.image_size,
        "contact_tile_size": args.contact_tile_size,
    }
    for name, value in positive_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0.")

    nonnegative_values = {
        "bg_gap_ratio": args.bg_gap_ratio,
        "context_margin_ratio": args.context_margin_ratio,
    }
    for name, value in nonnegative_values.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0.")

    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0.")
    if args.selection == "random" and args.max_images <= 0:
        raise ValueError(
            "--selection random requires a positive --max-images."
        )


def load_manifest(
    manifest_path: Path,
    split_path: Path | None,
    include_splits: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest = pd.read_csv(manifest_path)

    image_id_col = find_column(
        manifest, ["image_id", "capture_id"], "image identifier"
    )
    image_path_col = find_column(
        manifest,
        ["image_path", "image_path_declared", "resolved_image_path"],
        "image path",
    )
    phone_col = find_column(
        manifest, ["phone_id", "phone", "phone_label"], "phone column"
    )
    light_col = find_column(
        manifest, ["light_id", "light", "light_label"], "light column"
    )
    ph_col = find_column(
        manifest, ["ph_nominal", "ph_value", "ph"], "pH column"
    )

    frame = manifest.copy()
    frame["__image_id__"] = frame[image_id_col].astype(str)
    frame["__image_path__"] = frame[image_path_col].astype(str)
    frame["__phone__"] = frame[phone_col].astype(str)
    frame["__light__"] = frame[light_col].astype(str)
    frame["__ph__"] = pd.to_numeric(frame[ph_col], errors="coerce")

    if frame["__ph__"].isna().any():
        bad = frame.loc[frame["__ph__"].isna(), "__image_id__"].tolist()
        raise ValueError(f"Non-numeric pH values. Examples: {bad[:10]}")

    if frame["__image_id__"].duplicated().any():
        duplicates = frame.loc[
            frame["__image_id__"].duplicated(keep=False), "__image_id__"
        ].tolist()
        raise ValueError(
            f"Manifest image identifiers are not unique. Examples: {duplicates[:10]}"
        )

    split_column = None
    if split_path is not None:
        split_frame = pd.read_csv(split_path)
        split_id_col = find_column(
            split_frame, ["image_id", "capture_id"], "split image identifier"
        )
        split_column = find_column(
            split_frame, ["split", "partition", "subset"], "split column"
        )

        split_min = split_frame[[split_id_col, split_column]].copy()
        split_min.columns = ["__image_id__", "__split__"]
        split_min["__image_id__"] = split_min["__image_id__"].astype(str)
        split_min["__split__"] = (
            split_min["__split__"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"validation": "val", "valid": "val"})
        )

        frame = frame.merge(
            split_min,
            on="__image_id__",
            how="inner",
            validate="one_to_one",
        )
        frame = frame.loc[
            frame["__split__"].isin(include_splits)
        ].copy()
    else:
        frame["__split__"] = "all"

    if frame.empty:
        raise ValueError("No rows remain after loading/filtering the manifest.")

    return frame.reset_index(drop=True), {
        "image_id_column": image_id_col,
        "image_path_column": image_path_col,
        "phone_column": phone_col,
        "light_column": light_col,
        "ph_column": ph_col,
        "split_column": split_column or "",
    }


def select_rows(
    frame: pd.DataFrame,
    mode: str,
    audit_ph: Sequence[float],
    max_images: int,
    seed: int,
) -> pd.DataFrame:
    if mode == "all":
        selected = frame.copy()
    elif mode == "random":
        rng = random.Random(seed)
        count = min(max_images, len(frame))
        indices = rng.sample(range(len(frame)), count)
        selected = frame.iloc[indices].copy()
    else:
        masks = [
            np.isclose(frame["__ph__"].to_numpy(dtype=float), value)
            for value in audit_ph
        ]
        keep = np.any(np.vstack(masks), axis=0)
        selected = frame.loc[keep].copy()
        selected = selected.sort_values(
            ["__phone__", "__light__", "__ph__", "__image_id__"]
        )

    if max_images > 0 and len(selected) > max_images:
        if mode == "audit_grid":
            # Round-robin across phone/light cells to retain condition coverage.
            grouped = [
                group.reset_index(drop=True)
                for _, group in selected.groupby(
                    ["__phone__", "__light__"], sort=True
                )
            ]
            chosen_ids: list[str] = []
            positions = [0] * len(grouped)
            while len(chosen_ids) < max_images:
                progressed = False
                for group_index, group in enumerate(grouped):
                    position = positions[group_index]
                    if position >= len(group):
                        continue
                    chosen_ids.append(
                        str(group.iloc[position]["__image_id__"])
                    )
                    positions[group_index] += 1
                    progressed = True
                    if len(chosen_ids) >= max_images:
                        break
                if not progressed:
                    break
            selected = (
                selected.set_index("__image_id__", drop=False)
                .loc[chosen_ids]
                .reset_index(drop=True)
            )
        else:
            selected = selected.head(max_images).copy()

    if selected.empty:
        raise ValueError("No rows matched the requested selection.")
    return selected.reset_index(drop=True)


def resolve_image_path(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PROJECT_ROOT / path, Path.cwd() / path])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found: {path_text}")


def annotation_path_for(
    annotation_dir: Path,
    image_path: Path,
    image_id: str,
) -> Path:
    candidates = [
        annotation_dir / f"{image_path.stem}.json",
        annotation_dir / f"{image_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Annotation not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_roi_bbox(
    annotation_path: Path,
    image_size: tuple[int, int],
    roi_label: str,
) -> tuple[float, float, float, float]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    candidates = [
        shape
        for shape in annotation.get("shapes", [])
        if str(shape.get("label", "")) == roi_label
    ]
    if not candidates:
        raise ValueError(
            f"No LabelMe shape labelled {roi_label!r}: {annotation_path}"
        )

    shape = candidates[0]
    points = shape.get("points", [])
    if len(points) < 2:
        raise ValueError(f"ROI contains fewer than two points: {annotation_path}")

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]

    annotation_width = float(
        annotation.get("imageWidth") or image_size[0]
    )
    annotation_height = float(
        annotation.get("imageHeight") or image_size[1]
    )
    if annotation_width <= 0 or annotation_height <= 0:
        raise ValueError(f"Invalid annotation dimensions: {annotation_path}")

    scale_x = image_size[0] / annotation_width
    scale_y = image_size[1] / annotation_height

    x1 = min(xs) * scale_x
    x2 = max(xs) * scale_x
    y1 = min(ys) * scale_y
    y2 = max(ys) * scale_y

    x1 = float(np.clip(x1, 0, image_size[0]))
    x2 = float(np.clip(x2, 0, image_size[0]))
    y1 = float(np.clip(y1, 0, image_size[1]))
    y2 = float(np.clip(y2, 0, image_size[1]))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate ROI bbox: {annotation_path}")

    return x1, y1, x2, y2


def candidate_bg_bbox(
    roi: tuple[float, float, float, float],
    side: str,
    width_ratio: float,
    height_ratio: float,
    gap_ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    roi_width = x2 - x1
    roi_height = y2 - y1

    bg_width = roi_width * width_ratio
    bg_height = roi_height * height_ratio
    gap = roi_width * gap_ratio

    center_y = (y1 + y2) / 2.0
    bg_y1 = center_y - bg_height / 2.0
    bg_y2 = center_y + bg_height / 2.0

    if side == "left":
        bg_x2 = x1 - gap
        bg_x1 = bg_x2 - bg_width
    elif side == "right":
        bg_x1 = x2 + gap
        bg_x2 = bg_x1 + bg_width
    else:
        raise ValueError(f"Invalid side: {side}")

    return bg_x1, bg_y1, bg_x2, bg_y2


def bbox_fits(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> bool:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    return x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height


def clipped_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return (
        float(np.clip(x1, 0, width)),
        float(np.clip(y1, 0, height)),
        float(np.clip(x2, 0, width)),
        float(np.clip(y2, 0, height)),
    )


def choose_bg_bbox(
    roi: tuple[float, float, float, float],
    image_size: tuple[int, int],
    preferred_side: str,
    width_ratio: float,
    height_ratio: float,
    gap_ratio: float,
) -> tuple[tuple[float, float, float, float], str, bool, bool]:
    if preferred_side == "auto":
        candidates = ["left", "right"]
    else:
        opposite = "right" if preferred_side == "left" else "left"
        candidates = [preferred_side, opposite]

    for index, side in enumerate(candidates):
        bbox = candidate_bg_bbox(
            roi, side, width_ratio, height_ratio, gap_ratio
        )
        if bbox_fits(bbox, image_size):
            return bbox, side, index > 0, False

    # Neither side fully fits. Choose the side with the largest valid area.
    scored: list[tuple[float, str, tuple[float, float, float, float]]] = []
    for side in ["left", "right"]:
        raw = candidate_bg_bbox(
            roi, side, width_ratio, height_ratio, gap_ratio
        )
        clipped = clipped_bbox(raw, image_size)
        x1, y1, x2, y2 = clipped
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        scored.append((area, side, clipped))

    area, side, bbox = max(scored, key=lambda value: value[0])
    if area <= 0:
        raise ValueError("Could not place a non-empty background rectangle.")

    fallback = preferred_side != "auto" and side != preferred_side
    return bbox, side, fallback, True


def union_bbox(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def expand_bbox(
    bbox: tuple[float, float, float, float],
    margin_ratio: float,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    return clipped_bbox(
        (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y),
        image_size,
    )


def integer_crop_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1 = max(0, int(math.floor(bbox[0])))
    y1 = max(0, int(math.floor(bbox[1])))
    x2 = min(width, int(math.ceil(bbox[2])))
    y2 = min(height, int(math.ceil(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate crop bbox: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def local_bbox(
    global_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    crop_x1, crop_y1, _, _ = crop_bbox
    return (
        global_bbox[0] - crop_x1,
        global_bbox[1] - crop_y1,
        global_bbox[2] - crop_x1,
        global_bbox[3] - crop_y1,
    )


def determine_fill_color(
    crop: Image.Image,
    mode: str,
) -> tuple[int, int, int]:
    if mode == "white":
        return 255, 255, 255
    if mode == "black":
        return 0, 0, 0

    array = np.asarray(crop.convert("RGB"), dtype=np.float64)
    if mode == "mean":
        values = np.mean(array.reshape(-1, 3), axis=0)
    else:
        values = np.median(array.reshape(-1, 3), axis=0)
    return tuple(int(round(value)) for value in values)


def make_model_input(
    crop: Image.Image,
    image_size: int,
    resize_mode: str,
    fill_mode: str,
) -> tuple[
    Image.Image,
    dict[str, float],
]:
    crop = crop.convert("RGB")

    if resize_mode == "stretch":
        resized = crop.resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        return resized, {
            "resize_scale_x": image_size / crop.width,
            "resize_scale_y": image_size / crop.height,
            "resize_offset_x": 0.0,
            "resize_offset_y": 0.0,
        }

    scale = min(image_size / crop.width, image_size / crop.height)
    resized_width = max(1, int(round(crop.width * scale)))
    resized_height = max(1, int(round(crop.height * scale)))
    resized = crop.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )

    fill = determine_fill_color(crop, fill_mode)
    canvas = Image.new("RGB", (image_size, image_size), fill)
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas, {
        "resize_scale_x": scale,
        "resize_scale_y": scale,
        "resize_offset_x": float(offset_x),
        "resize_offset_y": float(offset_y),
    }


def transform_bbox_to_model(
    bbox_in_crop: tuple[float, float, float, float],
    transform: dict[str, float],
) -> tuple[float, float, float, float]:
    return (
        bbox_in_crop[0] * transform["resize_scale_x"]
        + transform["resize_offset_x"],
        bbox_in_crop[1] * transform["resize_scale_y"]
        + transform["resize_offset_y"],
        bbox_in_crop[2] * transform["resize_scale_x"]
        + transform["resize_offset_x"],
        bbox_in_crop[3] * transform["resize_scale_y"]
        + transform["resize_offset_y"],
    )


def draw_overlay(
    image: Image.Image,
    roi: tuple[float, float, float, float],
    bg: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    labels: Sequence[str],
    keep_original_size: bool,
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    line_width = max(3, int(round(max(image.size) / 500)))
    draw.rectangle(roi, outline="red", width=line_width)
    draw.rectangle(bg, outline="blue", width=line_width)
    draw.rectangle(crop, outline="green", width=line_width)

    text_lines = [
        "red: ROI",
        "blue: local background",
        "green: final context crop",
        *labels,
    ]
    text_height = 22 * len(text_lines) + 10
    panel_width = min(overlay.width, 760)
    draw.rectangle(
        [0, 0, panel_width, text_height],
        fill=(255, 255, 255),
    )
    y = 5
    for line in text_lines:
        draw.text((8, y), str(line), fill="black")
        y += 22

    if keep_original_size or max(overlay.size) <= 1400:
        return overlay

    scale = 1400 / max(overlay.size)
    return overlay.resize(
        (
            max(1, int(round(overlay.width * scale))),
            max(1, int(round(overlay.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def touches_edge(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    tolerance: float = 1.0,
) -> bool:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return (
        x1 <= tolerance
        or y1 <= tolerance
        or x2 >= width - tolerance
        or y2 >= height - tolerance
    )


def add_caption(
    image: Image.Image,
    lines: Sequence[str],
    tile_size: int,
) -> Image.Image:
    source = image.convert("RGB")
    if source.size != (tile_size, tile_size):
        source = source.resize(
            (tile_size, tile_size), Image.Resampling.LANCZOS
        )

    line_height = 18
    panel_height = line_height * len(lines) + 10
    tile = Image.new(
        "RGB",
        (tile_size, tile_size + panel_height),
        "white",
    )
    tile.paste(source, (0, 0))
    draw = ImageDraw.Draw(tile)
    y = tile_size + 4
    for line in lines:
        draw.text((5, y), str(line), fill="black")
        y += line_height
    return tile


def make_contact_sheets(
    records: pd.DataFrame,
    output_dir: Path,
    audit_ph: Sequence[float],
    tile_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for (phone, light), group in records.groupby(
        ["phone", "light"], sort=True
    ):
        tiles: list[Image.Image] = []
        for ph_value in audit_ph:
            matched = group.loc[np.isclose(group["ph_nominal"], ph_value)]
            if matched.empty:
                continue
            row = matched.iloc[0]
            image = Image.open(row["model_input_path"]).convert("RGB")
            tiles.append(
                add_caption(
                    image,
                    [
                        f"{phone} / {light}",
                        f"pH {ph_value:g}",
                        f"bg={row['bg_side_used']}",
                    ],
                    tile_size,
                )
            )

        if not tiles:
            continue

        sheet_width = tile_size * len(tiles)
        sheet_height = max(tile.height for tile in tiles)
        sheet = Image.new(
            "RGB",
            (sheet_width, sheet_height),
            "white",
        )
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (index * tile_size, 0))

        filename = (
            f"{safe_name(phone)}__{safe_name(light)}"
            f"__ph_grid.png"
        )
        sheet.save(output_dir / filename)


def main() -> None:
    args = parse_args()
    validate_args(args)

    manifest_path = resolve_existing(args.manifest)
    annotation_dir = resolve_existing(args.annotation_dir)
    split_path = (
        resolve_existing(args.split_file)
        if args.split_file
        else None
    )
    output_dir = resolve_output(args.output_dir)

    overlay_dir = output_dir / "overlays"
    crop_dir = output_dir / "crops_raw"
    model_dir = output_dir / "model_inputs"
    contact_dir = output_dir / "contact_sheets"
    for directory in [overlay_dir, crop_dir, model_dir, contact_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    include_splits = parse_string_list(args.include_splits)
    audit_ph = parse_float_list(args.audit_ph)

    frame, column_map = load_manifest(
        manifest_path,
        split_path,
        include_splits,
    )
    selected = select_rows(
        frame,
        args.selection,
        audit_ph,
        args.max_images,
        args.seed,
    )

    print(f"Manifest rows available: {len(frame)}")
    print(f"Rows selected: {len(selected)}")
    print(f"Output: {output_dir}")
    print(
        "Background geometry: "
        f"side={args.bg_side}, width={args.bg_width_ratio:.3f}×ROI, "
        f"height={args.bg_height_ratio:.3f}×ROI, "
        f"gap={args.bg_gap_ratio:.3f}×ROI width"
    )
    print(
        f"Context margin: {args.context_margin_ratio:.3f}; "
        f"model input: {args.image_size}×{args.image_size} "
        f"({args.resize_mode})"
    )

    records: list[dict[str, Any]] = []

    for index, row in selected.iterrows():
        image_id = str(row["__image_id__"])
        image_path = resolve_image_path(str(row["__image_path__"]))
        annotation_path = annotation_path_for(
            annotation_dir,
            image_path,
            image_id,
        )

        image = Image.open(image_path).convert("RGB")
        roi = load_roi_bbox(
            annotation_path,
            image.size,
            args.roi_label,
        )
        bg, bg_side_used, side_fallback, bg_clipped = choose_bg_bbox(
            roi=roi,
            image_size=image.size,
            preferred_side=args.bg_side,
            width_ratio=args.bg_width_ratio,
            height_ratio=args.bg_height_ratio,
            gap_ratio=args.bg_gap_ratio,
        )

        context_union = union_bbox(roi, bg)
        context = expand_bbox(
            context_union,
            args.context_margin_ratio,
            image.size,
        )
        crop_box = integer_crop_bbox(context, image.size)
        crop = image.crop(crop_box)

        roi_crop = local_bbox(roi, crop_box)
        bg_crop = local_bbox(bg, crop_box)

        model_input, transform = make_model_input(
            crop,
            args.image_size,
            args.resize_mode,
            args.letterbox_fill,
        )
        roi_model = transform_bbox_to_model(roi_crop, transform)
        bg_model = transform_bbox_to_model(bg_crop, transform)

        stem = (
            f"{index + 1:03d}"
            f"__{safe_name(image_id)}"
            f"__{safe_name(row['__phone__'])}"
            f"__{safe_name(row['__light__'])}"
            f"__ph{row['__ph__']:g}"
        )

        crop_path = crop_dir / f"{stem}.png"
        model_path = model_dir / f"{stem}.png"
        overlay_path = overlay_dir / f"{stem}.png"

        crop.save(crop_path)
        model_input.save(model_path)

        overlay = draw_overlay(
            image,
            roi,
            bg,
            context,
            [
                f"{row['__phone__']} / {row['__light__']} / pH {row['__ph__']:g}",
                f"bg side={bg_side_used}; fallback={side_fallback}; clipped={bg_clipped}",
            ],
            args.save_original_size_overlay,
        )
        overlay.save(overlay_path)

        crop_area = float(crop.width * crop.height)
        record = {
            "image_id": image_id,
            "source_image_path": str(image_path),
            "annotation_path": str(annotation_path),
            "phone": str(row["__phone__"]),
            "light": str(row["__light__"]),
            "ph_nominal": float(row["__ph__"]),
            "split": str(row["__split__"]),
            "roi_label": args.roi_label,
            "bg_side_requested": args.bg_side,
            "bg_side_used": bg_side_used,
            "bg_side_fallback": side_fallback,
            "bg_bbox_clipped": bg_clipped,
            "source_width": image.width,
            "source_height": image.height,
            "roi_x1": roi[0],
            "roi_y1": roi[1],
            "roi_x2": roi[2],
            "roi_y2": roi[3],
            "bg_x1": bg[0],
            "bg_y1": bg[1],
            "bg_x2": bg[2],
            "bg_y2": bg[3],
            "crop_x1": crop_box[0],
            "crop_y1": crop_box[1],
            "crop_x2": crop_box[2],
            "crop_y2": crop_box[3],
            "crop_width": crop.width,
            "crop_height": crop.height,
            "crop_touches_source_edge": touches_edge(context, image.size),
            "roi_area_ratio_in_crop": bbox_area(roi_crop) / crop_area,
            "bg_area_ratio_in_crop": bbox_area(bg_crop) / crop_area,
            "roi_crop_x1": roi_crop[0],
            "roi_crop_y1": roi_crop[1],
            "roi_crop_x2": roi_crop[2],
            "roi_crop_y2": roi_crop[3],
            "bg_crop_x1": bg_crop[0],
            "bg_crop_y1": bg_crop[1],
            "bg_crop_x2": bg_crop[2],
            "bg_crop_y2": bg_crop[3],
            "resize_mode": args.resize_mode,
            **transform,
            "roi_model_x1": roi_model[0],
            "roi_model_y1": roi_model[1],
            "roi_model_x2": roi_model[2],
            "roi_model_y2": roi_model[3],
            "bg_model_x1": bg_model[0],
            "bg_model_y1": bg_model[1],
            "bg_model_x2": bg_model[2],
            "bg_model_y2": bg_model[3],
            "crop_path": str(crop_path),
            "model_input_path": str(model_path),
            "overlay_path": str(overlay_path),
        }
        records.append(record)

    records_frame = pd.DataFrame(records)
    records_path = output_dir / "local_context_manifest.csv"
    records_frame.to_csv(
        records_path,
        index=False,
        encoding="utf-8-sig",
    )

    make_contact_sheets(
        records_frame,
        contact_dir,
        audit_ph,
        args.contact_tile_size,
    )

    audit_rows = [
        {
            "metric": "rows_processed",
            "value": len(records_frame),
        },
        {
            "metric": "background_side_fallback_count",
            "value": int(records_frame["bg_side_fallback"].sum()),
        },
        {
            "metric": "background_bbox_clipped_count",
            "value": int(records_frame["bg_bbox_clipped"].sum()),
        },
        {
            "metric": "crop_touches_source_edge_count",
            "value": int(records_frame["crop_touches_source_edge"].sum()),
        },
        {
            "metric": "mean_roi_area_ratio_in_crop",
            "value": float(records_frame["roi_area_ratio_in_crop"].mean()),
        },
        {
            "metric": "mean_bg_area_ratio_in_crop",
            "value": float(records_frame["bg_area_ratio_in_crop"].mean()),
        },
        {
            "metric": "min_crop_width",
            "value": int(records_frame["crop_width"].min()),
        },
        {
            "metric": "min_crop_height",
            "value": int(records_frame["crop_height"].min()),
        },
        {
            "metric": "max_crop_width",
            "value": int(records_frame["crop_width"].max()),
        },
        {
            "metric": "max_crop_height",
            "value": int(records_frame["crop_height"].max()),
        },
    ]
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "manifest": str(manifest_path),
        "annotation_dir": str(annotation_dir),
        "split_file": str(split_path) if split_path else None,
        "column_map": column_map,
        "selection": args.selection,
        "audit_ph": audit_ph,
        "rows_available": len(frame),
        "rows_processed": len(records_frame),
        "geometry": {
            "roi_label": args.roi_label,
            "bg_side": args.bg_side,
            "bg_width_ratio": args.bg_width_ratio,
            "bg_height_ratio": args.bg_height_ratio,
            "bg_gap_ratio": args.bg_gap_ratio,
            "context_margin_ratio": args.context_margin_ratio,
        },
        "model_input": {
            "image_size": args.image_size,
            "resize_mode": args.resize_mode,
            "letterbox_fill": args.letterbox_fill,
        },
        "important": (
            "The source nuisance-preview script contains no separate spatial "
            "background rectangle. This script defines an adjacent local "
            "background rectangle explicitly. Contact sheets must be checked "
            "for handwritten labels or other leakage."
        ),
    }
    with (output_dir / "local_context_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("\nCompleted.")
    print(f"Local-context manifest: {records_path}")
    print(f"Contact sheets: {contact_dir}")
    print(
        "Background side fallbacks:",
        int(records_frame["bg_side_fallback"].sum()),
    )
    print(
        "Clipped background boxes:",
        int(records_frame["bg_bbox_clipped"].sum()),
    )
    print(
        "Crops touching source edge:",
        int(records_frame["crop_touches_source_edge"].sum()),
    )
    print(
        "IMPORTANT: visually inspect contact sheets before CNN training."
    )


if __name__ == "__main__":
    main()


"""
python .\scripts\09_make_local_context_inputs.py `
  --manifest ".\data\processed\capture_manifest.csv" `
  --annotation-dir ".\data\raw\captures\annotations" `
  --output-dir ".\outputs\local_context_audit" `
  --roi-label "roi" `
  --bg-side "left" `
  --bg-width-ratio 1.0 `
  --bg-height-ratio 1.0 `
  --bg-gap-ratio 0.25 `
  --context-margin-ratio 0.15 `
  --image-size 224 `
  --resize-mode "letterbox" `
  --letterbox-fill "median" `
  --selection "audit_grid" `
  --audit-ph "2,6,8,9,12" `
  --seed 42
"""
"""
python .\scripts\09_make_local_context_inputs.py `
  --manifest ".\data\processed\capture_manifest.csv" `
  --annotation-dir ".\data\raw\captures\annotations" `
  --output-dir ".\outputs\local_context_audit_whitepad" `
  --roi-label "roi" `
  --bg-side "left" `
  --bg-width-ratio 1.0 `
  --bg-height-ratio 1.0 `
  --bg-gap-ratio 0.25 `
  --context-margin-ratio 0.15 `
  --image-size 224 `
  --resize-mode "letterbox" `
  --letterbox-fill "white" `
  --selection "audit_grid" `
  --audit-ph "2,6,8,9,12" `
  --seed 42
"""