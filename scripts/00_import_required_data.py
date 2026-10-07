#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIXED_ANNOTATION_EXTENSIONS = {".json"}
SEPARATE_ANNOTATION_EXTENSIONS = {".json", ".npy", ".npz", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the required direct-capture data into the new project. "
            "Use --collected-dir when images, LabelMe JSON files, and the feature CSV "
            "are stored together in one directory."
        )
    )
    parser.add_argument(
        "--collected-dir",
        default="",
        help="Directory containing mixed capture images, LabelMe JSON files, and feature CSV.",
    )
    parser.add_argument(
        "--features",
        default="",
        help=(
            "Path to captures_final_features.csv. Optional in --collected-dir mode; "
            "use it when more than one matching CSV exists."
        ),
    )
    parser.add_argument(
        "--images",
        default="",
        help="Separate directory containing original capture images.",
    )
    parser.add_argument(
        "--annotations",
        default="",
        help="Separate directory containing LabelMe JSON files or ROI masks.",
    )
    parser.add_argument(
        "--external-qub",
        default="",
        help="Optional QUB three_smartphones directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing_file(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def resolve_existing_dir(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} directory not found: {path}")
    return path


def find_feature_csv(collected_dir: Path) -> Path:
    exact = collected_dir / "captures_final_features.csv"
    if exact.is_file():
        return exact.resolve()

    candidates = sorted(
        path.resolve()
        for path in collected_dir.glob("captures_final_features*.csv")
        if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No captures_final_features*.csv file was found in "
            f"{collected_dir}. Specify it explicitly with --features."
        )

    candidate_text = "\n  - ".join(str(path) for path in candidates)
    raise ValueError(
        "Multiple feature CSV files were found. Specify the intended file with --features:\n"
        f"  - {candidate_text}"
    )


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {dst}. Use --overwrite to replace it."
        )
    shutil.copy2(src, dst)


def iter_files_with_extensions(root: Path, extensions: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def copy_tree_filtered(
    src: Path,
    dst: Path,
    allowed_extensions: set[str],
    overwrite: bool,
) -> list[Path]:
    copied: list[Path] = []
    for path in iter_files_with_extensions(src, allowed_extensions):
        relative = path.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Destination already exists: {target}. Use --overwrite to replace it."
            )
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def determine_sources(args: argparse.Namespace) -> tuple[str, Path, Path, Path | None]:
    if args.collected_dir:
        collected_dir = resolve_existing_dir(args.collected_dir, "Collected data")
        features = (
            resolve_existing_file(args.features, "Feature CSV")
            if args.features
            else find_feature_csv(collected_dir)
        )
        if args.images or args.annotations:
            raise ValueError(
                "Do not combine --collected-dir with --images or --annotations. "
                "Use --features only when selecting among multiple CSV files."
            )
        return "mixed_collected_directory", features, collected_dir, collected_dir

    if not args.features or not args.images:
        raise ValueError(
            "Use either --collected-dir, or provide both --features and --images."
        )

    features = resolve_existing_file(args.features, "Feature CSV")
    images = resolve_existing_dir(args.images, "Image")
    annotations = (
        resolve_existing_dir(args.annotations, "Annotation")
        if args.annotations
        else None
    )
    return "separate_paths", features, images, annotations


def main() -> None:
    args = parse_args()
    mode, features, images_source, annotations_source = determine_sources(args)
    qub_source = (
        resolve_existing_dir(args.external_qub, "External QUB")
        if args.external_qub
        else None
    )

    feature_dst = PROJECT_ROOT / "data" / "raw" / "captures" / "captures_final_features.csv"
    image_dst = PROJECT_ROOT / "data" / "raw" / "captures" / "images"
    annotation_dst = PROJECT_ROOT / "data" / "raw" / "captures" / "annotations"
    qub_dst = PROJECT_ROOT / "data" / "raw" / "external_qub" / "three_smartphones"

    copy_file(features, feature_dst, args.overwrite)
    copied_images = copy_tree_filtered(
        images_source,
        image_dst,
        IMAGE_EXTENSIONS,
        args.overwrite,
    )

    copied_annotations: list[Path] = []
    if annotations_source is not None:
        annotation_extensions = (
            MIXED_ANNOTATION_EXTENSIONS
            if mode == "mixed_collected_directory"
            else SEPARATE_ANNOTATION_EXTENSIONS
        )
        copied_annotations = copy_tree_filtered(
            annotations_source,
            annotation_dst,
            annotation_extensions,
            args.overwrite,
        )

    copied_qub: list[Path] = []
    if qub_source is not None:
        copied_qub = copy_tree_filtered(
            qub_source,
            qub_dst,
            IMAGE_EXTENSIONS | {".csv", ".json", ".npy", ".npz"},
            args.overwrite,
        )

    inventory = {
        "import_mode": mode,
        "features": {
            "source": str(features),
            "destination": str(feature_dst),
            "sha256": sha256(feature_dst),
        },
        "images": {
            "source": str(images_source),
            "destination": str(image_dst),
            "count": len(copied_images),
        },
        "annotations": {
            "source": str(annotations_source) if annotations_source else "",
            "destination": str(annotation_dst),
            "count": len(copied_annotations),
        },
        "external_qub": {
            "source": str(qub_source) if qub_source else "",
            "destination": str(qub_dst),
            "count": len(copied_qub),
        },
    }
    inventory_path = PROJECT_ROOT / "data" / "raw" / "data_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8") as file_obj:
        json.dump(inventory, file_obj, ensure_ascii=False, indent=2)

    print("Import mode:", mode)
    print("Imported feature CSV:", feature_dst)
    print("Imported capture images:", len(copied_images))
    print("Imported annotations:", len(copied_annotations))
    print("Imported optional QUB files:", len(copied_qub))
    print("Inventory:", inventory_path)

    if len(copied_images) != 195:
        print(f"WARNING: expected 195 direct-capture images, copied {len(copied_images)}")
    if copied_annotations and len(copied_annotations) != len(copied_images):
        print(
            "WARNING: image and annotation counts differ: "
            f"images={len(copied_images)}, annotations={len(copied_annotations)}"
        )


if __name__ == "__main__":
    main()

"""
python .\scripts\00_import_required_data.py `
  --collected-dir "C:\.Soyeon\DS\학부연구생\DIC\data\collected"
"""