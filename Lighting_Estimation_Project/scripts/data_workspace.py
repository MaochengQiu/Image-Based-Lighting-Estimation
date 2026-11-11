"""Manage dataset-independent storage and source inventories (standard library only)."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("raw", "manifests", "processed", "tokens")
EXTENSIONS = {".hdr", ".exr", ".npy"}


def dataset_name(value):
    if not value or value in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise argparse.ArgumentTypeError("Use letters, digits, underscores or hyphens for dataset names")
    return value


def initialize(name=None):
    for layer in LAYERS:
        path = ROOT / "data" / layer
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch(exist_ok=True)
        if name:
            (path / name).mkdir(exist_ok=True)
    print(f"Data root: {ROOT / 'data'}")
    if name:
        print(f"Place original files in: {ROOT / 'data/raw' / name}")


def inventory(name):
    raw = ROOT / "data/raw" / name
    if not raw.is_dir():
        raise ValueError(f"Missing raw directory: {raw}; run init --dataset {name}")
    paths = sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No HDR/EXR/NPY files found in {raw}; no inventory was written")
    output = ROOT / "data/manifests" / name / "sources.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves manually assigned scene IDs and official splits.
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_id", "pano_path", "split"])
        writer.writeheader()
        for path in paths:
            relative = Path("..") / ".." / "raw" / name / path.relative_to(raw)
            writer.writerow({"scene_id": "", "pano_path": relative.as_posix(), "split": ""})
    print(f"Indexed {len(paths)} candidate files: {output}")
    print("This is an inventory, not validation of HDR format or radiometric calibration.")
    print("Fill scene_id from room/scene metadata. Prefer official train/val/test splits.")
    print("Files from the same room must share scene_id. Do not assign file-level IDs blindly.")


def check_sources(path):
    path = path.resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"scene_id", "pano_path"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain scene_id,pano_path[,split]")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty source inventory")
    groups, seen, provided = {}, set(), []
    for number, row in enumerate(rows, 2):
        scene = (row.get("scene_id") or "").strip()
        pano = (row.get("pano_path") or "").strip()
        split = (row.get("split") or "").strip()
        if not scene or not pano:
            raise ValueError(f"Line {number}: fill scene_id and pano_path before preparation")
        source = (path.parent / pano).resolve()
        if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Line {number}: missing or unsupported source: {source}")
        identity = str(source).casefold()
        if identity in seen:
            raise ValueError(f"Line {number}: duplicate source path: {source}")
        seen.add(identity)
        provided.append(bool(split))
        if split and split not in {"train", "val", "test"}:
            raise ValueError(f"Line {number}: split must be train, val or test")
        if scene in groups and groups[scene] != split:
            raise ValueError(f"Line {number}: inconsistent split for scene {scene}")
        groups[scene] = split
    if any(provided) and not all(provided):
        raise ValueError("Either specify a split on every row or leave every split empty")
    if all(provided) and set(groups.values()) != {"train", "val", "test"}:
        raise ValueError("Explicit splits must include train, val and test")
    if len(groups) < 3:
        raise ValueError("At least three independent scene groups are needed")
    print(json.dumps({"sources": len(rows), "scene_groups": len(groups),
                      "split_policy": "explicit" if all(provided) else "seeded_scene_split",
                      "checked": "paths and grouping only; HDR decoding happens during prepare"},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--dataset", type=dataset_name)
    index = sub.add_parser("index")
    index.add_argument("--dataset", type=dataset_name, required=True)
    check = sub.add_parser("check")
    check.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args.dataset)
        elif args.command == "index":
            inventory(args.dataset)
        else:
            check_sources(args.sources)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
