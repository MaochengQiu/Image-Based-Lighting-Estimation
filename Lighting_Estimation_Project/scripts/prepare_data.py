"""Prepare paired LDR/HDR data without importing or installing PyTorch."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_workspace import check_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/initial.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    check_sources(args.sources)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        from src.lighting.data import prepare
    except ModuleNotFoundError as error:
        parser.exit(1, f"Missing data dependency: {error.name}; install numpy and opencv-python.\n")
    count = prepare(args.sources, args.out_dir, config)
    print(f"Prepared {count} paired samples in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
