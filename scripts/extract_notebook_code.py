import argparse
import json
import sys
from pathlib import Path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--max-cell-chars", type=int, default=4000)
    args = parser.parse_args()

    for pattern in args.paths:
        for path in sorted(Path().glob(pattern)):
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"### {path}")
            for idx, cell in enumerate(data.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                print(f"\n-- cell {idx} len={len(source)}")
                print(source[: args.max_cell_chars])


if __name__ == "__main__":
    main()
