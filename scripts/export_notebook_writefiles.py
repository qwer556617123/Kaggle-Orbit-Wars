import argparse
import json
from pathlib import Path


def safe_name(path):
    return path.parent.name + "__" + path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--out", default="references/kaggle_notebooks/_extracted")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for nb_path in sorted(Path().glob(args.pattern)):
        data = json.loads(nb_path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.startswith("%%writefile"):
                continue
            lines = source.splitlines()
            body = "\n".join(lines[1:]) + "\n"
            target = out_dir / f"{safe_name(nb_path)}_cell{idx}.py"
            target.write_text(body, encoding="utf-8")
            print(target)


if __name__ == "__main__":
    main()
