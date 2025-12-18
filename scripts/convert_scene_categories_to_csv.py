"""Convert sceneCategories.txt to labels.csv format for ImgSegFolder.

Usage:
    uv run scripts/convert_scene_categories_to_csv.py /path/to/dataset

This converts:
    stem1 label1
    stem2 label2

To:
    stem,scene
    stem1,label1
    stem2,label2
"""

import argparse
import csv
import pathlib


def main():
    parser = argparse.ArgumentParser(
        description="Convert sceneCategories.txt to labels.csv"
    )
    parser.add_argument("root", type=pathlib.Path, help="Dataset root directory")
    parser.add_argument("--input", default="sceneCategories.txt", help="Input filename")
    parser.add_argument("--output", default="labels.csv", help="Output filename")
    parser.add_argument(
        "--label-col", default="scene", help="Name for the label column"
    )
    args = parser.parse_args()

    input_fpath = args.root / args.input
    output_fpath = args.root / args.output

    assert input_fpath.exists(), f"Input file not found: {input_fpath}"

    rows = []
    with open(input_fpath) as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            stem, _, label = line.rpartition(" ")
            rows.append({"stem": stem, args.label_col: label})

    with open(output_fpath, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=["stem", args.label_col])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(rows)} entries from {input_fpath} to {output_fpath}")


if __name__ == "__main__":
    main()
