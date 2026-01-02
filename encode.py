import os
import base64
import csv
from PIL import Image
import io
import numpy as np
import argparse

def png_file_to_base64(path: str) -> str:
    # Lossless PNG -> base64 string
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # H,W,C
    raw = arr.tobytes(order="C")
    # if compress:
    # raw = zlib.compress(raw, level=9)
    return base64.b64encode(raw).decode("ascii")

def make_csv_from_dir(
    img_dir: str,
    out_csv_path: str,
    start_num: int = 69000,
    end_num: int = 69999,
):
    """
    Expects filenames that include the integer index (e.g., 69000.png, 69001.png, ...).
    Creates rows with:
      - id: 0..(end_num-start_num)
      - data: base64(PNG bytes)
      - Usage: "Public"
    """
    # Build a map from number -> filepath by scanning directory once
    num_to_path = {}
    for fn in os.listdir(img_dir):
        root, ext = os.path.splitext(fn)
        if ext.lower() not in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
            continue
        try:
            n = int(root)  # assumes filename is like "69000.png"
        except ValueError:
            continue
        num_to_path[n] = os.path.join(img_dir, fn)

    expected = list(range(start_num, end_num + 1))
    missing = [n for n in expected if n not in num_to_path]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} files for indices like {missing[:10]} "
            f"(showing up to 10). Ensure filenames are like '69000.png'."
        )

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "data"])
        writer.writeheader()

        for idx, n in enumerate(expected):  # idx: 0..1000
            path = num_to_path[n]
            b64 = png_file_to_base64(path)
            writer.writerow({"id": idx, "data": b64})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_dir',
        type=str,
        required=True,
        help="Directory containing images named like 69000.png, 69001.png, ..."
    )
    parser.add_argument(
        '--sub_filename',
        type=str,
        default='submission.csv',
        help="Output CSV filename."
    )
    args = parser.parse_args()
    make_csv_from_dir(
        img_dir=args.img_dir,     # <- your directory
        out_csv_path=args.sub_filename,
        start_num=69000,
        end_num=69999,
    )
