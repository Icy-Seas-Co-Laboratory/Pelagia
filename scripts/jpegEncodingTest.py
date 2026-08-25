#!/usr/bin/env python3

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_QUALITIES = [50, 60, 70, 80, 85, 90, 92, 95, 97, 98, 99, 100]


def require_command(command: str):
    if shutil.which(command) is None:
        raise RuntimeError(
            f"Required command '{command}' was not found in PATH."
        )


def run(cmd):
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def extract_frame(avi_path: Path, frame_number: int, output_png: Path):
    """
    Extract exactly one frame from the AVI.

    frame_number is zero-based.
    """
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(avi_path),
        "-vf", f"select=eq(n\\,{frame_number})",
        "-vsync", "0",
        "-frames:v", "1",
        str(output_png),
    ])

    if not output_png.exists():
        raise RuntimeError(
            f"Could not extract frame {frame_number}. "
            "Check that the frame number exists in the AVI."
        )


def make_pgm(reference_png: Path, reference_pgm: Path):
    """
    Create an 8-bit grayscale reference image suitable for cjpeg.

    For grayscale/paletted shadowgraph imagery, this gives cjpeg a simple,
    unambiguous 8-bit grayscale source.
    """
    image = Image.open(reference_png)

    if image.mode != "L":
        image = image.convert("L")

    image.save(reference_pgm)


def encode_jpeg(reference_pgm: Path, output_jpg: Path, quality: int):
    """
    Encode using libjpeg-turbo's cjpeg.
    """
    with output_jpg.open("wb") as f:
        subprocess.run([
            "cjpeg",
            "-quality", str(quality),
            "-optimize",
            "-grayscale",
            str(reference_pgm),
        ], stdout=f, check=True)


def decode_jpeg(input_jpg: Path, output_pgm: Path):
    """
    Decode using libjpeg-turbo's djpeg.
    """
    with output_pgm.open("wb") as f:
        subprocess.run([
            "djpeg",
            "-grayscale",
            "-pnm",
            str(input_jpg),
        ], stdout=f, check=True)


def load_gray(path: Path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def calculate_metrics(original, decoded):
    original_f = original.astype(np.float32)
    decoded_f = decoded.astype(np.float32)

    error = decoded_f - original_f
    abs_error = np.abs(error)

    mae = float(np.mean(abs_error))
    mse = float(np.mean(error ** 2))
    rmse = math.sqrt(mse)
    max_error = float(np.max(abs_error))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "psnr": psnr,
        "abs_error": abs_error,
    }


def save_difference(abs_error, output_path: Path, scale=10):
    """
    Save scale * |original - decoded|.

    Clipped to [0, 255] for visualization.
    """
    difference = np.clip(abs_error * scale, 0, 255).astype(np.uint8)
    Image.fromarray(difference, mode="L").save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare libjpeg-turbo JPEG encodings of a selected AVI frame."
        )
    )

    parser.add_argument(
        "avi",
        type=Path,
        help="Input AVI file",
    )

    parser.add_argument(
        "frame",
        type=int,
        help="Zero-based frame number",
    )

    parser.add_argument(
        "-q",
        "--qualities",
        nargs="+",
        type=int,
        default=DEFAULT_QUALITIES,
        help=(
            "JPEG quality settings. "
            f"Default: {' '.join(map(str, DEFAULT_QUALITIES))}"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    require_command("ffmpeg")
    require_command("cjpeg")
    require_command("djpeg")

    avi_path = args.avi.resolve()

    if not avi_path.exists():
        raise FileNotFoundError(avi_path)

    if args.output is None:
        output_dir = Path(
            f"{avi_path.stem}_frame_{args.frame:06d}_jpeg_test"
        )
    else:
        output_dir = args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    reference_png = output_dir / "reference.png"
    reference_pgm = output_dir / "reference.pgm"

    print(f"\nExtracting frame {args.frame}...")
    extract_frame(
        avi_path,
        args.frame,
        reference_png,
    )

    make_pgm(reference_png, reference_pgm)

    original = load_gray(reference_pgm)

    raw_size = original.nbytes

    results = []

    for quality in args.qualities:
        print(f"\nJPEG quality {quality}")

        jpeg_path = output_dir / f"q{quality:03d}.jpg"
        decoded_path = output_dir / f"q{quality:03d}_decoded.pgm"
        difference_path = output_dir / f"q{quality:03d}_diff_x10.png"

        encode_jpeg(
            reference_pgm,
            jpeg_path,
            quality,
        )

        decode_jpeg(
            jpeg_path,
            decoded_path,
        )

        decoded = load_gray(decoded_path)

        if original.shape != decoded.shape:
            raise RuntimeError(
                f"Shape mismatch: original={original.shape}, "
                f"decoded={decoded.shape}"
            )

        metrics = calculate_metrics(original, decoded)

        save_difference(
            metrics["abs_error"],
            difference_path,
            scale=10,
        )

        jpeg_size = jpeg_path.stat().st_size
        compression_ratio = raw_size / jpeg_size

        result = {
            "quality": quality,
            "jpeg_bytes": jpeg_size,
            "bytes_per_pixel": jpeg_size / original.size,
            "compression_ratio": compression_ratio,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "max_error": metrics["max_error"],
            "psnr_db": metrics["psnr"],
        }

        results.append(result)

    csv_path = output_dir / "results.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nResults")
    print(
        f"{'Q':>4} "
        f"{'KiB':>10} "
        f"{'B/pixel':>10} "
        f"{'Ratio':>10} "
        f"{'MAE':>10} "
        f"{'RMSE':>10} "
        f"{'Max':>8} "
        f"{'PSNR':>10}"
    )

    for r in results:
        psnr = (
            "inf"
            if math.isinf(r["psnr_db"])
            else f"{r['psnr_db']:.2f}"
        )

        print(
            f"{r['quality']:4d} "
            f"{r['jpeg_bytes'] / 1024:10.1f} "
            f"{r['bytes_per_pixel']:10.4f} "
            f"{r['compression_ratio']:10.2f} "
            f"{r['mae']:10.4f} "
            f"{r['rmse']:10.4f} "
            f"{r['max_error']:8.0f} "
            f"{psnr:>10}"
        )

    print(f"\nOutput: {output_dir}")
    print(f"Metrics: {csv_path}")


if __name__ == "__main__":
    main()