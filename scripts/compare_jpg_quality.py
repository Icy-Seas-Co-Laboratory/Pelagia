#!/usr/bin/env python3
"""Compare JPEG and JPEG XL quality settings for one input image.

The script writes recompressed images, per-quality difference heatmaps, a contact
sheet, and CSV/JSON metrics so image-size and artifact tradeoffs can be checked
quickly. JPEG XL support uses the external ``cjxl`` and ``djxl`` commands when
available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_QUALITIES = [100, 98, 96, 94, 92, 90, 85, 80, 75, 70, 60, 50, 40, 30]
CODEC_ALIASES = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "jxl": "jxl",
    "jpegxl": "jxl",
    "jpeg_xl": "jxl",
    "jpeg-xl": "jxl",
}


def parse_quality_values(values: list[str] | None) -> list[int]:
    if not values:
        return DEFAULT_QUALITIES
    qualities: list[int] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            quality = int(item)
            if quality < 1 or quality > 100:
                raise argparse.ArgumentTypeError("JPEG quality values must be between 1 and 100.")
            qualities.append(quality)
    return sorted(dict.fromkeys(qualities), reverse=True)


def parse_codecs(values: list[str] | None) -> list[str]:
    if not values:
        codecs = ["jpeg"]
        if shutil.which("cjxl") and shutil.which("djxl"):
            codecs.append("jxl")
        return codecs
    codecs: list[str] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item in {"all", "both"}:
                codecs.extend(["jpeg", "jxl"])
                continue
            if item == "auto":
                codecs.extend(parse_codecs(None))
                continue
            if item not in CODEC_ALIASES:
                raise argparse.ArgumentTypeError("Codec values must be jpeg, jxl, all, or auto.")
            codecs.append(CODEC_ALIASES[item])
    return list(dict.fromkeys(codecs))


def prepare_codec_image(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    result = image
    if result.ndim == 3 and result.shape[2] == 4:
        result = result[:, :, :3]
        warnings.append("Dropped alpha channel before JPEG/JPEG XL encoding.")
    if result.dtype != np.uint8:
        original_dtype = str(result.dtype)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        warnings.append(f"Normalized {original_dtype} input to uint8 for encoding.")
    return np.ascontiguousarray(result), warnings


def encode_decode_jpeg(image: np.ndarray, quality: int) -> tuple[bytes, np.ndarray]:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode JPEG at quality {quality}.")
    payload = encoded.tobytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError(f"OpenCV failed to decode JPEG at quality {quality}.")
    return payload, decoded


def encode_decode_jxl(image: np.ndarray, quality: int) -> tuple[bytes, np.ndarray]:
    cjxl = shutil.which("cjxl")
    djxl = shutil.which("djxl")
    if not cjxl or not djxl:
        raise RuntimeError("JPEG XL comparison requires both 'cjxl' and 'djxl' on PATH.")
    with tempfile.TemporaryDirectory(prefix="pelagia-jxl-") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / "input.png"
        encoded_path = temp / "encoded.jxl"
        decoded_path = temp / "decoded.png"
        if not cv2.imwrite(str(input_path), image):
            raise RuntimeError("OpenCV failed to write temporary PNG for JPEG XL encoding.")
        subprocess.run(
            [cjxl, str(input_path), str(encoded_path), "--quality", str(int(quality)), "--quiet"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [djxl, str(encoded_path), str(decoded_path), "--quiet"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        payload = encoded_path.read_bytes()
        decoded = cv2.imread(str(decoded_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError(f"OpenCV failed to read decoded JPEG XL image at quality {quality}.")
        return payload, decoded


def encode_decode(image: np.ndarray, *, codec: str, quality: int) -> tuple[bytes, np.ndarray, str]:
    if codec == "jpeg":
        payload, decoded = encode_decode_jpeg(image, quality)
        return payload, decoded, ".jpg"
    if codec == "jxl":
        payload, decoded = encode_decode_jxl(image, quality)
        return payload, decoded, ".jxl"
    raise ValueError(f"Unsupported codec {codec!r}.")


def as_compare_pair(original: np.ndarray, decoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if original.ndim == decoded.ndim and original.shape == decoded.shape:
        return original, decoded
    if original.ndim == 2 and decoded.ndim == 3:
        return cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), decoded
    if original.ndim == 3 and decoded.ndim == 2:
        return original, cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Decoded image shape {decoded.shape} does not match input shape {original.shape}.")


def image_metrics(original: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    a, b = as_compare_pair(original, decoded)
    diff = a.astype(np.float32) - b.astype(np.float32)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return {
        "mse": mse,
        "mae": float(np.mean(abs_diff)),
        "max_abs_error": float(np.max(abs_diff)),
        "psnr_db": psnr,
        "global_ssim": global_ssim(a, b),
    }


def global_ssim(original: np.ndarray, decoded: np.ndarray) -> float:
    """Return a simple whole-image SSIM-style score without extra dependencies."""
    a = original.astype(np.float64)
    b = decoded.astype(np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
        b = b[:, :, None]
    scores: list[float] = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    for channel in range(a.shape[2]):
        x = a[:, :, channel]
        y = b[:, :, channel]
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        sigma_x = float(np.var(x))
        sigma_y = float(np.var(y))
        sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        scores.append(1.0 if denominator == 0 else numerator / denominator)
    return float(np.mean(scores))


def diff_heatmap(original: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    a, b = as_compare_pair(original, decoded)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    if diff.ndim == 3:
        diff = np.max(diff, axis=2)
    if int(diff.max()) == 0:
        scaled = diff.astype(np.uint8)
    else:
        scaled = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)


def preview_image(image: np.ndarray, width: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def labeled_tile(image: np.ndarray, label: str, width: int) -> np.ndarray:
    tile = preview_image(image, width)
    label_height = 28
    canvas = np.full((tile.shape[0] + label_height, tile.shape[1], 3), 255, dtype=np.uint8)
    canvas[label_height:, :, :] = tile
    cv2.putText(
        canvas,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_contact_sheet(original: np.ndarray, rows: list[dict[str, Any]], preview_width: int) -> np.ndarray:
    original_tile = labeled_tile(original, "original", preview_width)
    sheet_rows: list[np.ndarray] = []
    spacer = np.full((original_tile.shape[0], 8, 3), 255, dtype=np.uint8)
    for row in rows:
        jpeg_tile = labeled_tile(
            row["decoded"],
            f"{row['codec']} q={row['quality']} {row['size_kib']:.1f} KiB PSNR={row['psnr_db_label']}",
            preview_width,
        )
        diff_tile = labeled_tile(row["diff"], "absolute diff heatmap", preview_width)
        sheet_rows.append(np.hstack([original_tile, spacer, jpeg_tile, spacer, diff_tile]))
    row_spacer = np.full((10, sheet_rows[0].shape[1], 3), 255, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for index, row in enumerate(sheet_rows):
        if index:
            stacked.append(row_spacer)
        stacked.append(row)
    return np.vstack(stacked)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "codec",
        "quality",
        "size_bytes",
        "size_kib",
        "input_size_ratio",
        "mse",
        "mae",
        "max_abs_error",
        "psnr_db",
        "global_ssim",
        "encoded_path",
        "diff_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def print_table(rows: list[dict[str, Any]]) -> None:
    print("codec  quality  size_kib  ratio   psnr_db  ssim      mae    max_abs")
    for row in rows:
        psnr = row["psnr_db_label"]
        print(
            f"{row['codec']:<5}  "
            f"{row['quality']:>7}  "
            f"{row['size_kib']:>8.1f}  "
            f"{row['input_size_ratio']:>5.2f}  "
            f"{psnr:>7}  "
            f"{row['global_ssim']:>8.5f}  "
            f"{row['mae']:>5.2f}  "
            f"{row['max_abs_error']:>7.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image", type=Path, help="Image to recompress.")
    parser.add_argument(
        "-q",
        "--quality",
        action="append",
        help=(
            "Quality value or comma-separated list. JPEG uses OpenCV quality; "
            "JPEG XL uses cjxl --quality. Default: common 30-100 sweep."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for encoded images, diff images, metrics, and contact sheet.",
    )
    parser.add_argument(
        "--codec",
        action="append",
        help=(
            "Codec to compare: jpeg, jxl, all, or auto. Default: jpeg plus jxl "
            "when cjxl/djxl are available."
        ),
    )
    parser.add_argument("--preview-width", type=int, default=360, help="Tile width for the contact sheet.")
    args = parser.parse_args()

    input_path = args.input_image.expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input image was not found: {input_path}")
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"OpenCV could not read image: {input_path}")
    source, warnings = prepare_codec_image(image)
    qualities = parse_quality_values(args.quality)
    codecs = parse_codecs(args.codec)
    if "jxl" in codecs and (not shutil.which("cjxl") or not shutil.which("djxl")):
        parser.error("JPEG XL comparison requires both 'cjxl' and 'djxl' on PATH.")
    output_dir = (args.output_dir or input_path.with_name(f"{input_path.stem}_codec_quality")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_size = max(1, input_path.stat().st_size)

    rows: list[dict[str, Any]] = []
    for codec in codecs:
        for quality in qualities:
            payload, decoded, suffix = encode_decode(source, codec=codec, quality=quality)
            metrics = image_metrics(source, decoded)
            encoded_path = output_dir / f"{input_path.stem}_{codec}_q{quality:03d}{suffix}"
            diff_path = output_dir / f"{input_path.stem}_{codec}_q{quality:03d}_diff.png"
            encoded_path.write_bytes(payload)
            diff = diff_heatmap(source, decoded)
            ok = cv2.imwrite(str(diff_path), diff)
            if not ok:
                raise RuntimeError(f"OpenCV failed to write diff image: {diff_path}")
            size_bytes = len(payload)
            psnr_label = "inf" if math.isinf(metrics["psnr_db"]) else f"{metrics['psnr_db']:.2f}"
            rows.append(
                {
                    "codec": codec,
                    "quality": quality,
                    "size_bytes": size_bytes,
                    "size_kib": size_bytes / 1024.0,
                    "input_size_ratio": size_bytes / original_size,
                    "encoded_path": str(encoded_path),
                    "diff_path": str(diff_path),
                    "decoded": decoded,
                    "diff": diff,
                    "psnr_db_label": psnr_label,
                    **metrics,
                }
            )

    sheet = make_contact_sheet(source, rows, args.preview_width)
    sheet_path = output_dir / f"{input_path.stem}_codec_quality_sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"OpenCV failed to write contact sheet: {sheet_path}")

    report_rows = [
        {key: value for key, value in row.items() if key not in {"decoded", "diff", "psnr_db_label"}}
        for row in rows
    ]
    write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "input_image": str(input_path),
                "input_size_bytes": original_size,
                "output_dir": str(output_dir),
                "contact_sheet": str(sheet_path),
                "warnings": warnings,
                "codecs": codecs,
                "qualities": report_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for warning in warnings:
        print(f"warning: {warning}")
    print_table(rows)
    print(f"\nWrote comparison outputs to: {output_dir}")
    print(f"Contact sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
