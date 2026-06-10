from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 0/255 binary mask."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Mask augmentation expects a 2D mask.")
    return np.ascontiguousarray((array > 0).astype(np.uint8) * 255)


def dilate_binary_mask(
    mask: np.ndarray,
    *,
    kernel_size: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Dilate foreground pixels in a binary mask."""
    binary = as_binary_mask(mask)
    kernel = np.ones(_kernel_size(kernel_size), np.uint8)
    return np.ascontiguousarray(cv2.dilate(binary, kernel, iterations=max(1, int(iterations))))


def erode_binary_mask(
    mask: np.ndarray,
    *,
    kernel_size: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Erode foreground pixels in a binary mask."""
    binary = as_binary_mask(mask)
    kernel = np.ones(_kernel_size(kernel_size), np.uint8)
    return np.ascontiguousarray(cv2.erode(binary, kernel, iterations=max(1, int(iterations))))


def open_binary_mask(
    mask: np.ndarray,
    *,
    kernel_size: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Remove small foreground specks with morphological opening."""
    binary = as_binary_mask(mask)
    kernel = np.ones(_kernel_size(kernel_size), np.uint8)
    return np.ascontiguousarray(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=max(1, int(iterations)))
    )


def close_binary_mask(
    mask: np.ndarray,
    *,
    kernel_size: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Close small holes and gaps with morphological closing."""
    binary = as_binary_mask(mask)
    kernel = np.ones(_kernel_size(kernel_size), np.uint8)
    return np.ascontiguousarray(
        cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iterations)))
    )


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background holes enclosed by foreground regions."""
    binary = as_binary_mask(mask)
    if binary.size == 0:
        return binary
    flood = binary.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return np.ascontiguousarray(binary | holes)


def remove_small_mask_components(mask: np.ndarray, *, min_area: int | float = 1) -> np.ndarray:
    """Drop connected foreground components smaller than min_area pixels."""
    binary = as_binary_mask(mask)
    resolved_min_area = float(min_area)
    if resolved_min_area <= 1:
        return binary
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    kept = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, num):
        if float(stats[label, cv2.CC_STAT_AREA]) >= resolved_min_area:
            kept[labels == label] = 255
    return np.ascontiguousarray(kept)


def clear_border_components(mask: np.ndarray) -> np.ndarray:
    """Remove foreground components touching any image border."""
    binary = as_binary_mask(mask)
    if binary.size == 0:
        return binary
    num, labels, _, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    border_labels.discard(0)
    if not border_labels:
        return binary
    kept = binary.copy()
    for label in border_labels:
        kept[labels == label] = 0
    return np.ascontiguousarray(kept)


def augment_mask(
    mask: np.ndarray,
    *,
    enabled: bool = True,
    steps: Sequence[str] | None = None,
    dilate_kernel_size: tuple[int, int] = (3, 3),
    dilate_iterations: int = 1,
    erode_kernel_size: tuple[int, int] = (3, 3),
    erode_iterations: int = 1,
    open_kernel_size: tuple[int, int] = (3, 3),
    open_iterations: int = 1,
    close_kernel_size: tuple[int, int] = (3, 3),
    close_iterations: int = 1,
    fill_holes: bool = False,
    remove_small_components: bool = False,
    min_component_area: int | float = 1,
    clear_border: bool = False,
) -> np.ndarray:
    """Apply a configured sequence of mask cleanup and morphology steps."""
    result = as_binary_mask(mask)
    if not enabled:
        return result

    normalized_steps = [str(step).strip().lower() for step in (steps or []) if str(step).strip()]
    if fill_holes and "fill_holes" not in normalized_steps:
        normalized_steps.append("fill_holes")
    if remove_small_components and "remove_small_components" not in normalized_steps:
        normalized_steps.append("remove_small_components")
    if clear_border and "clear_border" not in normalized_steps:
        normalized_steps.append("clear_border")

    for step in normalized_steps:
        if step == "dilate":
            result = dilate_binary_mask(
                result,
                kernel_size=dilate_kernel_size,
                iterations=dilate_iterations,
            )
        elif step == "erode":
            result = erode_binary_mask(
                result,
                kernel_size=erode_kernel_size,
                iterations=erode_iterations,
            )
        elif step == "open":
            result = open_binary_mask(
                result,
                kernel_size=open_kernel_size,
                iterations=open_iterations,
            )
        elif step == "close":
            result = close_binary_mask(
                result,
                kernel_size=close_kernel_size,
                iterations=close_iterations,
            )
        elif step == "fill_holes":
            result = fill_mask_holes(result)
        elif step == "remove_small_components":
            result = remove_small_mask_components(result, min_area=min_component_area)
        elif step == "clear_border":
            result = clear_border_components(result)
        elif step in {"none", "identity"}:
            continue
        else:
            raise ValueError(f"Unsupported mask augmentation step {step!r}.")

    return as_binary_mask(result)


def _kernel_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    return max(1, width), max(1, height)
