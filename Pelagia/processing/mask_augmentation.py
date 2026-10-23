"""Morphological cleanup operations for binary segmentation masks."""

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
    """Fill background holes enclosed by foreground regions.

    The one-pixel background border makes the flood-fill seed independent of
    the source mask's corner values.  In particular, a foreground pixel at
    ``(0, 0)`` can no longer cause all exterior background to be mistaken for
    an enclosed hole.
    """
    binary = as_binary_mask(mask)
    if binary.size == 0:
        return binary
    holes = _enclosed_background(binary)
    output = binary.copy()
    output[holes] = 255
    return np.ascontiguousarray(output)


def fill_mask_holes_with_audit(
    mask: np.ndarray,
    *,
    protected_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Fill ordinary holes while retaining holes that contain protected seeds.

    ``protected_mask`` represents pixels owned by other ROI candidates.  When
    any protected pixel lies in an enclosed background component, that entire
    component remains background so hole filling cannot merge the candidates.
    """
    binary = as_binary_mask(mask)
    protected = (
        np.zeros(binary.shape, dtype=bool)
        if protected_mask is None
        else as_binary_mask(protected_mask) > 0
    )
    if protected.shape != binary.shape:
        raise ValueError("protected_mask and mask must have matching shapes.")
    if np.any((binary > 0) & protected):
        raise ValueError("protected_mask cannot overlap foreground mask pixels.")

    audit: dict[str, int | str] = {
        "hole_fill_method": "padded_border_flood_fill_v1",
        "hole_fill_version": 1,
        "background_connectivity": 4,
        "holes_detected": 0,
        "holes_filled": 0,
        "hole_pixels_filled": 0,
        "protected_holes": 0,
        "protected_hole_pixels": 0,
        "protected_seed_pixels": int(np.count_nonzero(protected)),
        "remaining_unprotected_hole_pixels": 0,
        "final_mask_validation": "passed",
    }
    if binary.size == 0:
        return binary, audit

    holes = _enclosed_background(binary)
    component_count, labels = cv2.connectedComponents(
        holes.astype(np.uint8), connectivity=4
    )
    protected_labels = set(int(value) for value in np.unique(labels[protected & holes]))
    protected_labels.discard(0)
    fill = holes.copy()
    for label in protected_labels:
        fill[labels == label] = False

    hole_count = max(0, int(component_count) - 1)
    protected_hole_pixels = int(np.count_nonzero(holes & ~fill))
    filled_pixels = int(np.count_nonzero(fill))
    output = binary.copy()
    output[fill] = 255

    remaining = _enclosed_background(output)
    remaining_unprotected = remaining & ~protected
    # Protected holes include background around the seed, so validate by
    # component ownership rather than requiring every remaining pixel itself
    # to be a protected pixel.
    if np.any(remaining_unprotected):
        remaining_count, remaining_labels = cv2.connectedComponents(
            remaining.astype(np.uint8), connectivity=4
        )
        for label in range(1, remaining_count):
            component = remaining_labels == label
            if not np.any(component & protected):
                raise RuntimeError("Heuristic hole filling left an unprotected enclosed hole.")
    if np.any((output > 0) & protected):
        raise RuntimeError("Heuristic hole filling absorbed a protected candidate seed.")

    audit.update(
        {
            "holes_detected": hole_count,
            "holes_filled": hole_count - len(protected_labels),
            "hole_pixels_filled": filled_pixels,
            "protected_holes": len(protected_labels),
            "protected_hole_pixels": protected_hole_pixels,
        }
    )
    return np.ascontiguousarray(output), audit


def finalize_heuristic_mask(
    mask: np.ndarray,
    *,
    seed_mask: np.ndarray,
    protected_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Apply mandatory hole filling and validate heuristic mask invariants."""
    binary = as_binary_mask(mask)
    seed = as_binary_mask(seed_mask) > 0
    protected = (
        np.zeros(binary.shape, dtype=bool)
        if protected_mask is None
        else as_binary_mask(protected_mask) > 0
    )
    if seed.shape != binary.shape or protected.shape != binary.shape:
        raise ValueError("Heuristic mask, seed_mask, and protected_mask must have matching shapes.")
    if np.any(seed & protected):
        raise ValueError("seed_mask and protected_mask cannot overlap.")

    normalized = binary > 0
    normalized |= seed
    normalized &= ~protected
    normalized |= seed
    filled, audit = fill_mask_holes_with_audit(
        normalized.astype(np.uint8) * 255,
        protected_mask=protected.astype(np.uint8) * 255,
    )
    if not np.all(filled[seed] == 255):
        raise RuntimeError("Heuristic final mask validation lost immutable seed pixels.")
    if np.any(filled[protected] > 0):
        raise RuntimeError("Heuristic final mask validation absorbed a competing seed.")
    audit["final_foreground_pixels"] = int(np.count_nonzero(filled))
    return filled, audit


def _enclosed_background(mask: np.ndarray) -> np.ndarray:
    """Return four-connected background components that cannot reach outside."""
    binary = as_binary_mask(mask)
    height, width = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255, flags=4)
    return flood[1:-1, 1:-1] == 0


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
