from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np

ThresholdFn = Callable[[np.ndarray], np.ndarray]


def as_uint8_grayscale(gray: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 2D image suitable for OpenCV thresholding."""
    image = np.asarray(gray)
    if image.ndim != 2:
        raise ValueError("Thresholding expects a 2D grayscale image.")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def threshold_manual(
    gray: np.ndarray,
    threshold: int | float,
) -> np.ndarray:
    """Apply a fixed binary threshold and return a uint8 mask."""
    image = as_uint8_grayscale(gray)
    _, mask = cv2.threshold(image, float(threshold), 255, cv2.THRESH_BINARY)
    return np.ascontiguousarray(mask)


def otsu_threshold_value(gray: np.ndarray) -> float:
    """Return the Otsu threshold value for a grayscale image."""
    image = as_uint8_grayscale(gray)
    threshold_value, _ = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return float(threshold_value)


def threshold_otsu(
    gray: np.ndarray,
    *,
    thresholding_maximum_value: int | float | None = None,
) -> np.ndarray:
    """Apply Otsu thresholding, optionally capped at a maximum threshold value."""
    threshold_value = otsu_threshold_value(gray)
    if thresholding_maximum_value is not None:
        threshold_value = min(threshold_value, float(thresholding_maximum_value))
    return threshold_manual(gray, threshold_value)


def threshold_bounded_otsu(
    gray: np.ndarray,
    *,
    min_contrast: int | float = 50,
    max_foreground_fraction: float = 0.9,
    thresholding_maximum_value: int | float | None = None,
) -> np.ndarray:
    """
    Apply Otsu thresholding with safeguards against background-noise separation.

    Thresholding assumes preprocessing has normalized polarity so foreground is
    brighter than background.
    """
    image = as_uint8_grayscale(gray)
    threshold_value = otsu_threshold_value(image)
    if thresholding_maximum_value is not None:
        threshold_value = min(threshold_value, float(thresholding_maximum_value))

    background = float(np.median(image))
    if threshold_value - background < float(min_contrast):
        threshold_value = min(255.0, background + float(min_contrast))

    mask = threshold_manual(image, threshold_value)
    foreground_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    if foreground_fraction > float(max_foreground_fraction):
        return np.zeros_like(image, dtype=np.uint8)
    return mask


def threshold_canny(
    gray: np.ndarray,
    *,
    canny_params: tuple[int | float, int | float] = (30, 80),
    blur_kernel: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Return a Canny edge mask for a grayscale image."""
    image = as_uint8_grayscale(gray)
    if blur_kernel[0] > 1 or blur_kernel[1] > 1:
        image = cv2.GaussianBlur(image, blur_kernel, 0)
    edges = cv2.Canny(
        image,
        float(canny_params[0]),
        float(canny_params[1]),
        L2gradient=True,
    )
    return np.ascontiguousarray(edges)


def threshold_adaptive_mean(
    gray: np.ndarray,
    *,
    block_size: int = 31,
    c: int | float = 5,
) -> np.ndarray:
    """Apply local mean adaptive thresholding."""
    image = as_uint8_grayscale(gray)
    resolved_block_size = _odd_block_size(block_size)
    return np.ascontiguousarray(
        cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            resolved_block_size,
            float(c),
        )
    )


def threshold_adaptive_gaussian(
    gray: np.ndarray,
    *,
    block_size: int = 31,
    c: int | float = 5,
) -> np.ndarray:
    """Apply local Gaussian-weighted adaptive thresholding."""
    image = as_uint8_grayscale(gray)
    resolved_block_size = _odd_block_size(block_size)
    return np.ascontiguousarray(
        cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            resolved_block_size,
            float(c),
        )
    )


def threshold_percentile_background(
    gray: np.ndarray,
    *,
    background_percentile: int | float = 50,
    min_contrast: int | float = 50,
) -> np.ndarray:
    """Threshold relative to a percentile-estimated background brightness."""
    image = as_uint8_grayscale(gray)
    background = float(np.percentile(image, background_percentile))
    threshold_value = min(255.0, background + float(min_contrast))
    return threshold_manual(image, threshold_value)


def threshold_hysteresis(
    gray: np.ndarray,
    *,
    low_threshold: int | float,
    high_threshold: int | float,
    connectivity: int = 8,
) -> np.ndarray:
    """Keep weak threshold pixels only when connected to strong threshold pixels."""
    image = as_uint8_grayscale(gray)
    strong = image >= float(high_threshold)
    weak = image >= float(low_threshold)

    if not np.any(strong):
        return np.zeros_like(image, dtype=np.uint8)

    num, labels, _, _ = cv2.connectedComponentsWithStats(
        weak.astype(np.uint8),
        connectivity=int(connectivity),
    )
    if num <= 1:
        return np.zeros_like(image, dtype=np.uint8)

    connected_labels = np.unique(labels[strong])
    connected_labels = connected_labels[connected_labels != 0]
    mask = np.isin(labels, connected_labels).astype(np.uint8) * 255
    return np.ascontiguousarray(mask)


def threshold_sobel_edges(
    gray: np.ndarray,
    *,
    threshold: int | float | None = None,
    percentile: int | float = 90,
    kernel_size: int = 3,
) -> np.ndarray:
    """Threshold Sobel gradient magnitude to produce an edge mask."""
    image = as_uint8_grayscale(gray)
    resolved_kernel_size = _odd_kernel_size(kernel_size)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=resolved_kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=resolved_kernel_size)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if threshold is None:
        threshold = float(np.percentile(magnitude, percentile))
    mask = (magnitude >= float(threshold)).astype(np.uint8) * 255
    return np.ascontiguousarray(mask)


def dilate_mask(mask: np.ndarray, *, kernel_size: tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask with a rectangular kernel."""
    binary = np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    kernel = np.ones(kernel_size, np.uint8)
    return np.ascontiguousarray(cv2.dilate(binary, kernel, iterations=int(iterations)))


def combine_masks_or(*masks: np.ndarray) -> np.ndarray:
    """Combine threshold masks using a logical OR."""
    if not masks:
        raise ValueError("At least one mask is required.")
    combined = np.zeros_like(np.asarray(masks[0]), dtype=np.uint8)
    for mask in masks:
        combined = combined | np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    return np.ascontiguousarray(combined)


def combine_masks_and(*masks: np.ndarray) -> np.ndarray:
    """Combine threshold masks using a logical AND."""
    if not masks:
        raise ValueError("At least one mask is required.")
    combined = np.ascontiguousarray((np.asarray(masks[0]) > 0).astype(np.uint8) * 255)
    for mask in masks[1:]:
        combined = combined & np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    return np.ascontiguousarray(combined)


def threshold_ensemble_or(gray: np.ndarray, threshold_fns: Sequence[ThresholdFn]) -> np.ndarray:
    """Apply multiple threshold callables and combine their masks by OR."""
    return combine_masks_or(*(fn(gray) for fn in threshold_fns))


def threshold_ensemble_and(gray: np.ndarray, threshold_fns: Sequence[ThresholdFn]) -> np.ndarray:
    """Apply multiple threshold callables and combine their masks by AND."""
    return combine_masks_and(*(fn(gray) for fn in threshold_fns))


def threshold_boundedotsu_canny(
    gray: np.ndarray,
    *,
    run_canny: bool = True,
    canny_params: tuple[int | float, int | float] = (30, 80),
    dilate_kernel: tuple[int, int] = (3, 3),
    min_contrast: int | float = 50,
    max_foreground_fraction: float = 0.9,
    thresholding_maximum_value: int | float | None = None,
) -> np.ndarray:
    """Combine bounded Otsu thresholding with optional Canny edge support."""
    mask = threshold_bounded_otsu(
        gray,
        min_contrast=min_contrast,
        max_foreground_fraction=max_foreground_fraction,
        thresholding_maximum_value=thresholding_maximum_value,
    )
    if not np.any(mask):
        return mask
    if run_canny:
        mask = combine_masks_or(mask, threshold_canny(gray, canny_params=canny_params))
    return dilate_mask(mask, kernel_size=dilate_kernel)


def threshold_bounded_otsu_canny(*args, **kwargs) -> np.ndarray:
    """Alias for callers that prefer separated words in the function name."""
    return threshold_boundedotsu_canny(*args, **kwargs)


def _odd_block_size(block_size: int) -> int:
    resolved = int(block_size)
    if resolved < 3:
        raise ValueError("block_size must be >= 3.")
    if resolved % 2 == 0:
        resolved += 1
    return resolved


def _odd_kernel_size(kernel_size: int) -> int:
    resolved = int(kernel_size)
    if resolved < 1:
        raise ValueError("kernel_size must be >= 1.")
    if resolved % 2 == 0:
        resolved += 1
    return resolved
