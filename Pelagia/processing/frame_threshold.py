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


def calculate_threshold_mask(
    gray: np.ndarray,
    *,
    threshold: int | float | ThresholdFn | None = None,
    method: str = "otsu",
    manual_threshold: int | float | None = None,
    thresholding_maximum_value: int | float | None = None,
    bounded_otsu_min_contrast: int | float = 50,
    bounded_otsu_max_foreground_fraction: float = 0.9,
    canny_enabled: bool = True,
    canny_low_threshold: int | float = 30,
    canny_high_threshold: int | float = 80,
    canny_blur_kernel: int = 5,
    adaptive_block_size: int = 31,
    adaptive_c: int | float = 5,
    percentile_background_percentile: int | float = 50,
    percentile_min_contrast: int | float = 50,
    hysteresis_low_threshold: int | float = 30,
    hysteresis_high_threshold: int | float = 80,
    hysteresis_connectivity: int = 8,
    sobel_percentile: int | float = 90,
    sobel_threshold: int | float | None = None,
    sobel_kernel_size: int = 3,
) -> np.ndarray:
    """Calculate the first-pass binary mask using one thresholding algorithm."""
    if callable(threshold):
        return _binary_mask(threshold(gray))

    resolved_method = str(method).replace("-", "_").lower()
    if resolved_method == "auto":
        resolved_method = "otsu" if threshold is None else "manual"

    if resolved_method == "manual":
        resolved_threshold = threshold if threshold is not None else manual_threshold
        if resolved_threshold is None:
            raise ValueError("Manual thresholding requires a threshold value.")
        return _binary_mask(threshold_manual(gray, resolved_threshold))
    if resolved_method == "otsu":
        return _binary_mask(
            threshold_otsu(
                gray,
                thresholding_maximum_value=thresholding_maximum_value,
            )
        )
    if resolved_method == "bounded_otsu":
        return _binary_mask(
            threshold_bounded_otsu(
                gray,
                min_contrast=bounded_otsu_min_contrast,
                max_foreground_fraction=bounded_otsu_max_foreground_fraction,
                thresholding_maximum_value=thresholding_maximum_value,
            )
        )
    if resolved_method in {"bounded_otsu_canny", "boundedotsu_canny"}:
        base = threshold_bounded_otsu(
            gray,
            min_contrast=bounded_otsu_min_contrast,
            max_foreground_fraction=bounded_otsu_max_foreground_fraction,
            thresholding_maximum_value=thresholding_maximum_value,
        )
        if not np.any(base):
            return _binary_mask(base)
        if not canny_enabled:
            return _binary_mask(base)
        edges = threshold_canny(
            gray,
            canny_params=(canny_low_threshold, canny_high_threshold),
            blur_kernel=(int(canny_blur_kernel), int(canny_blur_kernel)),
        )
        return _binary_mask(combine_masks_or(base, edges))
    if resolved_method == "canny":
        return _binary_mask(
            threshold_canny(
                gray,
                canny_params=(canny_low_threshold, canny_high_threshold),
                blur_kernel=(int(canny_blur_kernel), int(canny_blur_kernel)),
            )
        )
    if resolved_method == "adaptive_mean":
        return _binary_mask(threshold_adaptive_mean(gray, block_size=int(adaptive_block_size), c=adaptive_c))
    if resolved_method == "adaptive_gaussian":
        return _binary_mask(
            threshold_adaptive_gaussian(gray, block_size=int(adaptive_block_size), c=adaptive_c)
        )
    if resolved_method == "percentile_background":
        return _binary_mask(
            threshold_percentile_background(
                gray,
                background_percentile=percentile_background_percentile,
                min_contrast=percentile_min_contrast,
            )
        )
    if resolved_method == "hysteresis":
        return _binary_mask(
            threshold_hysteresis(
                gray,
                low_threshold=hysteresis_low_threshold,
                high_threshold=hysteresis_high_threshold,
                connectivity=int(hysteresis_connectivity),
            )
        )
    if resolved_method == "sobel_edges":
        return _binary_mask(
            threshold_sobel_edges(
                gray,
                threshold=threshold if threshold is not None else sobel_threshold,
                percentile=sobel_percentile,
                kernel_size=int(sobel_kernel_size),
            )
        )
    raise ValueError(f"Unsupported threshold method {method!r}.")


def threshold_bounded_otsu_canny(
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


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Threshold functions must return a 2D mask.")
    return np.ascontiguousarray((array > 0).astype(np.uint8) * 255)
