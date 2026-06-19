from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import cv2
import numpy as np

from ..domain import DetectionRecord
from .detection_recording import build_candidate_detection_record
from .frame_codec import decode_array_payload, encode_array_payload
from .frame_model import FrameData
from .frame_preprocess import as_binary_mask, as_grayscale_array
from .roi_assembly import assemble_candidate_rois
from .roi_filter import filter_candidate_rois


RefinementFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
FrameLoader = Callable[[str], np.ndarray]


class RoiRefinementModel(Protocol):
    """Minimal model interface for U-Net style ROI mask refinement."""

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Return a KxHxW refined-mask batch for a KxHxWx2 input batch."""


@dataclass(slots=True)
class RoiRefinementOptions:
    """Options controlling tiled ROI refinement and optional crop expansion."""

    tile_size: int = 256
    overlap_fraction: float = 0.25
    max_iterations: int = 3
    expansion_pixels: int | None = None
    edge_touch_margin: int = 1
    output_threshold: float = 0.5
    batch_size: int | None = None
    encoding: str | None = None
    overlap_reconciliation_enabled: bool = True
    overlap_iou_threshold: float = 0.5
    overlap_containment_threshold: float = 0.8
    residual_discovery_enabled: bool = False
    residual_max_iterations: int = 1
    residual_roi_assembly_method: str | None = None
    residual_roi_assembly_connectivity: int = 8
    residual_min_area: float | None = None
    residual_min_width: float | None = None
    residual_min_height: float | None = None
    residual_min_width_plus_height: float | None = None
    residual_padding: int | None = None

    def __post_init__(self) -> None:
        if int(self.tile_size) < 1:
            raise ValueError("tile_size must be >= 1.")
        self.tile_size = int(self.tile_size)
        if not 0 <= float(self.overlap_fraction) < 1:
            raise ValueError("overlap_fraction must be >= 0 and < 1.")
        self.overlap_fraction = float(self.overlap_fraction)
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1.")
        self.max_iterations = int(self.max_iterations)
        if self.expansion_pixels is not None and int(self.expansion_pixels) < 1:
            raise ValueError("expansion_pixels must be >= 1 when provided.")
        if int(self.edge_touch_margin) < 1:
            raise ValueError("edge_touch_margin must be >= 1.")
        self.edge_touch_margin = int(self.edge_touch_margin)
        if self.batch_size is not None and int(self.batch_size) < 1:
            raise ValueError("batch_size must be >= 1 when provided.")
        if self.batch_size is not None:
            self.batch_size = int(self.batch_size)
        if not 0 <= float(self.overlap_iou_threshold) <= 1:
            raise ValueError("overlap_iou_threshold must be between 0 and 1.")
        self.overlap_iou_threshold = float(self.overlap_iou_threshold)
        if not 0 <= float(self.overlap_containment_threshold) <= 1:
            raise ValueError("overlap_containment_threshold must be between 0 and 1.")
        self.overlap_containment_threshold = float(self.overlap_containment_threshold)
        if int(self.residual_max_iterations) < 1:
            raise ValueError("residual_max_iterations must be >= 1.")
        self.residual_max_iterations = int(self.residual_max_iterations)
        if int(self.residual_roi_assembly_connectivity) not in {4, 8}:
            raise ValueError("residual_roi_assembly_connectivity must be 4 or 8.")
        self.residual_roi_assembly_connectivity = int(self.residual_roi_assembly_connectivity)
        if self.residual_padding is not None and int(self.residual_padding) < 0:
            raise ValueError("residual_padding must be >= 0 when provided.")
        if self.residual_padding is not None:
            self.residual_padding = int(self.residual_padding)

    @property
    def overlap_pixels(self) -> int:
        return int(round(self.tile_size * self.overlap_fraction))

    @property
    def stride(self) -> int:
        return max(1, self.tile_size - self.overlap_pixels)

    @property
    def resolved_expansion_pixels(self) -> int:
        return int(self.expansion_pixels or self.stride)


@dataclass(slots=True)
class RoiTile:
    """One fixed-size tile plus coordinates needed to merge it back."""

    detection_index: int
    tile_index: int
    frame_id: str
    roi_index: int
    crop_bbox: tuple[int, int, int, int]
    local_bbox: tuple[int, int, int, int]
    frame_bbox: tuple[int, int, int, int]
    valid_shape: tuple[int, int]
    image: np.ndarray
    mask: np.ndarray


@dataclass(slots=True)
class DetectionRefinementResult:
    """Runtime result for moving a candidate ROI mask toward a refined ROI mask."""

    candidate_detection: DetectionRecord
    roi: np.ndarray
    candidate_mask: np.ndarray
    refined_mask: np.ndarray
    crop_bbox: tuple[int, int, int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    tiles: list[RoiTile] = field(default_factory=list)
    method: str = "identity"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_detection_record(self, *, encoding: str | None = None) -> DetectionRecord:
        """Return a copy of the candidate record annotated as refined metadata."""
        if _refined_mask_area(self) <= 0:
            raise ValueError(
                f"Refined detection for candidate {_detection_identifier(self.candidate_detection)!r} has an empty mask."
            )
        metadata = dict(self.candidate_detection.metadata or {})
        metadata.update(self.metadata)
        metadata.update(
            {
                "detection_stage": "refined",
                "mask_kind": "refined",
                "candidate_detection_id": self.candidate_detection.id,
                "refinement_method": self.method,
            }
        )
        requested_encoding = encoding or self.candidate_detection.roi_encoding or "png"
        roi_payload, roi_encoding, roi_format = encode_array_payload(self.roi, requested_encoding)
        mask_payload, mask_encoding, mask_format = encode_array_payload(self.refined_mask, requested_encoding)
        bbox = self.bbox or _mask_bbox_in_frame(self.refined_mask, self.crop_bbox or _detection_crop_bbox(self.candidate_detection))
        crop_bbox = self.crop_bbox or _detection_crop_bbox(self.candidate_detection)
        area, perimeter, major_axis, minor_axis = _mask_measurements(self.refined_mask)
        min_gray, mean_gray = _gray_stats(self.roi, self.refined_mask)
        return replace(
            self.candidate_detection,
            bbox_x=int(bbox[0]),
            bbox_y=int(bbox[1]),
            bbox_w=int(bbox[2]),
            bbox_h=int(bbox[3]),
            crop_bbox_x=int(crop_bbox[0]),
            crop_bbox_y=int(crop_bbox[1]),
            crop_bbox_w=int(crop_bbox[2]),
            crop_bbox_h=int(crop_bbox[3]),
            area=float(area),
            perimeter=float(perimeter),
            major_axis_length=float(major_axis),
            minor_axis_length=float(minor_axis),
            min_gray_value=int(min_gray),
            mean_gray_value=float(mean_gray),
            roi_payload=roi_payload,
            mask_payload=mask_payload,
            roi_encoding=roi_encoding,
            roi_format=roi_format,
            roi_dtype=str(self.roi.dtype),
            roi_shape=list(self.roi.shape),
            mask_encoding=mask_encoding,
            mask_format=mask_format,
            mask_dtype=str(self.refined_mask.dtype),
            mask_shape=list(self.refined_mask.shape),
            metadata=metadata,
        )


class IdentityRoiRefinementModel:
    """Trivial refinement model: return the candidate mask channel unchanged."""

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4 or batch.shape[-1] != 2:
            raise ValueError("Refinement batch must have shape KxHxWx2.")
        return np.ascontiguousarray(batch[..., 1])


class CallableTileRefinementModel:
    """Adapter for legacy/refinement callbacks that accept one ROI and mask."""

    def __init__(self, refiner: RefinementFn):
        self.refiner = refiner

    def predict(self, batch: np.ndarray) -> np.ndarray:
        outputs = []
        for tile in batch:
            roi = tile[..., 0]
            mask = (tile[..., 1] > 0).astype(np.uint8) * 255
            outputs.append(as_binary_mask(self.refiner(roi, mask)).astype(np.float32) / 255.0)
        return np.stack(outputs, axis=0)


def decode_detection_roi(detection: DetectionRecord) -> np.ndarray:
    """Decode a stored detection ROI payload into a contiguous numpy array."""
    if detection.roi_payload is None:
        raise ValueError("Detection does not include ROI payload data.")
    metadata = {
        "array_encoding": detection.roi_encoding,
        "dtype": detection.roi_dtype,
        "shape": detection.roi_shape,
    }
    return np.ascontiguousarray(decode_array_payload(detection.roi_payload, metadata))


def decode_detection_candidate_mask(detection: DetectionRecord, roi: np.ndarray | None = None) -> np.ndarray:
    """Decode the candidate ROI mask, or derive one from non-zero ROI pixels."""
    if detection.mask_payload is not None:
        metadata = {
            "array_encoding": detection.mask_encoding,
            "dtype": detection.mask_dtype,
            "shape": detection.mask_shape,
        }
        return as_binary_mask(decode_array_payload(detection.mask_payload, metadata))

    source_roi = roi if roi is not None else decode_detection_roi(detection)
    return as_binary_mask(as_grayscale_array(source_roi))


def load_detection_roi_from_frame(
    detection: DetectionRecord,
    *,
    frame_loader: FrameLoader,
) -> np.ndarray:
    """Load the source frame and crop the candidate ROI when no ROI payload is stored."""
    frame = as_grayscale_array(frame_loader(detection.frame_id))
    return _crop_detection_roi_from_frame(detection, frame)


def _crop_detection_roi_from_frame(detection: DetectionRecord, frame: np.ndarray) -> np.ndarray:
    """Crop a detection ROI from an already-loaded grayscale source frame."""
    crop_x, crop_y, crop_w, crop_h = _detection_crop_bbox(detection)
    if crop_x < 0 or crop_y < 0 or crop_w < 1 or crop_h < 1:
        raise ValueError(f"Detection {detection.id or detection.roi_index!r} has an invalid crop bbox.")
    frame_h, frame_w = frame.shape[:2]
    if crop_x + crop_w > frame_w or crop_y + crop_h > frame_h:
        raise ValueError(
            f"Detection crop bbox {(crop_x, crop_y, crop_w, crop_h)} exceeds frame bounds {(frame_w, frame_h)}."
        )
    return np.ascontiguousarray(frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w])


def build_roi_tiles(
    detection: DetectionRecord,
    roi: np.ndarray,
    mask: np.ndarray,
    *,
    options: RoiRefinementOptions | None = None,
    detection_index: int = 0,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> list[RoiTile]:
    """Build fixed-size overlapping tiles for one ROI crop and mask."""
    resolved_options = options or RoiRefinementOptions()
    image = as_grayscale_array(roi)
    binary_mask = as_binary_mask(mask)
    if image.shape[:2] != binary_mask.shape[:2]:
        raise ValueError(
            f"Candidate mask shape {binary_mask.shape[:2]} does not match ROI shape {image.shape[:2]}."
        )
    crop_x, crop_y, crop_w, crop_h = crop_bbox or _detection_crop_bbox(detection, image.shape)
    starts_y = _tile_starts(image.shape[0], resolved_options.tile_size, resolved_options.stride)
    starts_x = _tile_starts(image.shape[1], resolved_options.tile_size, resolved_options.stride)
    tiles: list[RoiTile] = []
    tile_index = 0
    for y0 in starts_y:
        for x0 in starts_x:
            valid_h = min(resolved_options.tile_size, image.shape[0] - y0)
            valid_w = min(resolved_options.tile_size, image.shape[1] - x0)
            tile_image = np.zeros((resolved_options.tile_size, resolved_options.tile_size), dtype=image.dtype)
            tile_mask = np.zeros((resolved_options.tile_size, resolved_options.tile_size), dtype=np.uint8)
            tile_image[:valid_h, :valid_w] = image[y0:y0 + valid_h, x0:x0 + valid_w]
            tile_mask[:valid_h, :valid_w] = binary_mask[y0:y0 + valid_h, x0:x0 + valid_w]
            tiles.append(
                RoiTile(
                    detection_index=detection_index,
                    tile_index=tile_index,
                    frame_id=detection.frame_id,
                    roi_index=detection.roi_index,
                    crop_bbox=(int(crop_x), int(crop_y), int(crop_w), int(crop_h)),
                    local_bbox=(int(x0), int(y0), int(valid_w), int(valid_h)),
                    frame_bbox=(int(crop_x + x0), int(crop_y + y0), int(valid_w), int(valid_h)),
                    valid_shape=(int(valid_h), int(valid_w)),
                    image=np.ascontiguousarray(tile_image),
                    mask=np.ascontiguousarray(tile_mask),
                )
            )
            tile_index += 1
    return tiles


def refinement_batch_from_tiles(tiles: list[RoiTile]) -> np.ndarray:
    """Return KxHxWx2 model input from ROI tiles."""
    if not tiles:
        raise ValueError("At least one ROI tile is required.")
    batch = np.zeros((len(tiles), tiles[0].image.shape[0], tiles[0].image.shape[1], 2), dtype=np.float32)
    for index, tile in enumerate(tiles):
        batch[index, :, :, 0] = tile.image.astype(np.float32)
        batch[index, :, :, 1] = (tile.mask > 0).astype(np.float32)
    return np.ascontiguousarray(batch)


def predict_refined_tile_masks(
    tiles: list[RoiTile],
    *,
    model: RoiRefinementModel | None = None,
    options: RoiRefinementOptions | None = None,
) -> list[np.ndarray]:
    """Run a U-Net style model over ROI tiles and return binary tile masks."""
    resolved_model = model or IdentityRoiRefinementModel()
    resolved_options = options or RoiRefinementOptions()
    if resolved_options.batch_size:
        prediction_chunks = []
        for start in range(0, len(tiles), resolved_options.batch_size):
            prediction_chunks.append(
                np.asarray(resolved_model.predict(refinement_batch_from_tiles(tiles[start:start + resolved_options.batch_size])))
            )
        predictions = np.concatenate(prediction_chunks, axis=0)
    else:
        predictions = np.asarray(resolved_model.predict(refinement_batch_from_tiles(tiles)))
    if predictions.ndim != 3 or predictions.shape[0] != len(tiles):
        raise ValueError("Refinement model must return a KxHxW mask batch.")
    if predictions.shape[1:3] != tiles[0].image.shape[:2]:
        raise ValueError("Refinement model output tile size does not match input tile size.")
    return [_prediction_to_mask(prediction, threshold=resolved_options.output_threshold) for prediction in predictions]


def merge_refined_tiles(
    tiles: list[RoiTile],
    tile_masks: list[np.ndarray],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Merge fixed-size tile masks back into a full ROI-crop mask."""
    if len(tiles) != len(tile_masks):
        raise ValueError("tiles and tile_masks must have the same length.")
    accumulator = np.zeros(output_shape, dtype=np.float32)
    weights = np.zeros(output_shape, dtype=np.float32)
    for tile, tile_mask in zip(tiles, tile_masks):
        x0, y0, width, height = tile.local_bbox
        valid_mask = as_binary_mask(tile_mask)[:height, :width].astype(np.float32) / 255.0
        accumulator[y0:y0 + height, x0:x0 + width] += valid_mask
        weights[y0:y0 + height, x0:x0 + width] += 1.0
    merged = np.divide(accumulator, np.maximum(weights, 1.0))
    return np.ascontiguousarray((merged >= 0.5).astype(np.uint8) * 255)


def refine_detection(
    detection: DetectionRecord,
    *,
    model: RoiRefinementModel | None = None,
    refiner: RefinementFn | None = None,
    frame_loader: FrameLoader | None = None,
    options: RoiRefinementOptions | None = None,
    method: str | None = None,
) -> DetectionRefinementResult:
    """
    Refine one candidate detection ROI using tiled mask prediction.

    The initial pass uses the stored ROI and mask payloads. The full source frame
    is only loaded if the refined mask reaches an uncovered edge of the current
    ROI crop and a ``frame_loader`` is supplied.
    """
    resolved_options = options or RoiRefinementOptions()
    resolved_model: RoiRefinementModel | None = model
    if resolved_model is None and refiner is not None:
        resolved_model = CallableTileRefinementModel(refiner)
    if resolved_model is None:
        resolved_model = IdentityRoiRefinementModel()
    resolved_method = method or (
        "identity" if isinstance(resolved_model, IdentityRoiRefinementModel) else resolved_model.__class__.__name__
    )

    full_frame: np.ndarray | None = None
    frame_loaded = False
    roi_source = "payload"
    if detection.roi_payload is None:
        if frame_loader is None:
            raise ValueError(
                "Detection does not include ROI payload data and no frame_loader was supplied."
            )
        full_frame = as_grayscale_array(frame_loader(detection.frame_id))
        frame_loaded = True
        roi_source = "frame"
        roi = _crop_detection_roi_from_frame(detection, full_frame)
    else:
        roi = as_grayscale_array(decode_detection_roi(detection))
    candidate_mask = decode_detection_candidate_mask(detection, roi=roi)
    if candidate_mask.shape[:2] != roi.shape[:2]:
        raise ValueError(
            f"Candidate mask shape {candidate_mask.shape[:2]} does not match ROI shape {roi.shape[:2]}."
        )

    crop_bbox = _detection_crop_bbox(detection, roi.shape)
    expansion_count = 0
    boundary_expansion_required = False
    last_tiles: list[RoiTile] = []
    refined_mask = candidate_mask

    for iteration in range(resolved_options.max_iterations):
        last_tiles = build_roi_tiles(
            detection,
            roi,
            refined_mask,
            options=resolved_options,
            crop_bbox=crop_bbox,
        )
        tile_masks = predict_refined_tile_masks(
            last_tiles,
            model=resolved_model,
            options=resolved_options,
        )
        refined_mask = merge_refined_tiles(last_tiles, tile_masks, roi.shape[:2])
        touched = mask_touches_uncovered_edge(
            refined_mask,
            crop_bbox=crop_bbox,
            frame_shape=None if full_frame is None else full_frame.shape[:2],
            margin=resolved_options.edge_touch_margin,
        )
        if not any(touched.values()):
            break
        boundary_expansion_required = True
        if frame_loader is None:
            break
        if full_frame is None:
            full_frame = as_grayscale_array(frame_loader(detection.frame_id))
            frame_loaded = True
        new_crop_bbox = _expanded_crop_bbox(
            crop_bbox,
            touched,
            frame_shape=full_frame.shape[:2],
            expansion_pixels=resolved_options.resolved_expansion_pixels,
        )
        if new_crop_bbox == crop_bbox:
            break
        roi, refined_mask = _expand_roi_and_mask_from_frame(
            full_frame,
            current_roi=roi,
            current_mask=refined_mask,
            current_crop_bbox=crop_bbox,
            new_crop_bbox=new_crop_bbox,
        )
        crop_bbox = new_crop_bbox
        expansion_count += 1
        if iteration == resolved_options.max_iterations - 1:
            boundary_expansion_required = any(
                mask_touches_uncovered_edge(
                    refined_mask,
                    crop_bbox=crop_bbox,
                    frame_shape=full_frame.shape[:2],
                    margin=resolved_options.edge_touch_margin,
                ).values()
            )

    bbox = _mask_bbox_in_frame(refined_mask, crop_bbox)
    refined_foreground_pixels = int(np.count_nonzero(as_binary_mask(refined_mask)))
    return DetectionRefinementResult(
        candidate_detection=detection,
        roi=roi,
        candidate_mask=candidate_mask,
        refined_mask=refined_mask,
        crop_bbox=crop_bbox,
        bbox=bbox,
        tiles=last_tiles,
        method=resolved_method,
        metadata={
            "candidate_mask_kind": "candidate",
            "refined_mask_kind": "refined",
            "candidate_shape": list(candidate_mask.shape),
            "refined_shape": list(refined_mask.shape),
            "refined_foreground_pixels": refined_foreground_pixels,
            "refinement_tile_size": resolved_options.tile_size,
            "refinement_overlap_fraction": resolved_options.overlap_fraction,
            "refinement_tile_count": len(last_tiles),
            "refinement_expansion_count": expansion_count,
            "refinement_frame_loaded": frame_loaded,
            "refinement_initial_roi_source": roi_source,
            "refinement_boundary_expansion_required": boundary_expansion_required,
            "refinement_crop_bbox": crop_bbox,
            "refinement_bbox": bbox,
        },
    )


def refine_detections(
    detections: Iterable[DetectionRecord],
    *,
    model: RoiRefinementModel | None = None,
    refiner: RefinementFn | None = None,
    frame_loader: FrameLoader | None = None,
    options: RoiRefinementOptions | None = None,
    method: str | None = None,
) -> list[DetectionRefinementResult]:
    """Refine a batch of candidate detections and reconcile duplicate overlaps."""
    resolved_options = options or RoiRefinementOptions()
    results = [
        refine_detection(
            detection,
            model=model,
            refiner=refiner,
            frame_loader=frame_loader,
            options=resolved_options,
            method=method,
        )
        for detection in detections
    ]
    if resolved_options.residual_discovery_enabled:
        results = discover_residual_refinements(
            results,
            model=model,
            refiner=refiner,
            frame_loader=frame_loader,
            options=resolved_options,
            method=method,
        )
    results = _non_empty_refinement_results(results)
    return reconcile_overlapping_refinements(results, options=resolved_options)


def discover_residual_refinements(
    results: Iterable[DetectionRefinementResult],
    *,
    model: RoiRefinementModel | None = None,
    refiner: RefinementFn | None = None,
    frame_loader: FrameLoader | None = None,
    options: RoiRefinementOptions | None = None,
    method: str | None = None,
) -> list[DetectionRefinementResult]:
    """
    Discover additional targets left behind after refinement.

    Residual discovery subtracts the refined mask from the original candidate
    mask in the current ROI crop, assembles any remaining components into
    synthetic candidate detections, and refines those children. Residual
    children are refined inside the stored/expanded ROI crop only; full-frame
    loading remains reserved for the parent refinement expansion step.
    """
    resolved_options = options or RoiRefinementOptions()
    resolved_results = list(results)
    for result in resolved_results:
        result.metadata["residual_discovery_enabled"] = True
        result.metadata.setdefault("residual_discovery_children", [])
        result.metadata.setdefault("residual_discovery_child_count", 0)

    all_results = list(resolved_results)
    frontier = list(resolved_results)
    for generation in range(1, resolved_options.residual_max_iterations + 1):
        child_results: list[DetectionRefinementResult] = []
        for result in frontier:
            child_detections = _residual_candidate_detections(
                result,
                options=resolved_options,
                generation=generation,
            )
            for child_detection in child_detections:
                child_result = refine_detection(
                    child_detection,
                    model=model,
                    refiner=refiner,
                    frame_loader=None,
                    options=resolved_options,
                    method=method,
                )
                child_result.metadata.update(
                    {
                        "residual_discovery_enabled": True,
                        "residual_discovery_action": "split_child",
                        "residual_generation": generation,
                        "split_from_candidate_detection_id": child_detection.metadata.get(
                            "split_from_candidate_detection_id"
                        ),
                        "split_from_roi_index": child_detection.metadata.get("split_from_roi_index"),
                        "residual_component_index": child_detection.metadata.get(
                            "residual_component_index"
                        ),
                        "synthetic_candidate": True,
                    }
                )
                child_results.append(child_result)
                _record_residual_child(result, child_result)
        if not child_results:
            break
        all_results.extend(child_results)
        frontier = child_results

    for result in all_results:
        result.metadata.setdefault("residual_discovery_enabled", True)
        result.metadata.setdefault("residual_discovery_child_count", 0)
    return all_results


def reconcile_overlapping_refinements(
    results: Iterable[DetectionRefinementResult],
    *,
    options: RoiRefinementOptions | None = None,
) -> list[DetectionRefinementResult]:
    """
    Remove refined ROIs that substantially overlap a larger refined ROI.

    This handles the common case where candidate generation split one target
    into multiple ROIs, but refinement recovers the full target in more than
    one candidate. The kept ROI records metadata describing consumed
    candidates; consumed results are omitted from the returned list.
    """
    resolved_options = options or RoiRefinementOptions()
    resolved_results = list(results)
    if not resolved_options.overlap_reconciliation_enabled:
        for result in resolved_results:
            result.metadata["overlap_reconciliation_enabled"] = False
            result.metadata.setdefault("overlap_reconciliation_action", "unchecked")
        return resolved_results
    if len(resolved_results) <= 1:
        for result in resolved_results:
            result.metadata["overlap_reconciliation_enabled"] = True
            result.metadata.setdefault("overlap_reconciliation_action", "kept")
            result.metadata.setdefault("overlap_reconciliation_consumed_count", 0)
        return resolved_results

    kept: list[DetectionRefinementResult] = []
    sorted_results = sorted(resolved_results, key=_reconciliation_sort_key)
    for current in sorted_results:
        consumed_by: DetectionRefinementResult | None = None
        consumed_metrics: dict[str, Any] | None = None
        for keeper in kept:
            if current.candidate_detection.frame_id != keeper.candidate_detection.frame_id:
                continue
            metrics = _refined_overlap_metrics(current, keeper)
            if _substantial_refined_overlap(metrics, resolved_options):
                consumed_by = keeper
                consumed_metrics = metrics
                break

        current.metadata["overlap_reconciliation_enabled"] = True
        if consumed_by is None:
            current.metadata.setdefault("overlap_reconciliation_action", "kept")
            current.metadata.setdefault("overlap_reconciliation_consumed_count", 0)
            kept.append(current)
            continue

        current.metadata["overlap_reconciliation_action"] = "consumed"
        current.metadata["consumed_by_candidate_detection_id"] = _detection_identifier(
            consumed_by.candidate_detection
        )
        current.metadata["consumed_overlap_metrics"] = consumed_metrics or {}
        _record_consumed_refinement(consumed_by, current, consumed_metrics or {})

    return kept


def mask_touches_uncovered_edge(
    mask: np.ndarray,
    *,
    crop_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int] | None = None,
    margin: int = 1,
) -> dict[str, bool]:
    """Report whether a mask touches crop edges that can still be expanded."""
    binary = as_binary_mask(mask)
    if not np.any(binary):
        return {"left": False, "right": False, "top": False, "bottom": False}
    resolved_margin = max(1, int(margin))
    x, y, width, height = crop_bbox
    frame_height = None if frame_shape is None else int(frame_shape[0])
    frame_width = None if frame_shape is None else int(frame_shape[1])
    return {
        "left": bool(np.any(binary[:, :resolved_margin])) and x > 0,
        "right": bool(np.any(binary[:, max(0, width - resolved_margin):])) and (
            frame_width is None or x + width < frame_width
        ),
        "top": bool(np.any(binary[:resolved_margin, :])) and y > 0,
        "bottom": bool(np.any(binary[max(0, height - resolved_margin):, :])) and (
            frame_height is None or y + height < frame_height
        ),
    }


def _residual_candidate_detections(
    result: DetectionRefinementResult,
    *,
    options: RoiRefinementOptions,
    generation: int,
) -> list[DetectionRecord]:
    residual_mask = _residual_mask_for_result(result)
    if not np.any(residual_mask):
        result.metadata.setdefault("residual_discovery_remaining_pixels", 0)
        return []

    assembly_method = (
        options.residual_roi_assembly_method
        or result.candidate_detection.metadata.get("assembly_method")
        or "connected_components"
    )
    assembled = assemble_candidate_rois(
        residual_mask,
        method=str(assembly_method),
        connectivity=options.residual_roi_assembly_connectivity,
    )
    candidates = filter_candidate_rois(
        assembled,
        min_area=options.residual_min_area,
        min_width=options.residual_min_width,
        min_height=options.residual_min_height,
        min_width_plus_height=options.residual_min_width_plus_height,
    )
    result.metadata["residual_discovery_remaining_pixels"] = int(np.count_nonzero(residual_mask))
    result.metadata["residual_discovery_candidate_count"] = len(candidates)
    if not candidates:
        return []

    crop_bbox = result.crop_bbox or _detection_crop_bbox(result.candidate_detection, result.roi.shape)
    crop_x, crop_y, _, _ = crop_bbox
    residual_image = as_grayscale_array(result.roi)
    frame = FrameData(
        sourcePath="",
        filename=f"residual-{_detection_identifier(result.candidate_detection)}",
        frameNumber=int(result.candidate_detection.roi_index),
        data=np.ascontiguousarray(residual_image),
        bbox_x=int(crop_x),
        bbox_y=int(crop_y),
        metadata={
            "run_id": result.candidate_detection.run_id,
            "frame_id": result.candidate_detection.frame_id,
            "collections": result.candidate_detection.metadata.get("collections"),
            "residual_parent_candidate_detection_id": _detection_identifier(
                result.candidate_detection
            ),
        },
    )
    parent_padding = _candidate_padding(result.candidate_detection)
    padding = int(options.residual_padding if options.residual_padding is not None else parent_padding)
    detections: list[DetectionRecord] = []
    for index, candidate in enumerate(candidates, start=1):
        synthetic = build_candidate_detection_record(
            candidate,
            source_frame=frame,
            processed_frame=frame,
            roi_index=_residual_roi_index(result.candidate_detection, generation, index),
            padding=padding,
            encoding=result.candidate_detection.roi_encoding or options.encoding or "raw",
            store_roi_payload=True,
            always_store_mask=True,
            extra_metadata={
                "detection_stage": "candidate_residual",
                "synthetic_candidate": True,
                "residual_discovery_action": "split_candidate",
                "split_from_candidate_detection_id": _detection_identifier(
                    result.candidate_detection
                ),
                "split_from_roi_index": int(result.candidate_detection.roi_index),
                "residual_generation": int(generation),
                "residual_component_index": int(index),
                "residual_source_crop_bbox": crop_bbox,
                "residual_assembly_method": str(assembly_method),
            },
        )
        synthetic.id = _residual_detection_id(result.candidate_detection, generation, index)
        synthetic.metadata["synthetic_candidate_id"] = synthetic.id
        detections.append(synthetic)
    return detections


def _residual_mask_for_result(result: DetectionRefinementResult) -> np.ndarray:
    crop_bbox = result.crop_bbox or _detection_crop_bbox(result.candidate_detection, result.roi.shape)
    candidate_crop_bbox = _detection_crop_bbox(
        result.candidate_detection,
        result.candidate_mask.shape,
    )
    candidate_mask = np.zeros(as_binary_mask(result.refined_mask).shape, dtype=np.uint8)
    _paste_mask_into_crop(
        candidate_mask,
        as_binary_mask(result.candidate_mask),
        source_crop_bbox=candidate_crop_bbox,
        dest_crop_bbox=crop_bbox,
    )
    refined_mask = as_binary_mask(result.refined_mask)
    return np.ascontiguousarray(((candidate_mask > 0) & (refined_mask == 0)).astype(np.uint8) * 255)


def _paste_mask_into_crop(
    destination: np.ndarray,
    source: np.ndarray,
    *,
    source_crop_bbox: tuple[int, int, int, int],
    dest_crop_bbox: tuple[int, int, int, int],
) -> None:
    source_x, source_y, source_w, source_h = source_crop_bbox
    dest_x, dest_y, dest_w, dest_h = dest_crop_bbox
    rect = _intersection_rect(source_crop_bbox, dest_crop_bbox)
    if rect is None:
        return
    frame_x, frame_y, width, height = rect
    source_local_x = frame_x - source_x
    source_local_y = frame_y - source_y
    dest_local_x = frame_x - dest_x
    dest_local_y = frame_y - dest_y
    source_width = min(width, source.shape[1] - source_local_x, source_w - source_local_x)
    source_height = min(height, source.shape[0] - source_local_y, source_h - source_local_y)
    dest_width = min(width, destination.shape[1] - dest_local_x, dest_w - dest_local_x)
    dest_height = min(height, destination.shape[0] - dest_local_y, dest_h - dest_local_y)
    paste_width = max(0, min(source_width, dest_width))
    paste_height = max(0, min(source_height, dest_height))
    if paste_width <= 0 or paste_height <= 0:
        return
    destination[
        dest_local_y:dest_local_y + paste_height,
        dest_local_x:dest_local_x + paste_width,
    ] = source[
        source_local_y:source_local_y + paste_height,
        source_local_x:source_local_x + paste_width,
    ]


def _record_residual_child(
    parent: DetectionRefinementResult,
    child: DetectionRefinementResult,
) -> None:
    child_id = _detection_identifier(child.candidate_detection)
    entry = {
        "candidate_detection_id": child_id,
        "roi_index": int(child.candidate_detection.roi_index),
        "bbox": child.bbox,
        "crop_bbox": child.crop_bbox,
        "residual_generation": child.metadata.get("residual_generation"),
        "residual_component_index": child.metadata.get("residual_component_index"),
    }
    parent.metadata.setdefault("residual_discovery_children", [])
    parent.metadata["residual_discovery_children"].append(entry)
    parent.metadata["residual_discovery_child_count"] = len(
        parent.metadata["residual_discovery_children"]
    )


def _candidate_padding(detection: DetectionRecord) -> int:
    padding = detection.metadata.get("padding")
    if padding is not None:
        return max(0, int(padding))
    actual = detection.metadata.get("actual_padding")
    if isinstance(actual, dict) and actual:
        return max(0, max(int(value or 0) for value in actual.values()))
    return 0


def _residual_roi_index(detection: DetectionRecord, generation: int, component_index: int) -> int:
    return int(detection.roi_index) * 1000 + int(generation) * 100 + int(component_index)


def _residual_detection_id(detection: DetectionRecord, generation: int, component_index: int) -> str:
    return f"{_detection_identifier(detection)}:residual:{int(generation)}:{int(component_index)}"


def _record_consumed_refinement(
    keeper: DetectionRefinementResult,
    consumed: DetectionRefinementResult,
    metrics: dict[str, Any],
) -> None:
    consumed_id = _detection_identifier(consumed.candidate_detection)
    keeper_id = _detection_identifier(keeper.candidate_detection)
    consumed_entry = {
        "candidate_detection_id": consumed_id,
        "roi_index": int(consumed.candidate_detection.roi_index),
        "keeper_candidate_detection_id": keeper_id,
        "iou": metrics.get("iou", 0.0),
        "intersection_area": metrics.get("intersection_area", 0),
        "consumed_mask_area": metrics.get("a_area", 0),
        "keeper_mask_area": metrics.get("b_area", 0),
        "consumed_containment_fraction": metrics.get("a_containment", 0.0),
        "keeper_containment_fraction": metrics.get("b_containment", 0.0),
    }
    keeper.metadata.setdefault("overlap_reconciliation_action", "kept")
    keeper.metadata.setdefault("consumed_candidate_detection_ids", [])
    keeper.metadata.setdefault("overlap_reconciliation_consumed", [])
    if consumed_id not in keeper.metadata["consumed_candidate_detection_ids"]:
        keeper.metadata["consumed_candidate_detection_ids"].append(consumed_id)
    keeper.metadata["overlap_reconciliation_consumed"].append(consumed_entry)
    keeper.metadata["overlap_reconciliation_consumed_count"] = len(
        keeper.metadata["consumed_candidate_detection_ids"]
    )


def _substantial_refined_overlap(
    metrics: dict[str, Any],
    options: RoiRefinementOptions,
) -> bool:
    if metrics["intersection_area"] <= 0:
        return False
    return (
        metrics["iou"] >= options.overlap_iou_threshold
        or metrics["a_containment"] >= options.overlap_containment_threshold
        or metrics["b_containment"] >= options.overlap_containment_threshold
    )


def _refined_overlap_metrics(
    a: DetectionRefinementResult,
    b: DetectionRefinementResult,
) -> dict[str, Any]:
    a_mask = as_binary_mask(a.refined_mask)
    b_mask = as_binary_mask(b.refined_mask)
    a_area = int(np.count_nonzero(a_mask))
    b_area = int(np.count_nonzero(b_mask))
    rect = _intersection_rect(
        _result_mask_frame_rect(a),
        _result_mask_frame_rect(b),
    )
    if rect is None or a_area == 0 or b_area == 0:
        return {
            "a_candidate_detection_id": _detection_identifier(a.candidate_detection),
            "b_candidate_detection_id": _detection_identifier(b.candidate_detection),
            "intersection_area": 0,
            "union_area": a_area + b_area,
            "iou": 0.0,
            "a_area": a_area,
            "b_area": b_area,
            "a_containment": 0.0,
            "b_containment": 0.0,
        }
    a_slice = _mask_slice_for_frame_rect(a_mask, _result_mask_frame_rect(a), rect)
    b_slice = _mask_slice_for_frame_rect(b_mask, _result_mask_frame_rect(b), rect)
    intersection_area = int(np.count_nonzero((a_slice > 0) & (b_slice > 0)))
    union_area = int(a_area + b_area - intersection_area)
    return {
        "a_candidate_detection_id": _detection_identifier(a.candidate_detection),
        "b_candidate_detection_id": _detection_identifier(b.candidate_detection),
        "intersection_area": intersection_area,
        "union_area": union_area,
        "iou": 0.0 if union_area <= 0 else float(intersection_area / union_area),
        "a_area": a_area,
        "b_area": b_area,
        "a_containment": 0.0 if a_area <= 0 else float(intersection_area / a_area),
        "b_containment": 0.0 if b_area <= 0 else float(intersection_area / b_area),
    }


def _result_mask_frame_rect(result: DetectionRefinementResult) -> tuple[int, int, int, int]:
    crop_x, crop_y, _, _ = result.crop_bbox or _detection_crop_bbox(result.candidate_detection)
    mask = as_binary_mask(result.refined_mask)
    return (int(crop_x), int(crop_y), int(mask.shape[1]), int(mask.shape[0]))


def _intersection_rect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _mask_slice_for_frame_rect(
    mask: np.ndarray,
    mask_rect: tuple[int, int, int, int],
    frame_rect: tuple[int, int, int, int],
) -> np.ndarray:
    mask_x, mask_y, _, _ = mask_rect
    frame_x, frame_y, width, height = frame_rect
    local_x = int(frame_x - mask_x)
    local_y = int(frame_y - mask_y)
    return mask[local_y:local_y + height, local_x:local_x + width]


def _reconciliation_sort_key(result: DetectionRefinementResult) -> tuple[int, int, int, str]:
    refined_area = _refined_mask_area(result)
    candidate_area = int(result.candidate_detection.area or 0)
    return (
        -refined_area,
        -candidate_area,
        int(result.candidate_detection.roi_index),
        _detection_identifier(result.candidate_detection),
    )


def _detection_identifier(detection: DetectionRecord) -> str:
    return str(detection.id or f"{detection.frame_id}:{detection.roi_index}")


def _refined_mask_area(result: DetectionRefinementResult) -> int:
    return int(np.count_nonzero(as_binary_mask(result.refined_mask)))


def _non_empty_refinement_results(
    results: Iterable[DetectionRefinementResult],
) -> list[DetectionRefinementResult]:
    kept = []
    for result in results:
        foreground_pixels = _refined_mask_area(result)
        result.metadata["refined_foreground_pixels"] = foreground_pixels
        if foreground_pixels > 0:
            kept.append(result)
    return kept


def refined_storage_candidate_detection_id(result: DetectionRefinementResult) -> str:
    """Return the real candidate id to use for storing a refinement result."""
    parent_id = (
        result.candidate_detection.metadata.get("split_from_candidate_detection_id")
        or result.candidate_detection.metadata.get("residual_parent_candidate_detection_id")
    )
    return str(parent_id or result.candidate_detection.id)


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _prediction_to_mask(prediction: np.ndarray, *, threshold: float) -> np.ndarray:
    array = np.asarray(prediction)
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array.astype(np.uint8) * 255)
    if np.nanmax(array) > 1.0:
        return as_binary_mask(array)
    return np.ascontiguousarray((array >= float(threshold)).astype(np.uint8) * 255)


def _detection_crop_bbox(
    detection: DetectionRecord,
    roi_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int, int]:
    if (
        detection.crop_bbox_x is not None
        and detection.crop_bbox_y is not None
        and detection.crop_bbox_w is not None
        and detection.crop_bbox_h is not None
    ):
        return (
            int(detection.crop_bbox_x),
            int(detection.crop_bbox_y),
            int(detection.crop_bbox_w),
            int(detection.crop_bbox_h),
        )
    if roi_shape is not None and len(roi_shape) >= 2:
        return (int(detection.bbox_x), int(detection.bbox_y), int(roi_shape[1]), int(roi_shape[0]))
    return (int(detection.bbox_x), int(detection.bbox_y), int(detection.bbox_w), int(detection.bbox_h))


def _expanded_crop_bbox(
    crop_bbox: tuple[int, int, int, int],
    touched: dict[str, bool],
    *,
    frame_shape: tuple[int, int],
    expansion_pixels: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = crop_bbox
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    x0 = max(0, x - expansion_pixels) if touched.get("left") else x
    y0 = max(0, y - expansion_pixels) if touched.get("top") else y
    x1 = min(frame_width, x + width + expansion_pixels) if touched.get("right") else x + width
    y1 = min(frame_height, y + height + expansion_pixels) if touched.get("bottom") else y + height
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _expand_roi_and_mask_from_frame(
    frame: np.ndarray,
    *,
    current_roi: np.ndarray,
    current_mask: np.ndarray,
    current_crop_bbox: tuple[int, int, int, int],
    new_crop_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = new_crop_bbox
    new_roi = np.ascontiguousarray(frame[y:y + height, x:x + width])
    new_mask = np.zeros((height, width), dtype=np.uint8)
    old_x, old_y, old_width, old_height = current_crop_bbox
    dest_x = old_x - x
    dest_y = old_y - y
    new_mask[dest_y:dest_y + old_height, dest_x:dest_x + old_width] = as_binary_mask(current_mask)
    if current_roi.shape[:2] != (old_height, old_width):
        raise ValueError("Current ROI shape does not match current crop bbox.")
    return np.ascontiguousarray(new_roi), np.ascontiguousarray(new_mask)


def _mask_bbox_in_frame(mask: np.ndarray, crop_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    binary = as_binary_mask(mask)
    ys, xs = np.nonzero(binary)
    crop_x, crop_y, crop_w, crop_h = crop_bbox
    if xs.size == 0 or ys.size == 0:
        return (int(crop_x), int(crop_y), 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (
        int(crop_x + x0),
        int(crop_y + y0),
        min(int(x1 - x0), int(crop_w)),
        min(int(y1 - y0), int(crop_h)),
    )


def _mask_measurements(mask: np.ndarray) -> tuple[float, float, float, float]:
    binary = as_binary_mask(mask)
    area = float(np.count_nonzero(binary))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area, 0.0, 0.0, 0.0
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if len(largest) >= 5:
        try:
            (_, _), axes, _ = cv2.fitEllipse(largest)
            major, minor = sorted((float(axes[0]), float(axes[1])), reverse=True)
            return area, perimeter, major, minor
        except cv2.error:
            pass
    return area, perimeter, float(max(w, h)), float(min(w, h))


def _gray_stats(roi: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    gray = as_grayscale_array(roi)
    binary = as_binary_mask(mask)
    pixels = gray[binary > 0]
    if pixels.size == 0:
        return 0, 0.0
    return int(np.min(pixels)), float(np.mean(pixels))
