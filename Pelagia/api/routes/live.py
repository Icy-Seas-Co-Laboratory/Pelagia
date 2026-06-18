from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "is_dir": path.is_dir(),
        "size_bytes": None if path.is_dir() else stat.st_size,
        "modified_at": stat.st_mtime,
    }


if APIRouter is not None:
    from ...processing.detection_candidate import live_detection_candidate_wrapper, threshold_frame
    from ...processing.frame_codec import encode_array_payload
    from ...processing.frame_preprocess import preprocess_frame_for_segmentation
    from ...processing.frame_store import retrieve_frame, store_preprocessed_frame
    from ...processing.segmentation_options import (
        flatten_segmentation_options,
        resolve_segmentation_options,
        segment_frame_kwargs,
    )
    from ._common import as_response, detection_summary, frame_summary, get_context, get_repository

    router = APIRouter(prefix="/live", tags=["live"])

    def _option_overrides(local_values: dict[str, Any], option_names: list[str]) -> dict[str, Any]:
        return {name: local_values[name] for name in option_names}

    def _threshold_kwargs(resolved_options: dict[str, dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for group in ("source", "preprocessing", "thresholding"):
            kwargs.update(resolved_options.get(group, {}))
        kwargs.pop("frame_payload_kind", None)
        return kwargs

    def _as_record_dict(record: Any) -> dict[str, Any]:
        if record is None:
            return {}
        if hasattr(record, "__dataclass_fields__"):
            return {
                field: getattr(record, field)
                for field in record.__dataclass_fields__
            }
        return dict(record)

    def _is_live_sandbox_frame(record: Any) -> bool:
        metadata = _as_record_dict(record).get("metadata") or {}
        live_preview = metadata.get("live_preview") or {}
        return bool(live_preview.get("is_sandbox"))

    def _ensure_live_sandbox_frame(repository, frame_record: Any, *, operation: str) -> tuple[dict[str, Any], bool]:
        frame = _as_record_dict(frame_record)
        if _is_live_sandbox_frame(frame):
            return frame, False
        sandbox = repository.create_live_frame_copy(
            str(frame["id"]),
            operation=operation,
        )
        return dict(sandbox), True

    def _delete_unreferenced_kv_payload(context, key: str, *, exclude_frame_id: str | None = None) -> dict[str, Any]:
        repository = context.repository
        if repository is not None and hasattr(repository, "count_frame_payload_references"):
            if repository.count_frame_payload_references(key, exclude_frame_id=exclude_frame_id) > 0:
                return {"key": key, "deleted": False, "referenced": True, "missing": False}
        try:
            context.kvstore.key_delete(str(key))
            return {"key": key, "deleted": True, "referenced": False, "missing": False}
        except KeyError:
            return {"key": key, "deleted": False, "referenced": False, "missing": True}

    @router.get("/files")
    def list_server_files(
        directory: str,
        recursive: bool = False,
        include_hidden: bool = False,
        limit: int = 500,
    ) -> dict:
        if limit < 1:
            raise HTTPException(status_code=422, detail="limit must be >= 1.")

        root = Path(directory).expanduser().resolve()
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"Directory {str(root)!r} was not found.")
        if not root.is_dir():
            raise HTTPException(status_code=422, detail=f"{str(root)!r} is not a directory.")

        iterator = root.rglob("*") if recursive else root.iterdir()
        entries = []
        for path in iterator:
            try:
                if not include_hidden and any(part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                entries.append(_file_entry(path, root))
            except OSError:
                continue
            if len(entries) >= limit:
                break

        entries.sort(key=lambda item: (not item["is_dir"], item["relative_path"].lower()))
        return as_response(
            {
                "directory": str(root),
                "recursive": recursive,
                "include_hidden": include_hidden,
                "limit": limit,
                "count": len(entries),
                "entries": entries,
            }
        )

    @router.post("/preprocess")
    def preprocess_live_frame(
        request: Request,
        frame_id: str,
        encoding: str | None = None,
        flatfield_correction: bool | None = None,
        flatfield_q: float | None = None,
        flatfield_axis: int | None = None,
        flatfield_min_field_value: int | float | None = None,
        flatfield_max_field_value: int | float | None = None,
        apply_mask: bool | None = None,
        crop_enabled: bool | None = None,
        crop_x: int | None = None,
        crop_y: int | None = None,
        crop_w: int | None = None,
        crop_h: int | None = None,
        background_correction: bool | None = None,
        background_min_field_value: int | float | None = None,
        background_max_field_value: int | float | None = None,
        invert_intensity: bool | None = None,
    ) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")

        sandbox_row, sandbox_created = _ensure_live_sandbox_frame(
            repository,
            frame_record,
            operation="preprocess",
        )
        sandbox_frame_id = str(sandbox_row["id"])
        old_key = (
            sandbox_row.get("preprocessed_payload_ref")
            or sandbox_row.get("preprocessed_kvstore_hash")
        )
        try:
            source_frame = retrieve_frame(sandbox_frame_id, context=context, payload_kind="original")
            processed = preprocess_frame_for_segmentation(
                source_frame,
                flatfield_correction=flatfield_correction,
                flatfield_q=flatfield_q,
                flatfield_axis=flatfield_axis,
                flatfield_min_field_value=flatfield_min_field_value,
                flatfield_max_field_value=flatfield_max_field_value,
                apply_mask=apply_mask,
                crop_enabled=crop_enabled,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_w=crop_w,
                crop_h=crop_h,
                background_correction=background_correction,
                background_min_field_value=background_min_field_value,
                background_max_field_value=background_max_field_value,
                invert_intensity=invert_intensity,
                context=context,
            )
            stored_row = store_preprocessed_frame(
                sandbox_frame_id,
                processed,
                context=context,
                encoding=encoding,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        new_key = (
            stored_row.get("preprocessed_payload_ref")
            or stored_row.get("preprocessed_kvstore_hash")
        )
        deleted_old = False
        old_missing = False
        old_referenced = False
        if old_key and old_key != new_key:
            cleanup = _delete_unreferenced_kv_payload(
                context,
                str(old_key),
                exclude_frame_id=sandbox_frame_id,
            )
            deleted_old = bool(cleanup["deleted"])
            old_missing = bool(cleanup["missing"])
            old_referenced = bool(cleanup["referenced"])

        return as_response(
            {
                "status": "stored",
                "saved": True,
                "sandboxed": True,
                "sandbox_created": sandbox_created,
                "source_frame_id": str(_as_record_dict(frame_record)["id"]),
                "sandbox_frame_id": sandbox_frame_id,
                "frame_id": sandbox_frame_id,
                "run_id": _as_record_dict(frame_record).get("run_id"),
                "asset_id": _as_record_dict(frame_record).get("asset_id"),
                "old_preprocessed_key": old_key,
                "new_preprocessed_key": new_key,
                "old_preprocessed_deleted": deleted_old,
                "old_preprocessed_missing": old_missing,
                "old_preprocessed_referenced": old_referenced,
                "preprocessing": processed.metadata,
                "frame": frame_summary(stored_row),
            }
        )

    @router.get("/threshold")
    def threshold_live_frame(
        request: Request,
        frame_id: str,
        threshold: int | float | None = None,
        threshold_method: str | None = None,
        manual_threshold: int | float | None = None,
        thresholding_maximum_value: int | float | None = None,
        bounded_otsu_min_contrast: int | float | None = None,
        bounded_otsu_max_foreground_fraction: float | None = None,
        canny_enabled: bool | None = None,
        canny_low_threshold: int | float | None = None,
        canny_high_threshold: int | float | None = None,
        canny_blur_kernel: int | None = None,
        dilate_kernel_w: int | None = None,
        dilate_kernel_h: int | None = None,
        dilate_iterations: int | None = None,
        erode_kernel_w: int | None = None,
        erode_kernel_h: int | None = None,
        erode_iterations: int | None = None,
        open_kernel_w: int | None = None,
        open_kernel_h: int | None = None,
        open_iterations: int | None = None,
        close_kernel_w: int | None = None,
        close_kernel_h: int | None = None,
        close_iterations: int | None = None,
        fill_holes: bool | None = None,
        remove_small_components: bool | None = None,
        min_component_area: int | float | None = None,
        clear_border: bool | None = None,
        adaptive_block_size: int | None = None,
        adaptive_c: int | float | None = None,
        percentile_background_percentile: int | float | None = None,
        percentile_min_contrast: int | float | None = None,
        hysteresis_low_threshold: int | float | None = None,
        hysteresis_high_threshold: int | float | None = None,
        hysteresis_connectivity: int | None = None,
        sobel_percentile: int | float | None = None,
        sobel_threshold: int | float | None = None,
        sobel_kernel_size: int | None = None,
        frame_payload_kind: str = "original",
        apply_preprocessing: bool | None = None,
        flatfield_correction: bool | None = None,
        flatfield_q: float | None = None,
        flatfield_axis: int | None = None,
        flatfield_min_field_value: int | float | None = None,
        flatfield_max_field_value: int | float | None = None,
        apply_mask: bool | None = None,
        crop_enabled: bool | None = None,
        crop_x: int | None = None,
        crop_y: int | None = None,
        crop_w: int | None = None,
        crop_h: int | None = None,
        background_correction: bool | None = None,
        background_min_field_value: int | float | None = None,
        background_max_field_value: int | float | None = None,
        invert_intensity: bool | None = None,
        include_mask_payload: bool = False,
        mask_encoding: str = "png",
    ) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
        sandbox_row, sandbox_created = _ensure_live_sandbox_frame(
            repository,
            frame_record,
            operation="threshold",
        )
        sandbox_frame_id = str(sandbox_row["id"])

        option_names = [
            "threshold",
            "threshold_method",
            "manual_threshold",
            "thresholding_maximum_value",
            "bounded_otsu_min_contrast",
            "bounded_otsu_max_foreground_fraction",
            "canny_enabled",
            "canny_low_threshold",
            "canny_high_threshold",
            "canny_blur_kernel",
            "adaptive_block_size",
            "adaptive_c",
            "percentile_background_percentile",
            "percentile_min_contrast",
            "hysteresis_low_threshold",
            "hysteresis_high_threshold",
            "hysteresis_connectivity",
            "sobel_percentile",
            "sobel_threshold",
            "sobel_kernel_size",
            "frame_payload_kind",
            "apply_preprocessing",
            "flatfield_correction",
            "flatfield_q",
            "flatfield_axis",
            "flatfield_min_field_value",
            "flatfield_max_field_value",
            "apply_mask",
            "crop_enabled",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
            "background_correction",
            "background_min_field_value",
            "background_max_field_value",
            "invert_intensity",
        ]
        try:
            resolved_options = resolve_segmentation_options(
                _option_overrides(locals(), option_names),
                context.config.processing,
            )
            flat_options = flatten_segmentation_options(resolved_options)
            frame = retrieve_frame(
                sandbox_frame_id,
                context=context,
                payload_kind=flat_options["frame_payload_kind"],
            )
            result = threshold_frame(
                frame,
                **_threshold_kwargs(resolved_options),
                context=context,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        mask = result.threshold_mask
        foreground_pixels = int((mask > 0).sum())
        mask_payload: dict[str, Any] = {}
        if include_mask_payload:
            try:
                payload, encoding, payload_format = encode_array_payload(mask, mask_encoding)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            mask_payload = {
                "mask_payload_base64": base64.b64encode(payload).decode("ascii"),
                "mask_payload_bytes": len(payload),
                "mask_encoding": encoding,
                "mask_format": payload_format,
            }

        return as_response(
            {
                "frame_id": frame_id,
                "source_frame_id": str(_as_record_dict(frame_record)["id"]),
                "sandbox_frame_id": sandbox_frame_id,
                "sandboxed": True,
                "sandbox_created": sandbox_created,
                "run_id": _as_record_dict(frame_record).get("run_id"),
                "asset_id": _as_record_dict(frame_record).get("asset_id"),
                "saved": False,
                "frame_payload_kind": flat_options["frame_payload_kind"],
                "apply_preprocessing": flat_options["apply_preprocessing"],
                "resolved_options": resolved_options,
                "processed_frame_shape": result.metadata.get("processed_frame_shape"),
                "stage_counts": result.metadata.get("stage_counts"),
                "stage_durations_ms": result.metadata.get("stage_durations_ms"),
                "mask": {
                    "shape": list(mask.shape),
                    "dtype": str(mask.dtype),
                    "foreground_pixels": foreground_pixels,
                    "foreground_fraction": (
                        foreground_pixels / float(mask.size) if mask.size else 0.0
                    ),
                    **mask_payload,
                },
            }
        )

    @router.get("/detection-candidate")
    @router.get("/detection_candidate", include_in_schema=False)
    def detect_live_candidates(
        request: Request,
        frame_id: str,
        threshold: int | float | None = None,
        threshold_method: str | None = None,
        manual_threshold: int | float | None = None,
        thresholding_maximum_value: int | float | None = None,
        bounded_otsu_min_contrast: int | float | None = None,
        bounded_otsu_max_foreground_fraction: float | None = None,
        canny_enabled: bool | None = None,
        canny_low_threshold: int | float | None = None,
        canny_high_threshold: int | float | None = None,
        canny_blur_kernel: int | None = None,
        dilate_kernel_w: int | None = None,
        dilate_kernel_h: int | None = None,
        dilate_iterations: int | None = None,
        erode_kernel_w: int | None = None,
        erode_kernel_h: int | None = None,
        erode_iterations: int | None = None,
        open_kernel_w: int | None = None,
        open_kernel_h: int | None = None,
        open_iterations: int | None = None,
        close_kernel_w: int | None = None,
        close_kernel_h: int | None = None,
        close_iterations: int | None = None,
        fill_holes: bool | None = None,
        remove_small_components: bool | None = None,
        min_component_area: int | float | None = None,
        clear_border: bool | None = None,
        adaptive_block_size: int | None = None,
        adaptive_c: int | float | None = None,
        percentile_background_percentile: int | float | None = None,
        percentile_min_contrast: int | float | None = None,
        hysteresis_low_threshold: int | float | None = None,
        hysteresis_high_threshold: int | float | None = None,
        hysteresis_connectivity: int | None = None,
        sobel_percentile: int | float | None = None,
        sobel_threshold: int | float | None = None,
        sobel_kernel_size: int | None = None,
        frame_payload_kind: str = "original",
        apply_preprocessing: bool | None = None,
        flatfield_correction: bool | None = None,
        flatfield_q: float | None = None,
        flatfield_axis: int | None = None,
        flatfield_min_field_value: int | float | None = None,
        flatfield_max_field_value: int | float | None = None,
        apply_mask: bool | None = None,
        crop_enabled: bool | None = None,
        crop_x: int | None = None,
        crop_y: int | None = None,
        crop_w: int | None = None,
        crop_h: int | None = None,
        background_correction: bool | None = None,
        background_min_field_value: int | float | None = None,
        background_max_field_value: int | float | None = None,
        invert_intensity: bool | None = None,
        mask_augmentation_enabled: bool | None = None,
        mask_augmentation_steps: list[str] | None = None,
        roi_assembly_method: str | None = None,
        roi_assembly_connectivity: int | None = None,
        min_area: int | float | None = None,
        max_area: int | float | None = None,
        min_perimeter: int | float | None = None,
        max_perimeter: int | float | None = None,
        min_width: int | float | None = None,
        max_width: int | float | None = None,
        min_height: int | float | None = None,
        max_height: int | float | None = None,
        min_width_plus_height: int | float | None = None,
        max_width_plus_height: int | float | None = None,
        padding: int | None = None,
        roi_encoding: str | None = "png",
        zstd_min_bytes: int | None = None,
        store_roi_payload_min_area: int | float | None = None,
        store_roi_payload_min_width: int | float | None = None,
        store_roi_payload_min_height: int | float | None = None,
        store_roi_payload_min_width_plus_height: int | float | None = None,
        always_store_mask: bool | None = None,
        include_detection_payloads: bool = False,
        max_detections: int | None = 500,
    ) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
        sandbox_row, sandbox_created = _ensure_live_sandbox_frame(
            repository,
            frame_record,
            operation="detection_candidate",
        )
        sandbox_frame_id = str(sandbox_row["id"])

        option_names = [
            "threshold",
            "threshold_method",
            "manual_threshold",
            "thresholding_maximum_value",
            "bounded_otsu_min_contrast",
            "bounded_otsu_max_foreground_fraction",
            "canny_enabled",
            "canny_low_threshold",
            "canny_high_threshold",
            "canny_blur_kernel",
            "dilate_kernel_w",
            "dilate_kernel_h",
            "dilate_iterations",
            "erode_kernel_w",
            "erode_kernel_h",
            "erode_iterations",
            "open_kernel_w",
            "open_kernel_h",
            "open_iterations",
            "close_kernel_w",
            "close_kernel_h",
            "close_iterations",
            "fill_holes",
            "remove_small_components",
            "min_component_area",
            "clear_border",
            "adaptive_block_size",
            "adaptive_c",
            "percentile_background_percentile",
            "percentile_min_contrast",
            "hysteresis_low_threshold",
            "hysteresis_high_threshold",
            "hysteresis_connectivity",
            "sobel_percentile",
            "sobel_threshold",
            "sobel_kernel_size",
            "frame_payload_kind",
            "apply_preprocessing",
            "flatfield_correction",
            "flatfield_q",
            "flatfield_axis",
            "flatfield_min_field_value",
            "flatfield_max_field_value",
            "apply_mask",
            "crop_enabled",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
            "background_correction",
            "background_min_field_value",
            "background_max_field_value",
            "invert_intensity",
            "mask_augmentation_enabled",
            "mask_augmentation_steps",
            "roi_assembly_method",
            "roi_assembly_connectivity",
            "min_area",
            "max_area",
            "min_perimeter",
            "max_perimeter",
            "min_width",
            "max_width",
            "min_height",
            "max_height",
            "min_width_plus_height",
            "max_width_plus_height",
            "padding",
            "roi_encoding",
            "zstd_min_bytes",
            "store_roi_payload_min_area",
            "store_roi_payload_min_width",
            "store_roi_payload_min_height",
            "store_roi_payload_min_width_plus_height",
            "always_store_mask",
        ]
        local_values = locals()
        overrides = {name: local_values[name] for name in option_names}
        try:
            resolved_options = resolve_segmentation_options(
                overrides,
                context.config.processing,
            )
            flat_options = flatten_segmentation_options(resolved_options)
            detections = live_detection_candidate_wrapper(
                sandbox_frame_id,
                frame_payload_kind=flat_options["frame_payload_kind"],
                encode_payloads=include_detection_payloads,
                max_detections=max_detections,
                **segment_frame_kwargs(resolved_options),
                context=context,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        detection_rows = [detection_summary(detection) for detection in detections]
        metadata = getattr(detections[0], "metadata", {}) if detections else {}
        stage_counts = dict(metadata.get("stage_counts") or {})
        stage_counts.setdefault("recorded_detection_count", len(detections))
        stage_durations_ms = dict(metadata.get("stage_durations_ms") or {})

        response = {
            "frame_id": frame_id,
            "source_frame_id": str(_as_record_dict(frame_record)["id"]),
            "sandbox_frame_id": sandbox_frame_id,
            "sandboxed": True,
            "sandbox_created": sandbox_created,
            "run_id": _as_record_dict(frame_record).get("run_id"),
            "asset_id": _as_record_dict(frame_record).get("asset_id"),
            "saved": False,
            "frame_payload_kind": flat_options["frame_payload_kind"],
            "apply_preprocessing": flat_options["apply_preprocessing"],
            "resolved_options": resolved_options,
            "bbox_coordinate_space": metadata.get("bbox_coordinate_space"),
            "processed_frame_shape": metadata.get("processed_frame_shape"),
            "stage_counts": stage_counts,
            "stage_durations_ms": stage_durations_ms,
            "payloads_encoded": include_detection_payloads,
            "max_detections": max_detections,
            "candidate_limit_applied": metadata.get("candidate_limit_applied"),
            "candidate_detection_count": len(detections),
            "detection_count": len(detections),
            "candidate_detections": detection_rows,
            "detections": detection_rows,
        }

        return as_response(response)

    @router.get("/sandbox")
    def list_live_sandbox_frames(
        request: Request,
        source_frame_id: str | None = None,
        operation: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        if limit < 1:
            raise HTTPException(status_code=422, detail="limit must be >= 1.")
        repository = get_repository(request)
        rows = repository.list_live_frame_copies(
            source_frame_id=source_frame_id,
            operation=operation,
            limit=limit,
            offset=offset,
        )
        return as_response(
            {
                "sandbox_frames": [frame_summary(row) for row in rows],
                "limit": limit,
                "offset": max(0, int(offset)),
                "count": len(rows),
            }
        )

    @router.delete("/sandbox/{sandbox_frame_id}")
    def delete_live_sandbox_frame(request: Request, sandbox_frame_id: str) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        result = repository.delete_live_frame_copy(sandbox_frame_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Live sandbox frame {sandbox_frame_id!r} was not found.",
            )
        deleted_kvstore_keys = []
        for key in result.get("unreferenced_kvstore_keys", []):
            deleted_kvstore_keys.append(
                _delete_unreferenced_kv_payload(
                    context,
                    str(key),
                )
            )
        return as_response(
            {
                "status": "deleted",
                "sandbox_frame_id": sandbox_frame_id,
                "frame": frame_summary(result["frame"]),
                "generated_kvstore_keys": result.get("generated_kvstore_keys", []),
                "deleted_kvstore_keys": deleted_kvstore_keys,
            }
        )
else:
    router = None
