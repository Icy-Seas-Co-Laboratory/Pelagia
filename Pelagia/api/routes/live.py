from __future__ import annotations

import base64
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Query, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore

if APIRouter is not None:
    from ..auth import scoped_project_id
    from ...processing.detection_candidate import live_detection_candidate_wrapper, threshold_frame
    from ...processing.frame_codec import encode_array_payload
    from ...processing.frame_correction import build_background_payload_for_frames
    from ...processing.frame_preprocess import preprocess_frame_for_segmentation
    from ...processing.frame_store import retrieve_frame, store_preprocessed_frame
    from ...processing.mask_augmentation import augment_mask
    from ...processing.segmentation_options import (
        flatten_segmentation_options,
        resolve_segmentation_options,
        segment_frame_kwargs,
    )
    from ...services.project_settings import resolve_project_storage_settings
    from ._common import as_response, detection_summary, frame_summary, get_context, get_repository

    router = APIRouter(prefix="/live", tags=["live"])

    def _split_mask_step_values(values: list[Any]) -> list[str]:
        steps: list[str] = []
        for value in values:
            if value is None:
                continue
            for step in str(value).split(","):
                normalized = step.strip()
                if normalized:
                    steps.append(normalized)
        return steps

    def _parse_square_kernel_alias(value: Any) -> tuple[int, int]:
        text = str(value).strip().lower().replace("×", "x")
        if not text:
            raise ValueError("Kernel size aliases cannot be empty.")
        for separator in ("x", ","):
            if separator in text:
                width_text, height_text = [part.strip() for part in text.split(separator, 1)]
                return int(float(width_text)), int(float(height_text))
        size = int(float(text))
        return size, size

    def _option_overrides(request: Request, local_values: dict[str, Any], option_names: list[str]) -> dict[str, Any]:
        overrides = {name: local_values[name] for name in option_names}
        query = request.query_params
        mask_steps = _split_mask_step_values(query.getlist("mask_augmentation_steps"))
        mask_steps.extend(_split_mask_step_values(query.getlist("mask_augmentation_steps[]")))
        if mask_steps:
            overrides["mask_augmentation_steps"] = mask_steps

        for prefix in ("dilate", "erode", "open", "close"):
            width_name = f"{prefix}_kernel_w"
            height_name = f"{prefix}_kernel_h"
            for alias in (f"{prefix}_kernel", f"{prefix}_kernel_size"):
                values = query.getlist(alias)
                if not values:
                    continue
                width, height = _parse_square_kernel_alias(values[-1])
                if overrides.get(width_name) is None:
                    overrides[width_name] = width
                if overrides.get(height_name) is None:
                    overrides[height_name] = height
                break
        return overrides

    def _threshold_kwargs(resolved_options: dict[str, dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for group in ("source", "preprocessing", "thresholding"):
            kwargs.update(resolved_options.get(group, {}))
        kwargs.pop("frame_payload_kind", None)
        return kwargs

    def _augment_mask_from_options(mask, resolved_options: dict[str, dict[str, Any]]):
        options = resolved_options.get("mask_augmentation", {})
        return augment_mask(
            mask,
            enabled=bool(options.get("mask_augmentation_enabled", False)),
            steps=options.get("mask_augmentation_steps", []),
            dilate_kernel_size=(
                int(options.get("dilate_kernel_w", 3)),
                int(options.get("dilate_kernel_h", 3)),
            ),
            dilate_iterations=int(options.get("dilate_iterations", 1)),
            erode_kernel_size=(
                int(options.get("erode_kernel_w", 3)),
                int(options.get("erode_kernel_h", 3)),
            ),
            erode_iterations=int(options.get("erode_iterations", 1)),
            open_kernel_size=(
                int(options.get("open_kernel_w", 3)),
                int(options.get("open_kernel_h", 3)),
            ),
            open_iterations=int(options.get("open_iterations", 1)),
            close_kernel_size=(
                int(options.get("close_kernel_w", 3)),
                int(options.get("close_kernel_h", 3)),
            ),
            close_iterations=int(options.get("close_iterations", 1)),
            fill_holes=bool(options.get("fill_holes", False)),
            remove_small_components=bool(options.get("remove_small_components", False)),
            min_component_area=float(options.get("min_component_area", 1)),
            clear_border=bool(options.get("clear_border", False)),
        )

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

    def _has_preprocessed_payload(record: Any) -> bool:
        frame = _as_record_dict(record)
        return bool(frame.get("preprocessed_payload_ref") or frame.get("preprocessed_kvstore_hash"))

    def _require_live_payload(frame: Any, *, payload_kind: str, endpoint: str) -> None:
        if payload_kind not in {"preprocessed", "processed", "corrected"}:
            return
        if _has_preprocessed_payload(frame):
            return
        frame_id = _as_record_dict(frame).get("id")
        raise HTTPException(
            status_code=422,
            detail=(
                f"{endpoint} requested {payload_kind!r} frame payloads, but frame {frame_id!r} "
                "does not have a live preprocessed payload. Run /live/preprocess first or use frame_payload_kind='original'."
            ),
        )

    def _ensure_live_sandbox_frame(
        repository,
        frame_record: Any,
        *,
        operation: str,
        project_id: str,
    ) -> tuple[dict[str, Any], bool]:
        frame = _as_record_dict(frame_record)
        if _is_live_sandbox_frame(frame):
            return frame, False
        sandbox = repository.create_live_frame_copy(
            str(frame["id"]),
            operation=operation,
            project_id=project_id,
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

    def _merge_frame_id_lists(*values: list[str] | None) -> list[str]:
        frame_ids: list[str] = []
        for value in values:
            if not value:
                continue
            frame_ids.extend(str(frame_id) for frame_id in value)
        return list(dict.fromkeys(frame_ids))

    def _has_background_source_selection(
        *,
        asset_id: str | None,
        frame_ids: list[str],
        start_frame: int | None,
        end_frame: int | None,
        limit: int | None,
    ) -> bool:
        return bool(asset_id or frame_ids or start_frame is not None or end_frame is not None or limit is not None)

    def _resolve_live_background_frame_ids(
        request: Request,
        *,
        target_frame_record: Any,
        asset_id: str | None,
        frame_ids: list[str],
        start_frame: int | None,
        end_frame: int | None,
        limit: int | None,
    ) -> tuple[str | None, list[str]]:
        repository = get_repository(request)
        project_id = scoped_project_id(request)
        target = _as_record_dict(target_frame_record)
        resolved_asset_id = asset_id
        resolved_frame_ids = list(frame_ids)
        if not resolved_frame_ids:
            resolved_asset_id = resolved_asset_id or target.get("asset_id")
            if resolved_asset_id is None:
                raise HTTPException(
                    status_code=422,
                    detail="Background generation requires background_asset_id/asset_id or background_frame_ids/frame_ids.",
                )
            frames = repository.list_frames(
                resolved_asset_id,
                project_id=project_id,
                start_frame=start_frame,
                end_frame=end_frame,
                limit=limit,
            )
            resolved_frame_ids = [str(frame["id"]) for frame in frames]

        if not resolved_frame_ids:
            raise HTTPException(status_code=422, detail="No background source frames were selected.")

        for background_frame_id in resolved_frame_ids:
            frame_record = repository.get_frame_record(background_frame_id, project_id=project_id)
            if frame_record is None:
                raise HTTPException(status_code=404, detail=f"Frame {background_frame_id!r} was not found.")
            frame = _as_record_dict(frame_record)
            if resolved_asset_id is None:
                resolved_asset_id = frame.get("asset_id")
            elif frame.get("asset_id") != resolved_asset_id:
                raise HTTPException(
                    status_code=422,
                    detail="Background generation may only use frames from one asset.",
                )
        return resolved_asset_id, resolved_frame_ids

    def _assign_generated_background_to_frame(
        repository,
        frame_id: str,
        *,
        project_id: str,
        result: dict[str, Any],
        source_frame_ids: list[str],
        payload_kind: str,
    ) -> list[dict[str, Any]]:
        payload_ref = str(result["background_payload_ref"])
        metadata = {
            "frame_variant": "background",
            "background_method": result.get("background_method", "mean"),
            "background_source_payload_kind": payload_kind,
            "background_source_frame_ids": source_frame_ids,
            "background_source_frame_count": len(source_frame_ids),
            "kvstore_key": payload_ref,
            "kvstore_hash": payload_ref,
            "kvstore_encoding": result.get("background_payload_encoding"),
            "kvstore_format": result.get("background_payload_format"),
            "dtype": result.get("background_payload_dtype"),
            "shape": result.get("background_payload_shape") or [],
            "live_preprocess": True,
        }
        return repository.update_frame_background_payloads(
            [frame_id],
            project_id=project_id,
            kvstore_hash=payload_ref,
            payload_ref=payload_ref,
            payload_encoding=str(result["background_payload_encoding"]),
            payload_format=str(result["background_payload_format"]),
            payload_dtype=str(result["background_payload_dtype"]),
            payload_shape=list(result["background_payload_shape"] or []),
            metadata=metadata,
        )

    @router.post("/preprocess")
    def preprocess_live_frame(
        request: Request,
        frame_id: str,
        encoding: str | None = None,
        quality: int | None = None,
        asset_id: str | None = None,
        frame_ids: list[str] | None = Query(default=None),
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = None,
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
        background_asset_id: str | None = None,
        background_frame_ids: list[str] | None = Query(default=None),
        background_start_frame: int | None = None,
        background_end_frame: int | None = None,
        background_limit: int | None = None,
        background_payload_kind: str = "original",
        background_encoding: str = "zstd",
        background_quality: int | None = None,
        invert_intensity: bool | None = None,
    ) -> dict:
        project_id = scoped_project_id(request)
        context = get_context(request).for_project(project_id)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id, project_id=project_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")

        sandbox_row, sandbox_created = _ensure_live_sandbox_frame(
            repository,
            frame_record,
            operation="preprocess",
            project_id=project_id,
        )
        sandbox_frame_id = str(sandbox_row["id"])
        old_key = (
            sandbox_row.get("preprocessed_payload_ref")
            or sandbox_row.get("preprocessed_kvstore_hash")
        )
        merged_background_frame_ids = _merge_frame_id_lists(frame_ids, background_frame_ids)
        resolved_background_asset_id = background_asset_id or asset_id
        resolved_background_start_frame = background_start_frame if background_start_frame is not None else start_frame
        resolved_background_end_frame = background_end_frame if background_end_frame is not None else end_frame
        resolved_background_limit = background_limit if background_limit is not None else limit
        background_generation = None
        background_source_selected = _has_background_source_selection(
            asset_id=resolved_background_asset_id,
            frame_ids=merged_background_frame_ids,
            start_frame=resolved_background_start_frame,
            end_frame=resolved_background_end_frame,
            limit=resolved_background_limit,
        )
        try:
            if background_source_selected:
                source_asset_id, background_source_frame_ids = _resolve_live_background_frame_ids(
                    request,
                    target_frame_record=frame_record,
                    asset_id=resolved_background_asset_id,
                    frame_ids=merged_background_frame_ids,
                    start_frame=resolved_background_start_frame,
                    end_frame=resolved_background_end_frame,
                    limit=resolved_background_limit,
                )
                background_generation = build_background_payload_for_frames(
                    background_source_frame_ids,
                    context=context,
                    payload_kind=background_payload_kind,
                    encoding=background_encoding,
                    quality=background_quality,
                )
                _assign_generated_background_to_frame(
                    repository,
                    sandbox_frame_id,
                    project_id=project_id,
                    result=background_generation,
                    source_frame_ids=background_source_frame_ids,
                    payload_kind=background_payload_kind,
                )
                background_generation = {
                    **background_generation,
                    "asset_id": source_asset_id,
                    "start_frame": resolved_background_start_frame,
                    "end_frame": resolved_background_end_frame,
                    "limit": resolved_background_limit,
                    "payload_kind": background_payload_kind,
                    "encoding": background_encoding,
                    "quality": background_quality,
                }
                if background_correction is None:
                    background_correction = True
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
                quality=quality,
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
                "background_generation": background_generation,
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
        mask_augmentation_enabled: bool | None = None,
        mask_augmentation_steps: list[str] | None = Query(default=None),
        include_mask_payload: bool = False,
        mask_encoding: str = "png",
    ) -> dict:
        project_id = scoped_project_id(request)
        context = get_context(request).for_project(project_id)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id, project_id=project_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
        sandbox_row, sandbox_created = _ensure_live_sandbox_frame(
            repository,
            frame_record,
            operation="threshold",
            project_id=project_id,
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
        ]
        try:
            resolved_options = resolve_segmentation_options(
                _option_overrides(request, locals(), option_names),
                context.config.processing,
            )
            flat_options = flatten_segmentation_options(resolved_options)
            _require_live_payload(
                sandbox_row,
                payload_kind=flat_options["frame_payload_kind"],
                endpoint="/live/threshold",
            )
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
            augmented_mask = _augment_mask_from_options(
                result.threshold_mask,
                resolved_options,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        threshold_mask = result.threshold_mask
        mask = augmented_mask
        threshold_foreground_pixels = int((threshold_mask > 0).sum())
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
                "stage_counts": {
                    **dict(result.metadata.get("stage_counts") or {}),
                    "threshold_foreground_pixels": threshold_foreground_pixels,
                    "augmented_foreground_pixels": foreground_pixels,
                },
                "stage_durations_ms": result.metadata.get("stage_durations_ms"),
                "mask": {
                    "shape": list(mask.shape),
                    "dtype": str(mask.dtype),
                    "kind": "augmented",
                    "threshold_foreground_pixels": threshold_foreground_pixels,
                    "foreground_pixels": foreground_pixels,
                    "augmented_foreground_pixels": foreground_pixels,
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
        mask_augmentation_steps: list[str] | None = Query(default=None),
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
        roi_encoding: str | None = None,
        zstd_min_bytes: int | None = None,
        store_roi_payload_min_area: int | float | None = None,
        store_roi_payload_min_width: int | float | None = None,
        store_roi_payload_min_height: int | float | None = None,
        store_roi_payload_min_width_plus_height: int | float | None = None,
        always_store_mask: bool | None = None,
        include_detection_payloads: bool = False,
        max_detections: int | None = 500,
    ) -> dict:
        project_id = scoped_project_id(request)
        context = get_context(request).for_project(project_id)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id, project_id=project_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
        sandbox_row, sandbox_created = _ensure_live_sandbox_frame(
            repository,
            frame_record,
            operation="detection_candidate",
            project_id=project_id,
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
        try:
            local_values = locals()
            overrides = _option_overrides(request, local_values, option_names)
            if overrides.get("roi_encoding") is None:
                overrides["roi_encoding"] = resolve_project_storage_settings(
                    context,
                    project_id,
                ).roi_encoding
            resolved_options = resolve_segmentation_options(
                overrides,
                context.config.processing,
            )
            flat_options = flatten_segmentation_options(resolved_options)
            _require_live_payload(
                sandbox_row,
                payload_kind=flat_options["frame_payload_kind"],
                endpoint="/live/detection-candidate",
            )
            detections = live_detection_candidate_wrapper(
                sandbox_frame_id,
                frame_payload_kind=flat_options["frame_payload_kind"],
                encode_payloads=include_detection_payloads,
                max_detections=max_detections,
                **segment_frame_kwargs(resolved_options),
                context=context,
            )
        except HTTPException:
            raise
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

else:
    router = None
