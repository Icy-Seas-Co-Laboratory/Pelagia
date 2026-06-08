from __future__ import annotations

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
    from ...processing.detection_candidate import live_segment_wrapper
    from ...processing.frame_preprocess import preprocess_frame_for_segmentation
    from ...processing.frame_store import retrieve_frame, store_preprocessed_frame
    from ._common import as_response, detection_summary, frame_summary, get_context, get_repository

    router = APIRouter(prefix="/live", tags=["live"])

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
        apply_mask: bool | None = None,
        crop_enabled: bool | None = None,
        crop_x: int | None = None,
        crop_y: int | None = None,
        crop_w: int | None = None,
        crop_h: int | None = None,
        background_correction: bool | None = None,
        background_percentile: int | float | None = None,
        invert_intensity: bool | None = None,
    ) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")

        old_key = (
            getattr(frame_record, "preprocessed_payload_ref", None)
            or getattr(frame_record, "preprocessed_kvstore_hash", None)
        )
        try:
            source_frame = retrieve_frame(frame_id, context=context, payload_kind="original")
            processed = preprocess_frame_for_segmentation(
                source_frame,
                flatfield_correction=flatfield_correction,
                flatfield_q=flatfield_q,
                flatfield_axis=flatfield_axis,
                apply_mask=apply_mask,
                crop_enabled=crop_enabled,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_w=crop_w,
                crop_h=crop_h,
                background_correction=background_correction,
                background_percentile=background_percentile,
                invert_intensity=invert_intensity,
                context=context,
            )
            stored_row = store_preprocessed_frame(
                frame_id,
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
        if old_key and old_key != new_key:
            try:
                context.kvstore.key_delete(str(old_key))
                deleted_old = True
            except KeyError:
                old_missing = True
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Preprocessed frame was stored, but old KV payload deletion failed: {exc}",
                ) from exc

        return as_response(
            {
                "status": "stored",
                "saved": True,
                "frame_id": frame_id,
                "run_id": frame_record.run_id,
                "asset_id": frame_record.asset_id,
                "old_preprocessed_key": old_key,
                "new_preprocessed_key": new_key,
                "old_preprocessed_deleted": deleted_old,
                "old_preprocessed_missing": old_missing,
                "preprocessing": processed.metadata,
                "frame": frame_summary(stored_row),
            }
        )

    @router.get("/segmentation")
    def segment_live_frame(
        request: Request,
        frame_id: str,
        threshold: int | float | None = None,
        frame_payload_kind: str = "original",
        apply_preprocessing: bool | None = None,
        flatfield_correction: bool | None = None,
        flatfield_q: float | None = None,
        flatfield_axis: int | None = None,
        apply_mask: bool | None = None,
        crop_enabled: bool | None = None,
        crop_x: int | None = None,
        crop_y: int | None = None,
        crop_w: int | None = None,
        crop_h: int | None = None,
        background_correction: bool | None = None,
        background_percentile: int | float | None = None,
        invert_intensity: bool | None = None,
        min_perimeter: int | float | None = None,
        max_perimeter: int | float | None = None,
        padding: int | None = None,
    ) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")

        defaults = context.config.processing.segmentation
        flatfield_defaults = context.config.processing.flatfield
        preprocessing_defaults = context.config.processing.preprocessing
        try:
            detections = live_segment_wrapper(
                frame_id,
                threshold=threshold,
                frame_payload_kind=frame_payload_kind,
                apply_preprocessing=apply_preprocessing,
                flatfield_correction=(
                    flatfield_defaults.flatfield_correction
                    if flatfield_correction is None
                    else flatfield_correction
                ),
                flatfield_q=flatfield_defaults.flatfield_q if flatfield_q is None else flatfield_q,
                flatfield_axis=flatfield_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis,
                apply_mask=preprocessing_defaults.apply_mask if apply_mask is None else apply_mask,
                crop_enabled=(
                    preprocessing_defaults.crop_enabled
                    if crop_enabled is None
                    else crop_enabled
                ),
                crop_x=preprocessing_defaults.crop_x if crop_x is None else crop_x,
                crop_y=preprocessing_defaults.crop_y if crop_y is None else crop_y,
                crop_w=preprocessing_defaults.crop_w if crop_w is None else crop_w,
                crop_h=preprocessing_defaults.crop_h if crop_h is None else crop_h,
                background_correction=(
                    preprocessing_defaults.background_correction
                    if background_correction is None
                    else background_correction
                ),
                background_percentile=(
                    preprocessing_defaults.background_percentile
                    if background_percentile is None
                    else background_percentile
                ),
                invert_intensity=(
                    preprocessing_defaults.invert_intensity
                    if invert_intensity is None
                    else invert_intensity
                ),
                min_perimeter=defaults.min_perimeter if min_perimeter is None else min_perimeter,
                max_perimeter=defaults.max_perimeter if max_perimeter is None else max_perimeter,
                padding=defaults.padding if padding is None else padding,
                context=context,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        detection_rows = [detection_summary(detection) for detection in detections]

        response = {
            "frame_id": frame_id,
            "run_id": getattr(frame_record, "run_id", None),
            "asset_id": getattr(frame_record, "asset_id", None),
            "saved": False,
            "frame_payload_kind": frame_payload_kind,
            "apply_preprocessing": (
                frame_payload_kind in {"original", "raw"}
                if apply_preprocessing is None
                else apply_preprocessing
            ),
            "apply_mask": preprocessing_defaults.apply_mask if apply_mask is None else apply_mask,
            "crop_enabled": (
                preprocessing_defaults.crop_enabled
                if crop_enabled is None
                else crop_enabled
            ),
            "crop_x": preprocessing_defaults.crop_x if crop_x is None else crop_x,
            "crop_y": preprocessing_defaults.crop_y if crop_y is None else crop_y,
            "crop_w": preprocessing_defaults.crop_w if crop_w is None else crop_w,
            "crop_h": preprocessing_defaults.crop_h if crop_h is None else crop_h,
            "background_correction": (
                preprocessing_defaults.background_correction
                if background_correction is None
                else background_correction
            ),
            "background_percentile": (
                preprocessing_defaults.background_percentile
                if background_percentile is None
                else background_percentile
            ),
            "intensity_inverted": (
                preprocessing_defaults.invert_intensity
                if invert_intensity is None
                else invert_intensity
            ),
            "detection_count": len(detections),
            "detections": detection_rows,
        }

        return as_response(response)
else:
    router = None
