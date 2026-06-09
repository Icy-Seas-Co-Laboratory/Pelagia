from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request, Response
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    import numpy as np

    from ...processing.frame_correction import flatfield_global_correction_for_framedata
    from ...processing.frame_store import retrieve_frame
    from ._common import as_response, detection_summary, frame_summary, get_context, get_repository, page_metadata
    from ._images import encode_image, preview_image, scale_image

    router = APIRouter(prefix="/assets", tags=["assets"])

    @router.get("")
    def list_assets(
        request: Request,
        run_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        path: str | None = None,
        checksum: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        media_count: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        return {
            "assets": as_response(
                get_repository(request).list_assets(
                    run_id=run_id,
                    collection=collection,
                    kind=kind,
                    filename=filename,
                    path=path,
                    checksum=checksum,
                    min_size_bytes=min_size_bytes,
                    max_size_bytes=max_size_bytes,
                    media_count=media_count,
                    limit=limit,
                    offset=offset,
                )
            )
        }

    @router.get("/detections")
    def list_asset_detection_stats(
        request: Request,
        run_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        min_detection_count: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        stats = get_repository(request).list_asset_detection_stats(
            run_id=run_id,
            collection=collection,
            kind=kind,
            filename=filename,
            min_detection_count=min_detection_count,
            limit=limit,
            offset=offset,
        )
        return as_response(stats)

    @router.get("/processing-state")
    def list_asset_processing_state(
        request: Request,
        run_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        preprocessing_state: str | None = None,
        detection_state: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        stats = get_repository(request).list_asset_processing_state(
            run_id=run_id,
            collection=collection,
            kind=kind,
            filename=filename,
            preprocessing_state=preprocessing_state,
            detection_state=detection_state,
            limit=limit,
            offset=offset,
        )
        return as_response(
            {
                **stats,
                "page": page_metadata(limit=limit, offset=offset, count=len(stats.get("assets", []))),
            }
        )

    @router.get("/{asset_id}")
    def get_asset(request: Request, asset_id: str) -> dict:
        repository = get_repository(request)
        asset = repository.get_asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} was not found.")
        asset = dict(asset)
        asset["frame_count"] = repository.count_frames(asset_id)
        return {"asset": as_response(asset)}

    @router.get("/{asset_id}/frames")
    def list_frames(
        request: Request,
        asset_id: str,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = 100,
        offset: int = 0,
    ) -> dict[str, list]:
        frames = get_repository(request).list_frames(
            asset_id,
            start_frame=start_frame,
            end_frame=end_frame,
            limit=limit,
            offset=offset,
        )
        return {"frames": [frame_summary(frame) for frame in frames]}

    @router.get("/{asset_id}/framedata/{frame_num}")
    def get_frame_data(
        request: Request,
        asset_id: str,
        frame_num: int,
        format: str = "png",
        preview_max_dim: int = 128,
        scale: float = 1.0,
        flatfield_correction: bool = False,
        flatfield_q: float | None = None,
        flatfield_axis: int | None = None,
        background_correction: bool = False,
    ):
        if background_correction:
            raise HTTPException(
                status_code=501,
                detail="background_correction is not implemented for framedata yet.",
            )

        context = get_context(request)
        repository = get_repository(request)
        row = repository.get_frame_by_asset_index(asset_id, frame_num)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_num!r} was not found for asset {asset_id!r}.",
            )

        frame = retrieve_frame(str(row["id"]), context=context)
        array = frame.read()
        if array is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_num!r} has no image data.")

        if flatfield_correction:
            defaults = context.config.processing.flatfield
            resolved_q = defaults.flatfield_q if flatfield_q is None else flatfield_q
            resolved_axis = defaults.flatfield_axis if flatfield_axis is None else flatfield_axis
            try:
                array = flatfield_global_correction_for_framedata(
                    array,
                    q=resolved_q,
                    axis=resolved_axis,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            resolved_q = None
            resolved_axis = None

        processing_headers = {
            "X-Pelagia-Flatfield-Correction": str(flatfield_correction).lower(),
            "X-Pelagia-Background-Correction": str(background_correction).lower(),
        }
        if resolved_q is not None:
            processing_headers["X-Pelagia-Flatfield-Q"] = str(resolved_q)
        if resolved_axis is not None:
            processing_headers["X-Pelagia-Flatfield-Axis"] = str(resolved_axis)

        requested = format.lower()
        if requested == "matrix":
            matrix = np.asarray(scale_image(array, scale))
            return as_response(
                {
                    "asset_id": asset_id,
                    "frame_num": frame_num,
                    "frame_id": row["id"],
                    "dtype": str(matrix.dtype),
                    "shape": list(matrix.shape),
                    "scale": scale,
                    "flatfield_correction": flatfield_correction,
                    "flatfield_q": resolved_q,
                    "flatfield_axis": resolved_axis,
                    "background_correction": background_correction,
                    "data": matrix.tolist(),
                }
            )
        if requested == "preview":
            payload, media_type = encode_image(preview_image(array, preview_max_dim), "png")
            return Response(
                content=payload,
                media_type=media_type,
                headers={
                    "Content-Disposition": (
                        f'inline; filename="{asset_id}_frame_{frame_num}_preview.png"'
                    ),
                    "X-Pelagia-Preview": "true",
                    "X-Pelagia-Preview-Max-Dim": str(preview_max_dim),
                    **processing_headers,
                },
            )

        payload, media_type = encode_image(scale_image(array, scale), requested)
        extension = "jpg" if requested in {"jpg", "jpeg"} else "png"
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'inline; filename="{asset_id}_frame_{frame_num}.{extension}"'
                ),
                "X-Pelagia-Scale": str(scale),
                **processing_headers,
            },
        )

    @router.get("/{asset_id}/detections")
    def list_detections(
        request: Request,
        asset_id: str,
        frame_id: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        roi_index: int | None = None,
        min_bbox_x: int | None = None,
        max_bbox_x: int | None = None,
        min_bbox_y: int | None = None,
        max_bbox_y: int | None = None,
        min_bbox_w: int | None = None,
        max_bbox_w: int | None = None,
        min_bbox_h: int | None = None,
        max_bbox_h: int | None = None,
        min_area: float | None = None,
        max_area: float | None = None,
        min_perimeter: float | None = None,
        max_perimeter: float | None = None,
        roi_encoding: str | None = None,
        roi_format: str | None = None,
        mask_encoding: str | None = None,
        mask_format: str | None = None,
        limit: int | None = 100,
        offset: int = 0,
    ) -> dict:
        detections = get_repository(request).list_detections(
            asset_id,
            frame_id=frame_id,
            start_frame=start_frame,
            end_frame=end_frame,
            roi_index=roi_index,
            min_bbox_x=min_bbox_x,
            max_bbox_x=max_bbox_x,
            min_bbox_y=min_bbox_y,
            max_bbox_y=max_bbox_y,
            min_bbox_w=min_bbox_w,
            max_bbox_w=max_bbox_w,
            min_bbox_h=min_bbox_h,
            max_bbox_h=max_bbox_h,
            min_area=min_area,
            max_area=max_area,
            min_perimeter=min_perimeter,
            max_perimeter=max_perimeter,
            roi_encoding=roi_encoding,
            roi_format=roi_format,
            mask_encoding=mask_encoding,
            mask_format=mask_format,
            limit=limit,
            offset=offset,
        )
        summaries = [detection_summary(detection) for detection in detections]
        return {
            "detections": summaries,
            "page": page_metadata(limit=limit, offset=offset, count=len(summaries)),
        }
else:
    router = None
