from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request, Response
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    import numpy as np

    from ..schemas import AssetDetailResponse, AssetsListResponse, DetectionsListResponse, FramesListResponse
    from ..auth import scoped_project_id
    from ...processing.frame_correction import divide_background, flatfield_correction as apply_flatfield_array_correction
    from ...processing.frame_store import retrieve_frame
    from ._common import as_response, detection_summary, frame_summary, get_context, get_repository, page_metadata
    from ._images import encode_image, preview_image, scale_image

    router = APIRouter(prefix="/assets", tags=["assets"])

    @router.get("", response_model=AssetsListResponse)
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
                    project_id=scoped_project_id(request),
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
            project_id=scoped_project_id(request),
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
            project_id=scoped_project_id(request),
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

    @router.get("/{asset_id}", response_model=AssetDetailResponse)
    def get_asset(request: Request, asset_id: str) -> dict:
        repository = get_repository(request)
        project_id = scoped_project_id(request)
        asset = repository.get_asset(asset_id, project_id=project_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} was not found.")
        asset = dict(asset)
        asset["frame_count"] = repository.count_frames(asset_id, project_id=project_id)
        return {"asset": as_response(asset)}

    @router.get("/{asset_id}/frames", response_model=FramesListResponse)
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
            project_id=scoped_project_id(request),
            start_frame=start_frame,
            end_frame=end_frame,
            limit=limit,
            offset=offset,
        )
        return {"frames": [frame_summary(frame) for frame in frames]}

    @router.head("/{asset_id}/framedata/{frame_num}")
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
        flatfield_min_field_value: float | None = None,
        flatfield_max_field_value: float | None = None,
        background_correction: bool = False,
        background_min_field_value: float | None = None,
        background_max_field_value: float | None = None,
    ):
        context = get_context(request)
        repository = get_repository(request)
        row = repository.get_frame_by_asset_index(asset_id, frame_num, project_id=scoped_project_id(request))
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
            resolved_flatfield_min = (
                defaults.flatfield_min_field_value
                if flatfield_min_field_value is None
                else flatfield_min_field_value
            )
            resolved_flatfield_max = (
                defaults.flatfield_max_field_value
                if flatfield_max_field_value is None
                else flatfield_max_field_value
            )
            try:
                array = apply_flatfield_array_correction(
                    array,
                    q=resolved_q,
                    axis=resolved_axis,
                    min_field_value=resolved_flatfield_min,
                    max_field_value=resolved_flatfield_max,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            resolved_q = None
            resolved_axis = None
            resolved_flatfield_min = None
            resolved_flatfield_max = None

        if background_correction:
            defaults = context.config.processing.preprocessing
            resolved_background_min = (
                defaults.background_min_field_value
                if background_min_field_value is None
                else background_min_field_value
            )
            resolved_background_max = (
                defaults.background_max_field_value
                if background_max_field_value is None
                else background_max_field_value
            )
            try:
                background = getattr(frame, "bkg", None)
                if background is None:
                    raise ValueError(
                        "background_correction requires a generated background field for this frame."
                    )
                array = divide_background(
                    array,
                    background=background,
                    min_field_value=resolved_background_min,
                    max_field_value=resolved_background_max,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            resolved_background_min = None
            resolved_background_max = None

        processing_headers = {
            "X-Pelagia-Flatfield-Correction": str(flatfield_correction).lower(),
            "X-Pelagia-Background-Correction": str(background_correction).lower(),
        }
        if resolved_q is not None:
            processing_headers["X-Pelagia-Flatfield-Q"] = str(resolved_q)
        if resolved_axis is not None:
            processing_headers["X-Pelagia-Flatfield-Axis"] = str(resolved_axis)
        if resolved_flatfield_min is not None:
            processing_headers["X-Pelagia-Flatfield-Min-Field-Value"] = str(resolved_flatfield_min)
        if resolved_flatfield_max is not None:
            processing_headers["X-Pelagia-Flatfield-Max-Field-Value"] = str(resolved_flatfield_max)
        if resolved_background_min is not None:
            processing_headers["X-Pelagia-Background-Min-Field-Value"] = str(resolved_background_min)
        if resolved_background_max is not None:
            processing_headers["X-Pelagia-Background-Max-Field-Value"] = str(resolved_background_max)

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
                    "flatfield_min_field_value": resolved_flatfield_min,
                    "flatfield_max_field_value": resolved_flatfield_max,
                    "background_correction": background_correction,
                    "background_method": "divide" if background_correction else None,
                    "background_min_field_value": resolved_background_min,
                    "background_max_field_value": resolved_background_max,
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

    @router.get("/{asset_id}/detections", response_model=DetectionsListResponse)
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
            project_id=scoped_project_id(request),
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
