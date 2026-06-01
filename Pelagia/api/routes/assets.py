from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request, Response
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    import cv2
    import numpy as np

    from ...processing.frame_store import retrieve_frame
    from ._common import as_response, detection_summary, frame_summary, get_context, get_repository

    router = APIRouter(prefix="/assets", tags=["assets"])

    def _preview_image(array, max_dim: int) -> np.ndarray:
        if max_dim < 1:
            raise HTTPException(status_code=422, detail="preview_max_dim must be >= 1.")

        image = np.asarray(array)
        if image.ndim < 2:
            raise HTTPException(status_code=422, detail="Frame preview requires at least 2D image data.")

        height, width = image.shape[:2]
        if height < 1 or width < 1:
            raise HTTPException(status_code=422, detail="Frame preview requires non-empty image data.")

        scale = min(float(max_dim) / float(width), float(max_dim) / float(height), 1.0)
        if scale >= 1.0:
            return np.ascontiguousarray(image)

        preview_width = max(1, int(round(width * scale)))
        preview_height = max(1, int(round(height * scale)))
        return cv2.resize(
            np.ascontiguousarray(image),
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )

    def _encode_image(array, fmt: str) -> tuple[bytes, str]:
        image = np.ascontiguousarray(array)
        requested = fmt.lower()
        if requested == "jpg":
            requested = "jpeg"
        if requested == "png":
            ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
            media_type = "image/png"
        elif requested == "jpeg":
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            media_type = "image/jpeg"
        else:
            raise HTTPException(
                status_code=422,
                detail="Frame data format must be one of: png, jpg, jpeg, matrix, preview.",
            )
        if not ok:
            raise HTTPException(
                status_code=500,
                detail=f"Frame data could not be encoded as {requested}.",
            )
        return encoded.tobytes(), media_type

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
    ) -> dict[str, list]:
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
                )
            )
        }

    @router.get("/{asset_id}")
    def get_asset(request: Request, asset_id: str) -> dict:
        asset = get_repository(request).get_asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} was not found.")
        return {"asset": as_response(asset)}

    @router.get("/{asset_id}/frames")
    def list_frames(
        request: Request,
        asset_id: str,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = 100,
    ) -> dict[str, list]:
        frames = get_repository(request).list_frames(
            asset_id,
            start_frame=start_frame,
            end_frame=end_frame,
            limit=limit,
        )
        return {"frames": [frame_summary(frame) for frame in frames]}

    @router.get("/{asset_id}/framedata/{frame_num}")
    def get_frame_data(
        request: Request,
        asset_id: str,
        frame_num: int,
        format: str = "png",
        preview_max_dim: int = 128,
    ):
        repository = get_repository(request)
        row = repository.get_frame_by_asset_index(asset_id, frame_num)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_num!r} was not found for asset {asset_id!r}.",
            )

        frame = retrieve_frame(str(row["id"]), context=get_context(request))
        array = frame.read()
        if array is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_num!r} has no image data.")

        requested = format.lower()
        if requested == "matrix":
            matrix = np.asarray(array)
            return as_response(
                {
                    "asset_id": asset_id,
                    "frame_num": frame_num,
                    "frame_id": row["id"],
                    "dtype": str(matrix.dtype),
                    "shape": list(matrix.shape),
                    "data": matrix.tolist(),
                }
            )
        if requested == "preview":
            payload, media_type = _encode_image(_preview_image(array, preview_max_dim), "png")
            return Response(
                content=payload,
                media_type=media_type,
                headers={
                    "Content-Disposition": (
                        f'inline; filename="{asset_id}_frame_{frame_num}_preview.png"'
                    ),
                    "X-Pelagia-Preview": "true",
                    "X-Pelagia-Preview-Max-Dim": str(preview_max_dim),
                },
            )

        payload, media_type = _encode_image(array, requested)
        extension = "jpg" if requested in {"jpg", "jpeg"} else "png"
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'inline; filename="{asset_id}_frame_{frame_num}.{extension}"'
                )
            },
        )

    @router.get("/{asset_id}/detections")
    def list_detections(request: Request, asset_id: str) -> dict[str, list]:
        detections = get_repository(request).list_detections(asset_id)
        return {"detections": [detection_summary(detection) for detection in detections]}
else:
    router = None
