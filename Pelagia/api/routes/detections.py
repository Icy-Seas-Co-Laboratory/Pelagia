from __future__ import annotations

from typing import Literal

try:
    from fastapi import APIRouter, HTTPException, Request, Response
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    import numpy as np

    from ...processing.frame_codec import decode_array_payload
    from ._common import as_response, detection_summary, get_repository
    from ._images import encode_image, scale_image

    router = APIRouter(prefix="/detections", tags=["detections"])

    def _detection_image_array(detection: dict) -> np.ndarray:
        payload = detection.get("roi_payload")
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail=f"Detection {detection.get('id')!r} has no ROI image payload.",
            )
        try:
            return decode_array_payload(
                payload,
                {
                    "kvstore_encoding": detection.get("roi_encoding"),
                    "kvstore_format": detection.get("roi_format"),
                    "dtype": detection.get("roi_dtype"),
                    "shape": detection.get("roi_shape"),
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get("")
    def list_detections(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
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
        sort_by: Literal["area", "byte_size", "id", "asset_frame"] = "asset_frame",
        sort_dir: Literal["asc", "desc"] = "desc",
        limit: int | None = 100,
        offset: int = 0,
    ) -> dict[str, list]:
        detections = get_repository(request).list_detections(
            asset_id=asset_id,
            run_id=run_id,
            collection=collection,
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
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )
        return {"detections": [detection_summary(detection) for detection in detections]}

    @router.get("/{detection_id}")
    def get_detection(request: Request, detection_id: str) -> dict:
        detection = get_repository(request).get_detection(detection_id)
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Detection {detection_id!r} was not found.")
        return {"detection": as_response(detection)}

    @router.get("/{detection_id}/framedata")
    def get_detection_frame_data(
        request: Request,
        detection_id: str,
        format: str = "png",
        scale: float = 1.0,
    ):
        detection = get_repository(request).get_detection(detection_id)
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Detection {detection_id!r} was not found.")

        array = _detection_image_array(detection)
        requested = format.lower()
        if requested == "matrix":
            matrix = np.asarray(scale_image(array, scale))
            return as_response(
                {
                    "detection_id": detection_id,
                    "frame_id": detection.get("frame_id"),
                    "asset_id": detection.get("asset_id"),
                    "dtype": str(matrix.dtype),
                    "shape": list(matrix.shape),
                    "scale": scale,
                    "data": matrix.tolist(),
                }
            )

        payload, media_type = encode_image(scale_image(array, scale), requested)
        extension = "jpg" if requested in {"jpg", "jpeg"} else "png"
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'inline; filename="{detection_id}_framedata.{extension}"'
                ),
                "X-Pelagia-Scale": str(scale),
            },
        )
else:
    router = None
