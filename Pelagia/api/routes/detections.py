from __future__ import annotations

from typing import Literal

try:
    from fastapi import APIRouter, HTTPException, Request, Response
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    import numpy as np

    from ..schemas import DetectionDetailResponse, DetectionImageMatrixResponse, DetectionsListResponse
    from ..auth import scoped_project_id
    from ...processing.frame_codec import decode_array_payload
    from ._common import as_response, detection_summary, get_repository, page_metadata
    from ._images import (
        add_scale_bar,
        encode_image,
        invert_image,
        pad_image_to_square,
        resize_image_to_dimension,
        scale_image,
    )

    router = APIRouter(prefix="/detections", tags=["detections"])
    refined_router = APIRouter(prefix="/refined-detections", tags=["refined-detections"])
    routers = [refined_router]

    def _detection_payload_array(detection: dict, payload_kind: str) -> np.ndarray:
        if payload_kind not in {"roi", "mask"}:
            raise HTTPException(status_code=422, detail="payload_kind must be one of: roi, mask.")

        payload = detection.get(f"{payload_kind}_payload")
        if payload is None:
            label = "ROI image" if payload_kind == "roi" else "ROI mask"
            raise HTTPException(
                status_code=404,
                detail=f"Detection {detection.get('id')!r} has no {label} payload.",
            )
        try:
            return decode_array_payload(
                payload,
                {
                    "kvstore_encoding": detection.get(f"{payload_kind}_encoding"),
                    "kvstore_format": detection.get(f"{payload_kind}_format"),
                    "dtype": detection.get(f"{payload_kind}_dtype"),
                    "shape": detection.get(f"{payload_kind}_shape"),
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _apply_detection_mask(array: np.ndarray, detection: dict) -> np.ndarray:
        mask = _detection_payload_array(detection, "mask")
        image = np.asarray(array)
        mask_array = np.asarray(mask)
        if mask_array.shape[:2] != image.shape[:2]:
            raise HTTPException(
                status_code=422,
                detail=(
                    "ROI mask shape does not match ROI image shape: "
                    f"{list(mask_array.shape)} vs {list(image.shape)}."
                ),
            )
        keep = mask_array > 0
        if image.ndim > keep.ndim:
            keep = keep[..., np.newaxis]
        return np.where(keep, image, np.zeros((), dtype=image.dtype)).astype(image.dtype, copy=False)

    def _detection_payload_response(
        *,
        detection: dict,
        detection_id: str,
        payload_kind: str,
        format: str,
        scale: float,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        apply_mask: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        array = _detection_payload_array(detection, payload_kind)
        mask_applied = bool(apply_mask and payload_kind == "roi")
        if mask_applied:
            array = _apply_detection_mask(array, detection)
        requested = format.lower()
        source_height, source_width = np.asarray(array).shape[:2]
        transformed = np.asarray(array)
        square_padding_requested = bool(pad_square or square)
        if square_padding_requested:
            transformed = pad_image_to_square(transformed)
        if invert:
            transformed = invert_image(transformed)
        delivered = (
            resize_image_to_dimension(transformed, width=width, height=height)
            if width is not None or height is not None
            else scale_image(transformed, scale)
        )
        if scale_bar:
            delivered = add_scale_bar(
                delivered,
                length_px=scale_bar_length_px,
                height_px=scale_bar_height_px,
                margin_px=scale_bar_margin_px,
                color=scale_bar_color,
            )
        delivered_height, delivered_width = np.asarray(delivered).shape[:2]
        response_scale_x = delivered_width / source_width
        response_scale_y = delivered_height / source_height

        if requested == "matrix":
            matrix = np.asarray(delivered)
            return as_response(
                {
                    "detection_id": detection_id,
                    "frame_id": detection.get("frame_id"),
                    "asset_id": detection.get("asset_id"),
                    "payload_kind": payload_kind,
                    "dtype": str(matrix.dtype),
                    "shape": list(matrix.shape),
                    "scale": None if width is not None or height is not None else scale,
                    "requested_width": width,
                    "requested_height": height,
                    "pad_square": square_padding_requested,
                    "inverted": invert,
                    "mask_applied": mask_applied,
                    "scale_bar": scale_bar,
                    "data": matrix.tolist(),
                }
            )

        payload, media_type = encode_image(delivered, requested)
        extension = "jpg" if requested in {"jpg", "jpeg"} else "png"
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'inline; filename="{detection_id}_{payload_kind}.{extension}"'
                ),
                "X-Pelagia-Detection-Id": str(detection_id),
                "X-Pelagia-Payload-Kind": payload_kind,
                "X-Pelagia-Source-Width": str(source_width),
                "X-Pelagia-Source-Height": str(source_height),
                "X-Pelagia-Image-Width": str(delivered_width),
                "X-Pelagia-Image-Height": str(delivered_height),
                "X-Pelagia-Scale-X": str(response_scale_x),
                "X-Pelagia-Scale-Y": str(response_scale_y),
                "X-Pelagia-Scale": str(scale),
                "X-Pelagia-Pad-Square": str(square_padding_requested).lower(),
                "X-Pelagia-Inverted": str(bool(invert)).lower(),
                "X-Pelagia-Mask-Applied": str(mask_applied).lower(),
                "X-Pelagia-Scale-Bar": str(bool(scale_bar)).lower(),
            },
        )

    _IMAGE_RESPONSES = {
        200: {
            "content": {
                "image/png": {},
                "image/jpeg": {},
                "application/json": {"schema": DetectionImageMatrixResponse.model_json_schema()},
            },
            "description": "Detection ROI/mask image bytes or matrix data.",
        },
        404: {"description": "Detection or requested payload was not found."},
        422: {"description": "Unsupported image options."},
    }

    @router.get("", response_model=DetectionsListResponse)
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
        refinement_state: Literal["any", "refined", "unrefined", "has-refinement", "needs-refinement"] = "any",
        sort_by: Literal["area", "byte_size", "id", "asset_frame", "random"] = "asset_frame",
        sort_dir: Literal["asc", "desc"] = "desc",
        limit: int | None = 100,
        offset: int = 0,
    ) -> dict:
        detections = get_repository(request).list_detections(
            asset_id=asset_id,
            project_id=scoped_project_id(request),
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
            refinement_state=None if refinement_state == "any" else refinement_state,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )
        summaries = [detection_summary(detection) for detection in detections]
        return {
            "detections": summaries,
            "page": page_metadata(limit=limit, offset=offset, count=len(summaries)),
        }

    @refined_router.get("/{refined_detection_id}", response_model=DetectionDetailResponse)
    def get_refined_detection(request: Request, refined_detection_id: str) -> dict:
        detection = get_repository(request).get_refined_detection(
            refined_detection_id,
            project_id=scoped_project_id(request),
        )
        if detection is None:
            raise HTTPException(
                status_code=404,
                detail=f"Refined detection {refined_detection_id!r} was not found.",
            )
        return {"detection": detection_summary(detection, include_payload=True)}

    @refined_router.head("/{refined_detection_id}/roi", responses=_IMAGE_RESPONSES)
    @refined_router.get("/{refined_detection_id}/roi", responses=_IMAGE_RESPONSES)
    def get_refined_detection_roi_by_id(
        request: Request,
        refined_detection_id: str,
        format: str = "png",
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        apply_mask: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        detection = get_repository(request).get_refined_detection(
            refined_detection_id,
            project_id=scoped_project_id(request),
        )
        if detection is None:
            raise HTTPException(
                status_code=404,
                detail=f"Refined detection {refined_detection_id!r} was not found.",
            )
        return _detection_payload_response(
            detection=detection,
            detection_id=refined_detection_id,
            payload_kind="roi",
            format=format,
            scale=scale,
            width=width,
            height=height,
            pad_square=pad_square,
            square=square,
            invert=invert,
            apply_mask=apply_mask,
            scale_bar=scale_bar,
            scale_bar_length_px=scale_bar_length_px,
            scale_bar_height_px=scale_bar_height_px,
            scale_bar_margin_px=scale_bar_margin_px,
            scale_bar_color=scale_bar_color,
        )

    @refined_router.head("/{refined_detection_id}/mask", responses=_IMAGE_RESPONSES)
    @refined_router.get("/{refined_detection_id}/mask", responses=_IMAGE_RESPONSES)
    def get_refined_detection_mask_by_id(
        request: Request,
        refined_detection_id: str,
        format: str = "png",
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        detection = get_repository(request).get_refined_detection(
            refined_detection_id,
            project_id=scoped_project_id(request),
        )
        if detection is None:
            raise HTTPException(
                status_code=404,
                detail=f"Refined detection {refined_detection_id!r} was not found.",
            )
        return _detection_payload_response(
            detection=detection,
            detection_id=refined_detection_id,
            payload_kind="mask",
            format=format,
            scale=scale,
            width=width,
            height=height,
            pad_square=pad_square,
            square=square,
            invert=invert,
            scale_bar=scale_bar,
            scale_bar_length_px=scale_bar_length_px,
            scale_bar_height_px=scale_bar_height_px,
            scale_bar_margin_px=scale_bar_margin_px,
            scale_bar_color=scale_bar_color,
        )

    @router.get("/{detection_id}", response_model=DetectionDetailResponse)
    def get_detection(request: Request, detection_id: str) -> dict:
        detection = get_repository(request).get_detection(detection_id, project_id=scoped_project_id(request))
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Detection {detection_id!r} was not found.")
        return {"detection": detection_summary(detection, include_payload=True)}

    @router.head("/{detection_id}/framedata", responses=_IMAGE_RESPONSES)
    @router.get("/{detection_id}/framedata", responses=_IMAGE_RESPONSES)
    def get_detection_frame_data(
        request: Request,
        detection_id: str,
        format: str = "png",
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        apply_mask: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        detection = get_repository(request).get_detection(detection_id, project_id=scoped_project_id(request))
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Detection {detection_id!r} was not found.")

        return _detection_payload_response(
            detection=detection,
            detection_id=detection_id,
            payload_kind="roi",
            format=format,
            scale=scale,
            width=width,
            height=height,
            pad_square=pad_square,
            square=square,
            invert=invert,
            apply_mask=apply_mask,
            scale_bar=scale_bar,
            scale_bar_length_px=scale_bar_length_px,
            scale_bar_height_px=scale_bar_height_px,
            scale_bar_margin_px=scale_bar_margin_px,
            scale_bar_color=scale_bar_color,
        )

    @router.head("/{detection_id}/roi", responses=_IMAGE_RESPONSES)
    @router.get("/{detection_id}/roi", responses=_IMAGE_RESPONSES)
    def get_detection_roi(
        request: Request,
        detection_id: str,
        format: str = "png",
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        apply_mask: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        detection = get_repository(request).get_detection(detection_id, project_id=scoped_project_id(request))
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Detection {detection_id!r} was not found.")
        return _detection_payload_response(
            detection=detection,
            detection_id=detection_id,
            payload_kind="roi",
            format=format,
            scale=scale,
            width=width,
            height=height,
            pad_square=pad_square,
            square=square,
            invert=invert,
            apply_mask=apply_mask,
            scale_bar=scale_bar,
            scale_bar_length_px=scale_bar_length_px,
            scale_bar_height_px=scale_bar_height_px,
            scale_bar_margin_px=scale_bar_margin_px,
            scale_bar_color=scale_bar_color,
        )

    @router.head("/{detection_id}/mask", responses=_IMAGE_RESPONSES)
    @router.get("/{detection_id}/mask", responses=_IMAGE_RESPONSES)
    def get_detection_mask(
        request: Request,
        detection_id: str,
        format: str = "png",
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        detection = get_repository(request).get_detection(detection_id, project_id=scoped_project_id(request))
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Detection {detection_id!r} was not found.")
        return _detection_payload_response(
            detection=detection,
            detection_id=str(detection.get("id") or detection_id),
            payload_kind="mask",
            format=format,
            scale=scale,
            width=width,
            height=height,
            pad_square=pad_square,
            square=square,
            invert=invert,
            scale_bar=scale_bar,
            scale_bar_length_px=scale_bar_length_px,
            scale_bar_height_px=scale_bar_height_px,
            scale_bar_margin_px=scale_bar_margin_px,
            scale_bar_color=scale_bar_color,
        )

    @router.head("/{detection_id}/refined-mask", responses=_IMAGE_RESPONSES)
    @router.get("/{detection_id}/refined-mask", responses=_IMAGE_RESPONSES)
    def get_refined_detection_mask(
        request: Request,
        detection_id: str,
        format: str = "png",
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        detection = get_repository(request).get_refined_detection_for_candidate(
            detection_id,
            project_id=scoped_project_id(request),
        )
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Refined detection for {detection_id!r} was not found.")
        return _detection_payload_response(
            detection=detection,
            detection_id=str(detection.get("id") or detection_id),
            payload_kind="mask",
            format=format,
            scale=scale,
            width=width,
            height=height,
            pad_square=pad_square,
            square=square,
            invert=invert,
            scale_bar=scale_bar,
            scale_bar_length_px=scale_bar_length_px,
            scale_bar_height_px=scale_bar_height_px,
            scale_bar_margin_px=scale_bar_margin_px,
            scale_bar_color=scale_bar_color,
        )

    @router.head("/{detection_id}/refined-roi", responses=_IMAGE_RESPONSES)
    @router.get("/{detection_id}/refined-roi", responses=_IMAGE_RESPONSES)
    def get_refined_detection_roi(
        request: Request,
        detection_id: str,
        format: str = "png",
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        pad_square: bool = False,
        square: bool = False,
        invert: bool = False,
        apply_mask: bool = False,
        scale_bar: bool = False,
        scale_bar_length_px: int | None = None,
        scale_bar_height_px: int = 4,
        scale_bar_margin_px: int = 8,
        scale_bar_color: Literal["white", "black"] = "white",
    ):
        detection = get_repository(request).get_refined_detection_for_candidate(
            detection_id,
            project_id=scoped_project_id(request),
        )
        if detection is None:
            raise HTTPException(status_code=404, detail=f"Refined detection for {detection_id!r} was not found.")
        return _detection_payload_response(
            detection=detection,
            detection_id=str(detection.get("id") or detection_id),
            payload_kind="roi",
            format=format,
            scale=scale,
            width=width,
            height=height,
            pad_square=pad_square,
            square=square,
            invert=invert,
            apply_mask=apply_mask,
            scale_bar=scale_bar,
            scale_bar_length_px=scale_bar_length_px,
            scale_bar_height_px=scale_bar_height_px,
            scale_bar_margin_px=scale_bar_margin_px,
            scale_bar_color=scale_bar_color,
        )
else:
    router = None
    routers = []
