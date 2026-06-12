from __future__ import annotations

from typing import Any, Literal

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..schemas import OptionsResponse
    from ...domain import DetectionRecord, PipelineStage
    from ...processing.detection_refinement import (
        IdentityRoiRefinementModel,
        RoiRefinementOptions,
        refine_detections,
        refined_storage_candidate_detection_id,
    )
    from ...processing.frame_store import retrieve_frame
    from ...processing.oracle_unet_refiner import (
        OracleUnetRefinerError,
        resolve_refinement_model,
    )
    from ...processing.capabilities import roi_refinement_capabilities
    from ...services.models import ModelService
    from ._common import as_response, detection_summary, get_context, get_repository

    ModelKind = Literal["identity", "keras_artifact", "oracle_builder_unet"]
    RoiEncoding = Literal["png", "raw", "zstd", "auto"]

    class RoiRefinementRequest(BaseModel):
        model_config = ConfigDict(protected_namespaces=())

        detection_ids: list[str] = Field(default_factory=list)
        model_ref: str | None = None
        model_kind: ModelKind | None = None
        model_run_dir: str | None = None
        model_artifact: str | None = None
        batch_size: int | None = None
        tile_size: int | None = None
        overlap_fraction: float | None = None
        max_iterations: int | None = None
        expansion_pixels: int | None = None
        edge_touch_margin: int | None = None
        output_threshold: float | None = None
        encoding: RoiEncoding | None = None
        overlap_reconciliation_enabled: bool | None = None
        overlap_iou_threshold: float | None = None
        overlap_containment_threshold: float | None = None
        residual_discovery_enabled: bool | None = None
        residual_max_iterations: int | None = None
        residual_roi_assembly_method: str | None = None
        residual_roi_assembly_connectivity: int | None = None
        residual_min_area: float | None = None
        residual_min_width: float | None = None
        residual_min_height: float | None = None
        residual_min_width_plus_height: float | None = None
        residual_padding: int | None = None
        allow_frame_expansion: bool = True
        store: bool = True
        dry_run: bool = False

    class QueueRoiRefinementRequest(RoiRefinementRequest):
        run_id: str | None = None
        asset_id: str | None = None
        priority: int | None = None
        depends_on: list[str] | None = None

    router = APIRouter(prefix="/roi-refinement", tags=["roi-refinement"])

    def _model_dict(model: BaseModel) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def _resolved_encoding(value: str | None, default: str | None) -> str | None:
        encoding = value if value is not None else default
        if encoding is None:
            return None
        normalized = str(encoding).strip().lower()
        return None if normalized in {"", "auto", "default", "none", "null"} else normalized

    def _resolve_options(request: Request, body: RoiRefinementRequest) -> RoiRefinementOptions:
        defaults = get_context(request).config.processing.roi_refinement
        values = _model_dict(body)
        try:
            return RoiRefinementOptions(
                tile_size=values.get("tile_size") or defaults.tile_size,
                overlap_fraction=(
                    defaults.overlap_fraction
                    if values.get("overlap_fraction") is None
                    else values["overlap_fraction"]
                ),
                max_iterations=values.get("max_iterations") or defaults.max_iterations,
                expansion_pixels=(
                    defaults.expansion_pixels
                    if values.get("expansion_pixels") is None
                    else values["expansion_pixels"]
                ),
                edge_touch_margin=values.get("edge_touch_margin") or defaults.edge_touch_margin,
                output_threshold=(
                    defaults.output_threshold
                    if values.get("output_threshold") is None
                    else values["output_threshold"]
                ),
                batch_size=(
                    defaults.batch_size
                    if values.get("batch_size") is None
                    else values["batch_size"]
                ),
                encoding=_resolved_encoding(values.get("encoding"), defaults.encoding),
                overlap_reconciliation_enabled=(
                    defaults.overlap_reconciliation_enabled
                    if values.get("overlap_reconciliation_enabled") is None
                    else values["overlap_reconciliation_enabled"]
                ),
                overlap_iou_threshold=(
                    defaults.overlap_iou_threshold
                    if values.get("overlap_iou_threshold") is None
                    else values["overlap_iou_threshold"]
                ),
                overlap_containment_threshold=(
                    defaults.overlap_containment_threshold
                    if values.get("overlap_containment_threshold") is None
                    else values["overlap_containment_threshold"]
                ),
                residual_discovery_enabled=(
                    defaults.residual_discovery_enabled
                    if values.get("residual_discovery_enabled") is None
                    else values["residual_discovery_enabled"]
                ),
                residual_max_iterations=(
                    defaults.residual_max_iterations
                    if values.get("residual_max_iterations") is None
                    else values["residual_max_iterations"]
                ),
                residual_roi_assembly_method=(
                    defaults.residual_roi_assembly_method
                    if values.get("residual_roi_assembly_method") is None
                    else values["residual_roi_assembly_method"]
                ),
                residual_roi_assembly_connectivity=(
                    defaults.residual_roi_assembly_connectivity
                    if values.get("residual_roi_assembly_connectivity") is None
                    else values["residual_roi_assembly_connectivity"]
                ),
                residual_min_area=(
                    defaults.residual_min_area
                    if values.get("residual_min_area") is None
                    else values["residual_min_area"]
                ),
                residual_min_width=(
                    defaults.residual_min_width
                    if values.get("residual_min_width") is None
                    else values["residual_min_width"]
                ),
                residual_min_height=(
                    defaults.residual_min_height
                    if values.get("residual_min_height") is None
                    else values["residual_min_height"]
                ),
                residual_min_width_plus_height=(
                    defaults.residual_min_width_plus_height
                    if values.get("residual_min_width_plus_height") is None
                    else values["residual_min_width_plus_height"]
                ),
                residual_padding=(
                    defaults.residual_padding
                    if values.get("residual_padding") is None
                    else values["residual_padding"]
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _requested_model_metadata(request: Request, body: RoiRefinementRequest) -> dict[str, Any]:
        context = get_context(request)
        defaults = context.config.processing.roi_refinement
        using_defaults = body.model_kind is None and body.model_ref is None and body.model_run_dir is None
        model_ref = body.model_ref or (defaults.model_ref if defaults.enabled and using_defaults else None)
        model_kind = body.model_kind or (
            "keras_artifact"
            if model_ref
            else (defaults.model_kind if defaults.enabled and using_defaults else "identity")
        )
        model_run_dir = body.model_run_dir or defaults.model_run_dir
        model_artifact = body.model_artifact or defaults.model_artifact
        model_info = {
            "model_kind": model_kind,
            "model_ref": model_ref,
            "model_run_dir": model_run_dir,
            "model_artifact": model_artifact,
        }
        if model_ref:
            manifest = ModelService.from_config(context.config).find_model_artifact(model_ref)
            if manifest is None:
                raise HTTPException(status_code=422, detail=f"ROI refinement model_ref was not found: {model_ref!r}.")
            model_info["model"] = manifest
        return model_info

    def _resolve_model(request: Request, body: RoiRefinementRequest):
        context = get_context(request)
        defaults = context.config.processing.roi_refinement
        try:
            return resolve_refinement_model(
                context.config,
                model_kind=body.model_kind,
                model_ref=body.model_ref,
                model_run_dir=body.model_run_dir,
                model_artifact=body.model_artifact or defaults.model_artifact,
            )
        except (ValueError, OracleUnetRefinerError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/options", response_model=OptionsResponse)
    def get_roi_refinement_options(request: Request) -> dict:
        context = get_context(request)
        return as_response(roi_refinement_capabilities(context.config))

    @router.post("")
    def refine_candidate_rois(request: Request, body: RoiRefinementRequest) -> dict:
        if not body.detection_ids:
            raise HTTPException(status_code=422, detail="Provide at least one detection_id.")
        repository = get_repository(request)
        context = get_context(request)
        options = _resolve_options(request, body)
        model_metadata = _requested_model_metadata(request, body)

        candidate_rows = []
        missing_ids = []
        for detection_id in body.detection_ids:
            row = repository.get_detection(detection_id)
            if row is None:
                missing_ids.append(detection_id)
            else:
                candidate_rows.append(row)
        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Detection(s) not found: {', '.join(missing_ids)}",
            )

        if body.dry_run:
            return as_response(
                {
                    "dry_run": True,
                    "store": body.store,
                    "detection_ids": body.detection_ids,
                    "candidate_count": len(candidate_rows),
                    "resolved_options": _options_dict(options),
                    **model_metadata,
                }
            )

        model = _resolve_model(request, body)
        method = (
            "identity"
            if isinstance(model, IdentityRoiRefinementModel)
            else getattr(model, "method_name", None) or model.__class__.__name__
        )
        detection_records = [DetectionRecord.from_row(row) for row in candidate_rows]

        frame_loader = None
        if body.allow_frame_expansion:
            frame_loader = lambda frame_id: retrieve_frame(frame_id, context=context, payload_kind="preprocessed").read()

        try:
            results = refine_detections(
                detection_records,
                model=model,
                frame_loader=frame_loader,
                options=options,
                method=method,
            )
            refined_records = [
                result.as_detection_record(encoding=options.encoding)
                for result in results
            ]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        stored_rows = []
        if body.store:
            try:
                stored_rows = repository.upsert_refined_detections(
                    [
                        (refined_storage_candidate_detection_id(result), refined_record)
                        for result, refined_record in zip(results, refined_records)
                    ]
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to store refined detections: {exc}") from exc

        return as_response(
            {
                "dry_run": False,
                "stored": bool(body.store),
                "detection_ids": body.detection_ids,
                "candidate_count": len(candidate_rows),
                "refined_count": len(refined_records),
                "stored_count": len(stored_rows),
                "synthetic_refined_count": sum(
                    1 for result in results if result.candidate_detection.metadata.get("synthetic_candidate")
                ),
                "resolved_options": _options_dict(options),
                "allow_frame_expansion": body.allow_frame_expansion,
                **model_metadata,
                "refined_detections": [
                    detection_summary(row)
                    for row in (stored_rows if body.store else refined_records)
                ],
            }
        )

    @router.post("/jobs")
    def queue_roi_refinement_job(request: Request, body: QueueRoiRefinementRequest) -> dict:
        if not body.detection_ids:
            raise HTTPException(status_code=422, detail="Provide at least one detection_id.")
        repository = get_repository(request)
        context = get_context(request)
        options = _resolve_options(request, body)
        model_metadata = _requested_model_metadata(request, body)

        first_detection = repository.get_detection(body.detection_ids[0])
        if first_detection is None:
            raise HTTPException(status_code=404, detail=f"Detection {body.detection_ids[0]!r} was not found.")
        payload = {
            "detection_ids": body.detection_ids,
            "model_ref": body.model_ref,
            "model_kind": body.model_kind,
            "model_run_dir": body.model_run_dir,
            "model_artifact": body.model_artifact,
            "batch_size": options.batch_size,
            "tile_size": options.tile_size,
            "overlap_fraction": options.overlap_fraction,
            "max_iterations": options.max_iterations,
            "expansion_pixels": options.expansion_pixels,
            "edge_touch_margin": options.edge_touch_margin,
            "output_threshold": options.output_threshold,
            "encoding": options.encoding,
            "overlap_reconciliation_enabled": options.overlap_reconciliation_enabled,
            "overlap_iou_threshold": options.overlap_iou_threshold,
            "overlap_containment_threshold": options.overlap_containment_threshold,
            "residual_discovery_enabled": options.residual_discovery_enabled,
            "residual_max_iterations": options.residual_max_iterations,
            "residual_roi_assembly_method": options.residual_roi_assembly_method,
            "residual_roi_assembly_connectivity": options.residual_roi_assembly_connectivity,
            "residual_min_area": options.residual_min_area,
            "residual_min_width": options.residual_min_width,
            "residual_min_height": options.residual_min_height,
            "residual_min_width_plus_height": options.residual_min_width_plus_height,
            "residual_padding": options.residual_padding,
            "allow_frame_expansion": body.allow_frame_expansion,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        run_id = body.run_id or first_detection.get("run_id")
        asset_id = body.asset_id or first_detection.get("asset_id")
        if body.dry_run:
            return as_response(
                {
                    "dry_run": True,
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "priority": body.priority,
                    "depends_on": body.depends_on or [],
                    "payload": payload,
                    "resolved_options": _options_dict(options),
                    **model_metadata,
                }
            )
        job = repository.create_job(
            PipelineStage.ROI_REFINEMENT,
            run_id=run_id,
            asset_id=asset_id,
            priority=body.priority,
            payload=payload,
            depends_on=body.depends_on or [],
            summary=f"roi refinement queued for {len(body.detection_ids)} detections",
        )
        return {"job": as_response(job)}

    def _options_dict(options: RoiRefinementOptions) -> dict[str, Any]:
        return {
            "tile_size": options.tile_size,
            "overlap_fraction": options.overlap_fraction,
            "max_iterations": options.max_iterations,
            "expansion_pixels": options.expansion_pixels,
            "edge_touch_margin": options.edge_touch_margin,
            "output_threshold": options.output_threshold,
            "batch_size": options.batch_size,
            "encoding": options.encoding,
            "overlap_reconciliation_enabled": options.overlap_reconciliation_enabled,
            "overlap_iou_threshold": options.overlap_iou_threshold,
            "overlap_containment_threshold": options.overlap_containment_threshold,
            "residual_discovery_enabled": options.residual_discovery_enabled,
            "residual_max_iterations": options.residual_max_iterations,
            "residual_roi_assembly_method": options.residual_roi_assembly_method,
            "residual_roi_assembly_connectivity": options.residual_roi_assembly_connectivity,
            "residual_min_area": options.residual_min_area,
            "residual_min_width": options.residual_min_width,
            "residual_min_height": options.residual_min_height,
            "residual_min_width_plus_height": options.residual_min_width_plus_height,
            "residual_padding": options.residual_padding,
        }
else:
    router = None
