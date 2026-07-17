from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from ..domain import PipelineStage


COMMAND_TYPE_KEY = "command_type"
COMMAND_VERSION_KEY = "command_version"
COMMAND_VERSION = 1


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _payload_values(payload: Mapping[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in excluded and key not in {COMMAND_TYPE_KEY, COMMAND_VERSION_KEY}
    }


@dataclass(frozen=True, slots=True)
class FrameSelection:
    frame_ids: tuple[str, ...] = ()
    frame_id: str | None = None
    asset_id: str | None = None
    run_id: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    limit: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrameSelection":
        values = payload.get("frame_ids") or ()
        return cls(
            frame_ids=tuple(value for value in values if value),
            frame_id=None if payload.get("frame_id") is None else str(payload["frame_id"]),
            asset_id=None if payload.get("asset_id") is None else str(payload["asset_id"]),
            run_id=None if payload.get("run_id") is None else str(payload["run_id"]),
            start_frame=payload.get("start_frame"),
            end_frame=payload.get("end_frame"),
            limit=payload.get("limit"),
        )

    def to_payload(self) -> dict[str, Any]:
        return _compact(
            {
                "frame_ids": list(self.frame_ids),
                "frame_id": self.frame_id,
                "asset_id": self.asset_id,
                "run_id": self.run_id,
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "limit": self.limit,
            }
        )


@dataclass(frozen=True, slots=True)
class JobCommand:
    """Versioned job payload contract with flat legacy-compatible serialization."""

    command_type: ClassVar[str]
    stage: ClassVar[PipelineStage]

    @classmethod
    def _validate_payload(cls, payload: Mapping[str, Any]) -> None:
        command_type = payload.get(COMMAND_TYPE_KEY)
        if command_type is not None and command_type != cls.command_type:
            raise ValueError(
                f"Expected {cls.command_type!r} command payload, got {command_type!r}."
            )
        version = payload.get(COMMAND_VERSION_KEY)
        if version is not None and int(version) != COMMAND_VERSION:
            raise ValueError(f"Unsupported {cls.command_type!r} command version {version!r}.")

    def _payload_header(self) -> dict[str, Any]:
        return {COMMAND_TYPE_KEY: self.command_type, COMMAND_VERSION_KEY: COMMAND_VERSION}


@dataclass(frozen=True, slots=True)
class ExtractFramesCommand(JobCommand):
    command_type = "extract_frames"
    stage = PipelineStage.EXTRACT_FRAMES

    source_path: str | None = None
    kind: str | None = None
    recursive: bool = False
    n_tile: int | None = None
    adaptive_background_subtraction: bool | None = None
    adaptive_background_period: int | None = None
    apply_mask: bool | None = None
    mask_path: str | None = None
    enqueue_segment: bool = False
    padding: int | None = None
    roi_encoding: str | None = None
    generate_backgrounds: bool | None = None
    generate_flatfield_profiles: bool | None = None
    flatfield_axis: int | None = None
    background_window_stride: int | None = None
    background_window_width: int | None = None
    flatfield_window_stride: int | None = None
    flatfield_window_width: int | None = None
    background_encoding: str | None = None
    background_quality: int | None = None
    collections: tuple[str, ...] = ()
    checksum_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExtractFramesCommand":
        cls._validate_payload(payload)
        known = {
            "source_path", "kind", "recursive", "n_tile", "adaptive_background_subtraction",
            "adaptive_background_period", "apply_mask", "mask_path", "enqueue_segment",
            "padding", "roi_encoding", "generate_backgrounds", "generate_flatfield_profiles",
            "flatfield_axis",
            "background_window_stride",
            "background_window_width", "flatfield_window_stride", "flatfield_window_width",
            "background_encoding", "background_quality",
            "collections", "checksum_status", "metadata",
        }
        return cls(
            source_path=payload.get("source_path"),
            kind=payload.get("kind"),
            recursive=bool(payload.get("recursive", False)),
            n_tile=payload.get("n_tile"),
            adaptive_background_subtraction=payload.get("adaptive_background_subtraction"),
            adaptive_background_period=payload.get("adaptive_background_period"),
            apply_mask=payload.get("apply_mask"),
            mask_path=payload.get("mask_path"),
            enqueue_segment=bool(payload.get("enqueue_segment", False)),
            padding=payload.get("padding"),
            roi_encoding=payload.get("roi_encoding"),
            generate_backgrounds=payload.get("generate_backgrounds"),
            generate_flatfield_profiles=payload.get("generate_flatfield_profiles"),
            flatfield_axis=payload.get("flatfield_axis"),
            background_window_stride=payload.get("background_window_stride"),
            background_window_width=payload.get("background_window_width"),
            flatfield_window_stride=payload.get("flatfield_window_stride"),
            flatfield_window_width=payload.get("flatfield_window_width"),
            background_encoding=payload.get("background_encoding"),
            background_quality=payload.get("background_quality"),
            collections=tuple(str(value) for value in payload.get("collections") or ()),
            checksum_status=payload.get("checksum_status"),
            metadata=dict(payload.get("metadata") or {}),
            extra=_payload_values(payload, known),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._payload_header(),
            **_compact(self.extra),
            **_compact(
                {
                    "source_path": self.source_path,
                    "kind": self.kind,
                    "recursive": self.recursive,
                    "n_tile": self.n_tile,
                    "adaptive_background_subtraction": self.adaptive_background_subtraction,
                    "adaptive_background_period": self.adaptive_background_period,
                    "apply_mask": self.apply_mask,
                    "mask_path": self.mask_path,
                    "enqueue_segment": self.enqueue_segment,
                    "padding": self.padding,
                    "roi_encoding": self.roi_encoding,
                    "generate_backgrounds": self.generate_backgrounds,
                    "generate_flatfield_profiles": self.generate_flatfield_profiles,
                    "flatfield_axis": self.flatfield_axis,
                    "background_window_stride": self.background_window_stride,
                    "background_window_width": self.background_window_width,
                    "flatfield_window_stride": self.flatfield_window_stride,
                    "flatfield_window_width": self.flatfield_window_width,
                    "background_encoding": self.background_encoding,
                    "background_quality": self.background_quality,
                    "collections": list(self.collections),
                    "checksum_status": self.checksum_status,
                    "metadata": self.metadata,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class PreprocessFramesCommand(JobCommand):
    command_type = "preprocess_frames"
    stage = PipelineStage.PREPROCESS_FRAMES

    selection: FrameSelection
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PreprocessFramesCommand":
        cls._validate_payload(payload)
        selection = FrameSelection.from_payload(payload)
        return cls(
            selection=selection,
            options=_payload_values(payload, set(selection.to_payload())),
        )

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_header(), **self.selection.to_payload(), **_compact(self.options)}


@dataclass(frozen=True, slots=True)
class SegmentFramesCommand(JobCommand):
    command_type = "segment_frames"
    stage = PipelineStage.SEGMENT

    selection: FrameSelection
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SegmentFramesCommand":
        cls._validate_payload(payload)
        selection = FrameSelection.from_payload(payload)
        return cls(
            selection=selection,
            options=_payload_values(payload, set(selection.to_payload())),
        )

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_header(), **self.selection.to_payload(), **_compact(self.options)}


@dataclass(frozen=True, slots=True)
class FrameBackgroundCommand(JobCommand):
    command_type = "frame_background"
    stage = PipelineStage.BACKGROUND_FRAMES

    selection: FrameSelection
    payload_kind: str = "original"
    encoding: str = "zstd"
    quality: int | None = None
    window_stride: int | None = None
    window_width: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrameBackgroundCommand":
        cls._validate_payload(payload)
        selection = FrameSelection.from_payload(payload)
        return cls(
            selection=selection,
            payload_kind=str(payload.get("payload_kind", "original")),
            encoding=str(payload.get("encoding", "zstd")),
            quality=payload.get("quality"),
            window_stride=payload.get("window_stride"),
            window_width=payload.get("window_width"),
            extra=_payload_values(
                payload,
                {
                    *selection.to_payload(),
                    "payload_kind",
                    "encoding",
                    "quality",
                    "window_stride",
                    "window_width",
                },
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._payload_header(),
            **self.selection.to_payload(),
            **_compact(self.extra),
            **_compact(
                {
                    "payload_kind": self.payload_kind,
                    "encoding": self.encoding,
                    "quality": self.quality,
                    "window_stride": self.window_stride,
                    "window_width": self.window_width,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class RoiRefinementCommand(JobCommand):
    command_type = "roi_refinement"
    stage = PipelineStage.ROI_REFINEMENT

    detection_ids: tuple[str, ...]
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RoiRefinementCommand":
        cls._validate_payload(payload)
        return cls(
            detection_ids=tuple(str(value) for value in payload.get("detection_ids") or () if value),
            options=_payload_values(payload, {"detection_ids"}),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._payload_header(),
            "detection_ids": list(self.detection_ids),
            **_compact(self.options),
        }


def command_model(stage: PipelineStage | str) -> type[JobCommand] | None:
    return {
        PipelineStage.EXTRACT_FRAMES: ExtractFramesCommand,
        PipelineStage.PREPROCESS_FRAMES: PreprocessFramesCommand,
        PipelineStage.SEGMENT: SegmentFramesCommand,
        PipelineStage.BACKGROUND_FRAMES: FrameBackgroundCommand,
        PipelineStage.ROI_REFINEMENT: RoiRefinementCommand,
    }.get(PipelineStage(stage))


def command_from_payload(stage: PipelineStage | str, payload: Mapping[str, Any]) -> JobCommand:
    resolved = PipelineStage(stage)
    command_class = command_model(resolved)
    if command_class is None:
        raise ValueError(f"No typed command model is registered for stage {resolved.value!r}.")
    return command_class.from_payload(payload)
