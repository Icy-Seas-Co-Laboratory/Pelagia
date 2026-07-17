"""Generate aligned mean background windows while an asset is ingested."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..services.context import AppContext
from ..utils.serialization import json_ready
from .frame_codec import encode_array_payload
from .timing import measure_phase


@dataclass(slots=True)
class _WindowAccumulator:
    total: np.ndarray | None
    profile_total: np.ndarray | None
    profile_value_count: int
    background_count: int
    background_source_frame_ids: list[str]
    profile_frame_count: int
    profile_source_frame_ids: list[str]


class _BufferedAssignmentRepository:
    def __init__(self, repository: Any) -> None:
        self.repository = repository
        self.assignments: list[dict[str, Any]] = []

    def update_frame_background_payload_assignments(
        self,
        assignments: list[dict[str, Any]],
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.assignments.extend(assignments)
        return [{"id": assignment["frame_id"]} for assignment in assignments]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.repository, name)


class MeanFieldIngestAddon:
    """Accumulate aligned image means and axis-aware sensor profiles during ingestion."""

    def __init__(
        self,
        *,
        context: AppContext,
        project_id: str | None,
        window_stride: int,
        window_width: int,
        flatfield_window_stride: int | None = None,
        flatfield_window_width: int | None = None,
        encoding: str = "zstd",
        quality: int | None = None,
        generate_backgrounds: bool = True,
        generate_flatfield_profiles: bool = True,
        flatfield_axis: int = 0,
    ) -> None:
        self.context = context
        self.project_id = project_id
        resolved_flatfield_stride = int(
            window_stride if flatfield_window_stride is None else flatfield_window_stride
        )
        resolved_flatfield_width = int(
            window_width if flatfield_window_width is None else flatfield_window_width
        )
        self._delegates: tuple[MeanFieldIngestAddon, MeanFieldIngestAddon] | None = None
        self._assignment_buffers: tuple[
            _BufferedAssignmentRepository,
            _BufferedAssignmentRepository,
        ] | None = None
        if (
            generate_backgrounds
            and generate_flatfield_profiles
            and (
                int(window_stride) != resolved_flatfield_stride
                or int(window_width) != resolved_flatfield_width
            )
        ):
            background_repository = _BufferedAssignmentRepository(context.repository)
            flatfield_repository = _BufferedAssignmentRepository(context.repository)
            self._assignment_buffers = (background_repository, flatfield_repository)
            self._delegates = (
                MeanFieldIngestAddon(
                    context=replace(context, repository=background_repository),
                    project_id=project_id,
                    window_stride=window_stride,
                    window_width=window_width,
                    encoding=encoding,
                    quality=quality,
                    generate_backgrounds=True,
                    generate_flatfield_profiles=False,
                    flatfield_axis=flatfield_axis,
                ),
                MeanFieldIngestAddon(
                    context=replace(context, repository=flatfield_repository),
                    project_id=project_id,
                    window_stride=resolved_flatfield_stride,
                    window_width=resolved_flatfield_width,
                    encoding=encoding,
                    quality=quality,
                    generate_backgrounds=False,
                    generate_flatfield_profiles=True,
                    flatfield_axis=flatfield_axis,
                ),
            )
            return
        if generate_flatfield_profiles and not generate_backgrounds:
            window_stride = resolved_flatfield_stride
            window_width = resolved_flatfield_width
        self.window_stride = int(window_stride)
        self.window_width = int(window_width)
        if (
            self.window_stride < 1
            or self.window_width < 1
            or self.window_stride % 2 == 0
            or self.window_width % 2 == 0
        ):
            raise ValueError(
                "background_window_stride and background_window_width must be positive odd integers."
            )
        self.encoding = str(encoding)
        self.generate_backgrounds = bool(generate_backgrounds)
        self.generate_flatfield_profiles = bool(generate_flatfield_profiles)
        self.flatfield_axis = int(flatfield_axis)
        if self.flatfield_axis not in {0, 1}:
            raise ValueError("flatfield_axis must be 0 or 1.")
        if not self.generate_backgrounds and not self.generate_flatfield_profiles:
            raise ValueError("At least one ingestion field output must be enabled.")
        self.quality = (
            context.config.processing.frame_storage.image_quality
            if quality is None
            else int(quality)
        )
        self._application_half_width = self.window_stride // 2
        self._source_half_width = self.window_width // 2
        self._nominal_shape: tuple[int, ...] | None = None
        self._source_dtype: np.dtype[Any] | None = None
        self._shape_counts: dict[tuple[int, ...], int] = {}
        self._windows: dict[int, _WindowAccumulator] = {}
        self._outputs: dict[int, dict[str, Any]] = {}
        self._assignments: list[tuple[str, int]] = []
        self._target_centers: set[int] = set()
        self._last_frame_index: int | None = None

    def consume(self, frame_row: dict[str, Any], array: np.ndarray) -> None:
        """Consume one stored frame while its decoded array is still available."""
        if self._delegates is not None:
            for delegate in self._delegates:
                delegate.consume(frame_row, array)
            return
        frame_id = str(frame_row["id"])
        frame_index = int(frame_row["frame_index"])
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("Ingestion background frames must arrive in increasing frame-index order.")
        self._last_frame_index = frame_index

        source = np.asarray(array)
        if source.ndim < 2:
            raise ValueError("Background generation requires frames with at least two dimensions.")
        shape = tuple(int(value) for value in source.shape)
        self._shape_counts[shape] = self._shape_counts.get(shape, 0) + 1
        if self._nominal_shape is None:
            self._nominal_shape = shape
            self._source_dtype = source.dtype

        center = ((frame_index + self._application_half_width) // self.window_stride) * self.window_stride
        self._assignments.append((frame_id, center))
        self._target_centers.add(center)

        background_eligible = shape == self._nominal_shape
        profile_dimension = 1 - self.flatfield_axis
        profile_eligible = (
            source.ndim == 2
            and len(self._nominal_shape) == 2
            and shape[profile_dimension] == self._nominal_shape[profile_dimension]
        )
        if (
            (self.generate_backgrounds and background_eligible)
            or (self.generate_flatfield_profiles and profile_eligible)
        ):
            for source_center in self._source_centers(frame_index):
                accumulator = self._windows.get(source_center)
                if accumulator is None:
                    if np.issubdtype(source.dtype, np.unsignedinteger):
                        accumulator_dtype = np.uint64
                    elif np.issubdtype(source.dtype, np.signedinteger):
                        accumulator_dtype = np.int64
                    else:
                        accumulator_dtype = np.float64
                    accumulator = _WindowAccumulator(
                        total=(
                            np.zeros(self._nominal_shape, dtype=accumulator_dtype)
                            if self.generate_backgrounds
                            else None
                        ),
                        profile_total=(
                            np.zeros(self._nominal_shape[profile_dimension], dtype=np.float64)
                            if self.generate_flatfield_profiles
                            else None
                        ),
                        profile_value_count=0,
                        background_count=0,
                        background_source_frame_ids=[],
                        profile_frame_count=0,
                        profile_source_frame_ids=[],
                    )
                    self._windows[source_center] = accumulator
                with measure_phase("background.ingest_accumulate"):
                    if accumulator.total is not None and background_eligible:
                        accumulator.total += source
                        accumulator.background_count += 1
                        accumulator.background_source_frame_ids.append(frame_id)
                    if accumulator.profile_total is not None and profile_eligible:
                        # Summing is equivalent to stacking on flatfield_axis, but avoids
                        # allocating the potentially large concatenated image.
                        accumulator.profile_total += np.sum(
                            source,
                            axis=self.flatfield_axis,
                            dtype=np.float64,
                        )
                        accumulator.profile_value_count += int(source.shape[self.flatfield_axis])
                        accumulator.profile_frame_count += 1
                        accumulator.profile_source_frame_ids.append(frame_id)

        completed_centers = [
            value
            for value in self._windows
            if value in self._target_centers
            and frame_index > value + self._source_half_width
        ]
        for completed_center in completed_centers:
            self._store_window(completed_center)

    def finalize(self) -> dict[str, Any]:
        """Store remaining windows and batch-assign them to ingested frames."""
        if self._delegates is not None:
            background_result, flatfield_result = (
                delegate.finalize() for delegate in self._delegates
            )
            assert self._assignment_buffers is not None
            merged_assignments: dict[str, dict[str, Any]] = {}
            for assignment_buffer in self._assignment_buffers:
                for assignment in assignment_buffer.assignments:
                    frame_id = str(assignment["frame_id"])
                    merged_assignments.setdefault(frame_id, {"frame_id": frame_id}).update(
                        assignment
                    )
            updater = getattr(
                self.context.repository,
                "update_frame_background_payload_assignments",
                None,
            )
            if not callable(updater):
                raise RuntimeError(
                    "Repository does not support batched ingestion background assignments."
                )
            with measure_phase("background.database_update"):
                rows = updater(
                    list(merged_assignments.values()),
                    project_id=self.project_id,
                )
            return {
                "method": "mean",
                "backgrounds_generated": True,
                "flatfield_profiles_generated": True,
                "updated_frame_count": len(rows),
                "window_count": (
                    int(background_result["window_count"])
                    + int(flatfield_result["window_count"])
                ),
                "stored_window_count": (
                    int(background_result["stored_window_count"])
                    + int(flatfield_result["stored_window_count"])
                ),
                "background_window_count": int(background_result["window_count"]),
                "flatfield_window_count": int(flatfield_result["window_count"]),
                "background": background_result,
                "flatfield": flatfield_result,
            }
        if not self._assignments:
            return {
                "method": "mean",
                "window_stride": self.window_stride,
                "window_width": self.window_width,
                "window_count": 0,
                "stored_window_count": 0,
                "updated_frame_count": 0,
                "skipped_frame_count": 0,
                "skipped_background_frame_count": 0,
                "skipped_flatfield_frame_count": 0,
                "nominal_shape": [],
                "flatfield_axis": self.flatfield_axis,
                "flatfield_profile_length": None,
                "backgrounds_generated": self.generate_backgrounds,
                "flatfield_profiles_generated": self.generate_flatfield_profiles,
            }
        if self._nominal_shape is None:
            raise ValueError("No nominal frame geometry was observed during ingestion.")
        modal_shape = max(
            self._shape_counts,
            key=lambda shape: (self._shape_counts[shape], int(np.prod(shape))),
        )
        if self.generate_backgrounds and modal_shape != self._nominal_shape:
            raise ValueError(
                "The first ingested frame geometry was not the asset's nominal geometry; "
                "background generation cannot safely complete in one pass."
            )

        for center in list(self._windows):
            if center in self._target_centers:
                self._store_window(center)
            else:
                self._windows.pop(center)

        assignments = []
        for frame_id, center in self._assignments:
            output = self._outputs.get(center)
            if output is None:
                raise ValueError(f"No nominal-sized source frames were available for background window {center}.")
            assignments.append({"frame_id": frame_id, **output})

        updater = getattr(self.context.repository, "update_frame_background_payload_assignments", None)
        if not callable(updater):
            raise RuntimeError(
                "Repository does not support batched ingestion background assignments."
            )
        with measure_phase("background.database_update"):
            rows = updater(assignments, project_id=self.project_id)
        return {
            "method": "mean",
            "window_stride": self.window_stride,
            "window_width": self.window_width,
            "window_count": len({center for _, center in self._assignments}),
            "stored_window_count": len(self._outputs),
            "updated_frame_count": len(rows),
            "skipped_frame_count": sum(
                count
                for shape, count in self._shape_counts.items()
                if (
                    shape != self._nominal_shape
                    if self.generate_backgrounds
                    else len(shape) != 2
                    or shape[1 - self.flatfield_axis] != self._nominal_shape[1 - self.flatfield_axis]
                )
            ),
            "skipped_background_frame_count": sum(
                count for shape, count in self._shape_counts.items() if shape != self._nominal_shape
            ),
            "skipped_flatfield_frame_count": sum(
                count
                for shape, count in self._shape_counts.items()
                if len(shape) != 2
                or shape[1 - self.flatfield_axis] != self._nominal_shape[1 - self.flatfield_axis]
            ),
            "nominal_shape": list(self._nominal_shape),
            "flatfield_axis": self.flatfield_axis,
            "flatfield_profile_length": int(self._nominal_shape[1 - self.flatfield_axis]),
            "backgrounds_generated": self.generate_backgrounds,
            "flatfield_profiles_generated": self.generate_flatfield_profiles,
        }

    def _source_centers(self, frame_index: int) -> range:
        first = math.ceil((frame_index - self._source_half_width) / self.window_stride)
        last = math.floor((frame_index + self._source_half_width) / self.window_stride)
        return range(max(0, first) * self.window_stride, (last + 1) * self.window_stride, self.window_stride)

    def _store_window(self, center: int) -> None:
        accumulator = self._windows.pop(center)
        if accumulator.background_count < 1 and accumulator.profile_frame_count < 1:
            return
        assert self._source_dtype is not None
        window_metadata = {
            "method": "mean",
            "window_center": center,
            "window_start": center - self._source_half_width,
            "window_end": center + self._source_half_width,
            "window_stride": self.window_stride,
            "window_width": self.window_width,
            "application_start": center - self._application_half_width,
            "application_end": center + self._application_half_width,
        }
        output: dict[str, Any] = {}
        if accumulator.profile_total is not None and accumulator.profile_frame_count:
            profile = np.ascontiguousarray(
                accumulator.profile_total / accumulator.profile_value_count,
                dtype=np.float32,
            )
            profile_kind = "column_mean" if self.flatfield_axis == 0 else "row_mean"
            output["flatfield_profile"] = profile.tolist()
            output["flatfield_metadata"] = json_ready(
                {
                    **window_metadata,
                    "flatfield_method": profile_kind,
                    "flatfield_axis": self.flatfield_axis,
                    "stack_axis": self.flatfield_axis,
                    "profile_length": int(profile.size),
                    "source_value_count": accumulator.profile_value_count,
                    "source_frame_ids": accumulator.profile_source_frame_ids,
                    "source_frame_count": accumulator.profile_frame_count,
                }
            )
        elif accumulator.profile_total is not None:
            raise ValueError(
                f"No axis-compatible frames contributed to flatfield window {center}."
            )
        if accumulator.total is not None:
            if accumulator.background_count < 1:
                raise ValueError(
                    f"No nominal-sized frames contributed to background window {center}."
                )
            with measure_phase("background.ingest_finalize"):
                mean = np.rint(accumulator.total / accumulator.background_count)
                if np.issubdtype(self._source_dtype, np.integer):
                    bounds = np.iinfo(self._source_dtype)
                    mean = np.clip(mean, bounds.min, bounds.max)
                background = np.ascontiguousarray(mean.astype(self._source_dtype))
            with measure_phase("background.encode"):
                payload, payload_encoding, payload_format = encode_array_payload(
                    background,
                    self.encoding,
                    quality=self.quality,
                )
            kvstore = self.context.kvstore_for_project(self.project_id)
            if kvstore is None:
                raise RuntimeError("A project KVStore is required to store ingestion backgrounds.")
            with measure_phase("background.kvstore_write"):
                payload_ref = kvstore.put_store(payload)
            metadata = json_ready(
                {
                    **window_metadata,
                    "frame_variant": "background",
                    "background_method": "mean",
                    "background_source_payload_kind": "ingestion_decoded",
                    "background_source_frame_ids": accumulator.background_source_frame_ids,
                    "background_source_frame_count": accumulator.background_count,
                    "background_layout": "nominal_frame",
                    "background_window_center": center,
                    "background_window_start": center - self._source_half_width,
                    "background_window_end": center + self._source_half_width,
                    "background_window_stride": self.window_stride,
                    "background_window_width": self.window_width,
                    "background_application_start": center - self._application_half_width,
                    "background_application_end": center + self._application_half_width,
                    "kvstore_key": payload_ref,
                    "kvstore_hash": payload_ref,
                    "kvstore_encoding": payload_encoding,
                    "kvstore_format": payload_format,
                    "kvstore_quality": self.quality,
                    "dtype": str(background.dtype),
                    "shape": list(background.shape),
                }
            )
            output.update(
                {
                    "kvstore_hash": payload_ref,
                    "payload_ref": payload_ref,
                    "payload_encoding": payload_encoding,
                    "payload_format": payload_format,
                    "payload_dtype": str(background.dtype),
                    "payload_shape": list(background.shape),
                    "metadata": metadata,
                }
            )
        self._outputs[center] = output
