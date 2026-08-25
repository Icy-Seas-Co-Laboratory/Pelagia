"""Bounded, ordered video-frame encoding pipeline.

The pipeline deliberately has many encoder threads but exactly one consumer of
``DatasetBuilder``.  SQLite permits one writer at a time, and keeping that
ownership explicit also preserves deterministic frame order and shard rollover.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterable

from .builder import DatasetBuilder
from .jpeg import TurboJPEGEncoder
from .models import FrameRecord, HashRecord, SourceFile, StorageFormat
from .util import hash_bytes


class FramePipelineError(RuntimeError):
    """A frame producer, encoder, or ordered writer failed."""


@dataclass(frozen=True, slots=True)
class FrameTask:
    frame_id: int
    source_frame_number: int
    pixels: bytes


@dataclass(frozen=True, slots=True)
class PreviewCandidate:
    frame_id: int
    source_frame_number: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class _EncodedFrame:
    frame_id: int
    source_frame_number: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class _Failure:
    error: BaseException


_STOP = object()


def _put_bounded(target: queue.Queue[object], value: object, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            target.put(value, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


class OrderedFramePipeline:
    """Encode raw frames concurrently and add them to one SQLite writer in order."""

    def __init__(
        self,
        *,
        builder: DatasetBuilder,
        stream: str,
        source: SourceFile,
        storage_format: StorageFormat,
        width: int,
        height: int,
        pixel_format: str,
        jpeg_quality: int,
        jpeg_subsampling: str,
        workers: int,
        queue_depth: int,
        library_path: str | None = None,
        preview_frame_ids: set[int] | None = None,
        timestamp_source: str = "unknown",
    ) -> None:
        if workers < 1:
            raise ValueError("jpeg workers must be positive")
        if queue_depth < 1:
            raise ValueError("queue depth must be positive")
        self.builder = builder
        self.stream = stream
        self.source = source
        self.storage_format = storage_format
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.jpeg_quality = jpeg_quality
        self.jpeg_subsampling = jpeg_subsampling
        self.workers = workers
        self.library_path = library_path
        self.preview_frame_ids = preview_frame_ids or set()
        self.timestamp_source = timestamp_source
        self._tasks: queue.Queue[object] = queue.Queue(maxsize=queue_depth)
        self._results: queue.Queue[object] = queue.Queue(maxsize=queue_depth)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._next_source_frame: int | None = None
        self._pending_frames: dict[int, _EncodedFrame] = {}
        self.frames_written = 0
        self.preview_candidates: list[PreviewCandidate] = []

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def _record_failure(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error
        self._stop.set()

    def start(self, *, first_source_frame: int) -> None:
        self._next_source_frame = first_source_frame
        for number in range(self.workers):
            thread = threading.Thread(target=self._encode, name=f"pelagia-jpeg-{number}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def submit(self, task: FrameTask) -> None:
        if self._failure is not None:
            raise FramePipelineError("frame pipeline failed") from self._failure
        self._drain_ready()
        while True:
            try:
                self._tasks.put(task, timeout=0.05)
                break
            except queue.Full:
                self._consume_one()
        self._drain_ready()

    def close_input(self) -> None:
        for _ in self._threads:
            while True:
                try:
                    self._tasks.put(_STOP, timeout=0.05)
                    break
                except queue.Full:
                    self._consume_one()
        while any(thread.is_alive() for thread in self._threads):
            try:
                self._consume_one(block=False)
            except queue.Empty:
                # A worker can exit after the liveness check and before a
                # blocking dequeue.  Use a short join instead of waiting on a
                # result that will never arrive.
                threading.Event().wait(0.01)
            for thread in self._threads:
                thread.join(timeout=0.01)
        self._drain_ready()
        if self._next_source_frame is None:
            raise FramePipelineError("frame pipeline was not started")
        if self._pending_frames:
            raise FramePipelineError("encoded frames were not contiguous at pipeline shutdown")
        if self._failure is not None:
            raise FramePipelineError("frame pipeline failed") from self._failure

    def abort(self) -> None:
        self._stop.set()
        for _ in self._threads:
            try:
                self._tasks.put_nowait(_STOP)
            except queue.Full:
                break
        for thread in self._threads:
            thread.join(timeout=1)

    def _encode(self) -> None:
        encoder: TurboJPEGEncoder | None = None
        try:
            encoder = TurboJPEGEncoder(library_path=self.library_path)
            while not self._stop.is_set():
                try:
                    item = self._tasks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _STOP:
                    return
                assert isinstance(item, FrameTask)
                payload = encoder.encode(item.pixels, width=self.width, height=self.height,
                                         pixel_format=self.pixel_format, quality=self.jpeg_quality,
                                         subsampling=self.jpeg_subsampling)
                if not _put_bounded(self._results, _EncodedFrame(item.frame_id, item.source_frame_number, payload), self._stop):
                    return
        except BaseException as exc:
            self._record_failure(exc)
            _put_bounded(self._results, _Failure(exc), threading.Event())
        finally:
            if encoder is not None:
                encoder.close()

    def _drain_ready(self) -> None:
        while True:
            try:
                self._consume_one(block=False)
            except queue.Empty:
                return

    def _consume_one(self, *, block: bool = True) -> None:
        item = self._results.get() if block else self._results.get_nowait()
        if isinstance(item, _Failure):
            self._record_failure(item.error)
            raise FramePipelineError("JPEG worker failed") from item.error
        assert isinstance(item, _EncodedFrame)
        if item.source_frame_number in self._pending_frames:
            error = FramePipelineError(f"duplicate encoded source frame {item.source_frame_number}")
            self._record_failure(error)
            raise error
        self._pending_frames[item.source_frame_number] = item
        while self._next_source_frame in self._pending_frames:
            frame = self._pending_frames.pop(self._next_source_frame)
            blob_hash = HashRecord("sha256", "stored_blob", hash_bytes(frame.payload))
            self.builder.add_frame(stream=self.stream, source_file=self.source, frame_id=frame.frame_id,
                                   source_frame_number=frame.source_frame_number, encoded_bytes=frame.payload,
                                   storage_format=self.storage_format, width=self.width, height=self.height,
                                   timestamp_source=self.timestamp_source, blob_hash=blob_hash)
            if frame.frame_id in self.preview_frame_ids:
                self.preview_candidates.append(PreviewCandidate(frame.frame_id, frame.source_frame_number, frame.payload))
            self.frames_written += 1
            assert self._next_source_frame is not None
            self._next_source_frame += 1
