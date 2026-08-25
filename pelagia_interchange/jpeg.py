"""Small standard-library wrapper around libjpeg-turbo's legacy compressor API.

The wrapper intentionally uses the stable ``tjCompress2`` API, which is
available in libjpeg-turbo 2.x and 3.x.  It does not use the newer 3.x ``tj3``
API, so a single encoder configuration works with either supported major
release.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
from os import PathLike
from typing import Literal, TypeAlias


PixelFormat: TypeAlias = Literal["gray", "rgb"]
"""Packed eight-bit input pixel formats accepted by :class:`TurboJPEGEncoder`."""

Subsampling: TypeAlias = Literal["gray", "444", "422", "420"]
"""JPEG chroma-subsampling modes accepted by :class:`TurboJPEGEncoder`."""

PixelBuffer: TypeAlias = bytes | bytearray | memoryview
"""Common contiguous byte-buffer types accepted as source pixels."""


class TurboJPEGError(RuntimeError):
    """A libjpeg-turbo operation could not be completed."""


class TurboJPEGUnavailableError(TurboJPEGError):
    """libjpeg-turbo could not be found or does not expose the legacy API."""


# Values from the public turbojpeg.h enums.  These enum values have remained
# stable across the 2.x and 3.x releases supported by this module.
_TJPF_RGB = 0
_TJPF_GRAY = 6
_TJSAMP_444 = 0
_TJSAMP_422 = 1
_TJSAMP_420 = 2
_TJSAMP_GRAY = 3
_TJFLAG_NOREALLOC = 1024

_UByte = ctypes.c_ubyte
_UBytePointer = ctypes.POINTER(_UByte)
_LibraryPath = str | PathLike[str]

_PIXEL_FORMATS: dict[PixelFormat, tuple[int, int]] = {
    "gray": (_TJPF_GRAY, 1),
    "rgb": (_TJPF_RGB, 3),
}
_SUBSAMPLING: dict[Subsampling, int] = {
    "444": _TJSAMP_444,
    "422": _TJSAMP_422,
    "420": _TJSAMP_420,
    "gray": _TJSAMP_GRAY,
}


def _candidate_library_names(library_path: _LibraryPath | None) -> list[str]:
    """Return ordered library candidates without attempting to load them."""
    if library_path is not None:
        return [os.fspath(library_path)]

    candidates: list[str] = []
    environment_path = os.environ.get("TURBOJPEG_LIBRARY")
    if environment_path:
        candidates.append(environment_path)
    found = ctypes.util.find_library("turbojpeg")
    if found:
        candidates.append(found)

    if sys.platform == "darwin":
        candidates.extend((
            "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib",
            "/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib",
            "libturbojpeg.dylib",
        ))
    elif os.name == "nt":
        candidates.extend(("turbojpeg.dll", "libturbojpeg.dll"))
    else:
        candidates.extend(("libturbojpeg.so.0", "libturbojpeg.so"))
    return list(dict.fromkeys(candidates))


def find_turbojpeg_library(library_path: _LibraryPath | None = None) -> str | None:
    """Return the first library path or name exposing the required legacy API.

    ``library_path`` takes precedence over automatic discovery.  With no
    explicit path, ``TURBOJPEG_LIBRARY`` is tried before platform conventions.
    The returned value is suitable for passing to :class:`TurboJPEGEncoder`.
    """
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    for candidate in _candidate_library_names(library_path):
        try:
            library = loader(candidate)
        except OSError:
            continue
        if all(hasattr(library, symbol) for symbol in
               ("tjInitCompress", "tjCompress2", "tjBufSize", "tjDestroy")):
            return candidate
    return None


def turbojpeg_available(library_path: _LibraryPath | None = None) -> bool:
    """Return whether a loadable library exposes the required legacy symbols."""
    return find_turbojpeg_library(library_path) is not None


def turbojpeg_api_version(library_path: _LibraryPath | None = None) -> int | None:
    """Return the detected TurboJPEG API generation (2 or 3), if available.

    The legacy API does not expose an exact runtime package-version function.
    This helper therefore reports the newest API generation indicated by its
    exported symbols: ``tjGetErrorStr2`` identifies 2.x+, and ``tj3Init``
    identifies 3.x.  It returns ``None`` when no compatible library is found.
    """
    candidate = find_turbojpeg_library(library_path)
    if candidate is None:
        return None
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    try:
        library = loader(candidate)
    except OSError:
        return None
    if hasattr(library, "tj3Init"):
        return 3
    if hasattr(library, "tjGetErrorStr2"):
        return 2
    return None


class _TurboJPEGLibrary:
    """Typed bindings for the subset of the legacy compressor API we use."""

    def __init__(self, library_path: _LibraryPath | None) -> None:
        candidate = find_turbojpeg_library(library_path)
        if candidate is None:
            requested = f" at {os.fspath(library_path)!r}" if library_path is not None else ""
            raise TurboJPEGUnavailableError(
                "libjpeg-turbo was not found" + requested + "; install libjpeg-turbo "
                "or pass library_path (or set TURBOJPEG_LIBRARY)."
            )

        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        try:
            self.library = loader(candidate)
        except OSError as error:
            raise TurboJPEGUnavailableError(
                f"Could not load libjpeg-turbo from {candidate!r}: {error}"
            ) from error
        self.path = candidate
        try:
            self.init_compress = self.library.tjInitCompress
            self.compress2 = self.library.tjCompress2
            self.buf_size = self.library.tjBufSize
            self.destroy = self.library.tjDestroy
        except AttributeError as error:
            raise TurboJPEGUnavailableError(
                f"{candidate!r} does not expose the libjpeg-turbo 2.x/3.x legacy compressor API"
            ) from error

        self.init_compress.argtypes = []
        self.init_compress.restype = ctypes.c_void_p
        self.compress2.argtypes = [
            ctypes.c_void_p, _UBytePointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(_UBytePointer), ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.compress2.restype = ctypes.c_int
        self.buf_size.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.buf_size.restype = ctypes.c_ulong
        self.destroy.argtypes = [ctypes.c_void_p]
        self.destroy.restype = ctypes.c_int

        self.error_string2 = getattr(self.library, "tjGetErrorStr2", None)
        if self.error_string2 is not None:
            self.error_string2.argtypes = [ctypes.c_void_p]
            self.error_string2.restype = ctypes.c_char_p
        self.error_string = getattr(self.library, "tjGetErrorStr", None)
        if self.error_string is not None:
            self.error_string.argtypes = []
            self.error_string.restype = ctypes.c_char_p

    def error_message(self, handle: ctypes.c_void_p | None) -> str:
        """Return the most precise error string the loaded ABI supports."""
        message: bytes | None = None
        if self.error_string2 is not None:
            message = self.error_string2(handle)
        elif self.error_string is not None:
            message = self.error_string()
        return message.decode("utf-8", "replace") if message else "unknown libjpeg-turbo error"


class TurboJPEGEncoder:
    """Encode packed GRAY or RGB frames using one libjpeg-turbo handle.

    Instances are context-manageable and own exactly one ``tjInitCompress``
    handle.  TurboJPEG handles must not be used concurrently; create one
    encoder per worker thread, or use :func:`thread_local_encoder`.
    """

    def __init__(self, library_path: _LibraryPath | None = None) -> None:
        """Initialize an encoder, optionally loading an explicit library path."""
        self._bindings = _TurboJPEGLibrary(library_path)
        handle = self._bindings.init_compress()
        if not handle:
            raise TurboJPEGError(f"tjInitCompress failed: {self._bindings.error_message(None)}")
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p(handle)

    @property
    def library_path(self) -> str:
        """Return the path or loader name used for the loaded library."""
        return self._bindings.path

    @property
    def closed(self) -> bool:
        """Return whether this encoder's native handle has been destroyed."""
        return self._handle is None

    def __enter__(self) -> TurboJPEGEncoder:
        """Return this open encoder for use in a ``with`` statement."""
        self._require_handle()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: object) -> None:
        """Destroy the native handle when its context exits."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup for callers that did not use :meth:`close`."""
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Destroy the native compressor handle; this method is idempotent."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        if self._bindings.destroy(handle) != 0:
            raise TurboJPEGError(f"tjDestroy failed: {self._bindings.error_message(handle)}")

    def encode(self, pixels: PixelBuffer, width: int, height: int, *,
               pixel_format: PixelFormat, quality: int = 90,
               subsampling: Subsampling | None = None, pitch: int | None = None) -> bytes:
        """Encode an eight-bit packed GRAY or RGB frame to JPEG bytes.

        ``pitch`` is the byte distance between input rows; it defaults to a
        tightly packed row.  GRAY input must use grayscale output.  RGB input
        defaults to 4:4:4 and may use any supported subsampling mode.
        """
        handle = self._require_handle()
        if width <= 0 or height <= 0:
            raise ValueError("width and height must both be positive")
        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100")
        if pixel_format not in _PIXEL_FORMATS:
            raise ValueError("pixel_format must be 'gray' or 'rgb'")

        pixel_value, bytes_per_pixel = _PIXEL_FORMATS[pixel_format]
        resolved_subsampling: Subsampling = (
            "gray" if pixel_format == "gray" else "444"
        ) if subsampling is None else subsampling
        if resolved_subsampling not in _SUBSAMPLING:
            raise ValueError("subsampling must be 'gray', '444', '422', or '420'")
        if pixel_format == "gray" and resolved_subsampling != "gray":
            raise ValueError("GRAY input requires gray subsampling")

        row_bytes = width * bytes_per_pixel
        actual_pitch = row_bytes if pitch is None else pitch
        if actual_pitch < row_bytes:
            raise ValueError(f"pitch must be at least {row_bytes} bytes for each row")
        required_bytes = (height - 1) * actual_pitch + row_bytes
        source, source_pointer = _source_buffer(pixels, required_bytes)

        subsampling_value = _SUBSAMPLING[resolved_subsampling]
        capacity = self._bindings.buf_size(width, height, subsampling_value)
        if capacity == 0:
            raise ValueError("image dimensions or subsampling are outside libjpeg-turbo limits")
        output = (_UByte * capacity)()
        output_pointer = ctypes.cast(output, _UBytePointer)
        jpeg_pointer = output_pointer
        jpeg_size = ctypes.c_ulong(capacity)
        result = self._bindings.compress2(
            handle, source_pointer, width, actual_pitch, height, pixel_value,
            ctypes.byref(jpeg_pointer), ctypes.byref(jpeg_size), subsampling_value, quality,
            _TJFLAG_NOREALLOC,
        )
        # ``source`` owns any input copy or exported buffer and must remain
        # alive until tjCompress2 has returned.
        del source
        if result != 0:
            raise TurboJPEGError(f"tjCompress2 failed: {self._bindings.error_message(handle)}")
        if ctypes.cast(jpeg_pointer, ctypes.c_void_p).value != ctypes.cast(output_pointer, ctypes.c_void_p).value:
            raise TurboJPEGError("tjCompress2 replaced the caller-owned output buffer despite TJFLAG_NOREALLOC")
        if jpeg_size.value > capacity:
            raise TurboJPEGError("tjCompress2 returned an output size larger than tjBufSize")
        return bytes(output[:jpeg_size.value])

    def encode_gray(self, pixels: PixelBuffer, width: int, height: int, *, quality: int = 90,
                    pitch: int | None = None) -> bytes:
        """Encode an eight-bit grayscale frame using grayscale JPEG output."""
        return self.encode(pixels, width, height, pixel_format="gray", quality=quality,
                           subsampling="gray", pitch=pitch)

    def encode_rgb(self, pixels: PixelBuffer, width: int, height: int, *, quality: int = 90,
                   subsampling: Subsampling = "444", pitch: int | None = None) -> bytes:
        """Encode an eight-bit RGB frame with the requested JPEG subsampling."""
        return self.encode(pixels, width, height, pixel_format="rgb", quality=quality,
                           subsampling=subsampling, pitch=pitch)

    def _require_handle(self) -> ctypes.c_void_p:
        """Return the live handle or raise a clear lifecycle error."""
        if self._handle is None:
            raise TurboJPEGError("TurboJPEGEncoder is closed")
        return self._handle


def _source_buffer(pixels: PixelBuffer, required_bytes: int) -> tuple[object, _UBytePointer]:
    """Return a contiguous byte pointer while retaining its backing object."""
    try:
        view = memoryview(pixels)
        byte_view = view.cast("B")
    except TypeError as error:
        raise TypeError("pixels must provide a contiguous bytes-like buffer") from error
    if not byte_view.contiguous:
        raise ValueError("pixels must provide a contiguous bytes-like buffer")
    if byte_view.nbytes < required_bytes:
        raise ValueError(f"pixels contains {byte_view.nbytes} bytes; at least {required_bytes} are required")
    if byte_view.readonly:
        copied = ctypes.create_string_buffer(byte_view.tobytes())
        return copied, ctypes.cast(copied, _UBytePointer)
    array = (_UByte * byte_view.nbytes).from_buffer(byte_view)
    return (byte_view, array), ctypes.cast(array, _UBytePointer)


_thread_local = threading.local()


def thread_local_encoder(library_path: _LibraryPath | None = None) -> TurboJPEGEncoder:
    """Return the calling thread's reusable encoder for ``library_path``.

    A separate native handle is created in each thread.  Call
    :func:`close_thread_local_encoder` before a long-lived thread shuts down
    when deterministic cleanup is needed.
    """
    key = os.fspath(library_path) if library_path is not None else None
    encoders: dict[str | None, TurboJPEGEncoder] = getattr(_thread_local, "encoders", {})
    encoder = encoders.get(key)
    if encoder is None or encoder.closed:
        encoder = TurboJPEGEncoder(library_path)
        encoders[key] = encoder
        _thread_local.encoders = encoders
    return encoder


def close_thread_local_encoder(library_path: _LibraryPath | None = None) -> None:
    """Close and remove the calling thread's cached encoder, if present."""
    key = os.fspath(library_path) if library_path is not None else None
    encoders: dict[str | None, TurboJPEGEncoder] = getattr(_thread_local, "encoders", {})
    encoder = encoders.pop(key, None)
    if encoder is not None:
        encoder.close()


__all__ = [
    "PixelBuffer", "PixelFormat", "Subsampling", "TurboJPEGEncoder", "TurboJPEGError",
    "TurboJPEGUnavailableError", "close_thread_local_encoder", "find_turbojpeg_library",
    "thread_local_encoder", "turbojpeg_api_version", "turbojpeg_available",
]
