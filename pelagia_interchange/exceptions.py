"""User-facing exception hierarchy."""


class InterchangeError(Exception):
    """Base exception for the interchange package."""


class FormatError(InterchangeError):
    """A package or shard does not conform to the format."""


class CompatibilityError(FormatError):
    """The reader cannot safely interpret a format or schema version."""


class IntegrityError(InterchangeError):
    """Stored bytes do not match their declared integrity metadata."""


class DatasetStateError(InterchangeError):
    """An operation is invalid for the dataset lifecycle state."""


class FrameNotFoundError(InterchangeError, KeyError):
    """No frame matched the requested identity."""


class UnsafePathError(InterchangeError):
    """A path could escape its intended package or output root."""

