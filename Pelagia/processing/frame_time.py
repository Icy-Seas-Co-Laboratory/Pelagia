import re
from datetime import datetime, timedelta, timezone


_FILENAME_TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[ T-](?P<hour>\d{2})[-:](?P<minute>\d{2})[-:](?P<second>\d{2})(?:\.(?P<fraction>\d{1,6}))?"
)


def parse_filename_timestamp_utc(filename: str) -> datetime | None:
    match = _FILENAME_TIMESTAMP_RE.search(filename)
    if match is None:
        return None

    year, month, day = (int(part) for part in match.group("date").split("-"))
    fraction = match.group("fraction") or "0"
    return datetime(
        year,
        month,
        day,
        int(match.group("hour")),
        int(match.group("minute")),
        int(match.group("second")),
        int(fraction.ljust(6, "0")),
        tzinfo=timezone.utc,
    )


def timestamp_for_frame(start: datetime | None, fps: float, frame_number: int) -> datetime | None:
    if start is None or fps <= 0:
        return None
    return start + timedelta(seconds=(frame_number - 1) / fps)


_parse_filename_timestamp_utc = parse_filename_timestamp_utc
_timestamp_for_frame = timestamp_for_frame
