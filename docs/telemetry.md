# Telemetry ingestion and lookup

Pelagia imports operational measurements as project- and run-scoped telemetry.
The original delimited bytes are preserved in the selected project's
content-addressed KVStore before parsing. Their payload key, filename, resolved
operator path, SHA-256 checksum, byte size, parser identity, mapping metadata,
and converted observations are recorded together. Importing a corrected file is
a new source; do not alter an already imported source in place.

## Time basis

`timestamptz` is the canonical stored time for observations and timeline events.
Pelagia normalizes every input timestamp to an aware UTC instant before writing.
Use ISO 8601 values with `Z` or a numeric offset whenever the source provides
one (for example, `2026-08-21T18:04:05.125Z`). Naive ISO values and custom
format values are interpreted in the mapping's `source_timezone`, which defaults
to `UTC`. Unix timestamps are interpreted as UTC.

The database representation does not claim a source clock precision or a
synchronization method. Preserve those facts, if known, in mapping metadata or
source metadata instead of inferring them from the `timestamptz` type.

## CSV mapping configuration

Pelagia exposes `POST /telemetry/analyze` for UI clients that need to inspect a
server-side CSV before importing it. The response includes detected headers,
bounded preview rows, normalized UTC timestamps, sampling diagnostics, and
numeric-column quality counts. `GET /telemetry/catalog` exposes the versioned
unit registry, valid affine conversion metadata, interpolation methods, and
project-local parameter and sensor records. Clients should treat inferred
parameter names and units as suggestions requiring operator confirmation.
`POST /runs` can create a project-scoped empty run when telemetry arrives
before image assets; the telemetry import route verifies that the selected run
belongs to the active project before queueing work.

Use `import-telemetry` with a JSON mapping kept alongside the acquisition data
or in version control. The mapping mirrors the ingestion service contract:

```json
{
  "timestamp_column": "time_utc",
  "timestamp_format": "iso8601",
  "source_timezone": "UTC",
  "delimiter": ",",
  "parser_name": "shipboard.logger",
  "parser_version": "2.4",
  "metadata": {"clock_source": "navigation computer"},
  "streams": [
    {
      "column": "temperature_c",
      "stream_key": "ctd.temperature",
      "sensor_key": "ctd-01",
      "parameter_key": "temperature",
      "native_unit": "degC",
      "canonical_unit": "degC",
      "display_name": "Water temperature",
      "standard_name": "sea_water_temperature",
      "manufacturer": "Sea-Bird Scientific",
      "model": "SBE 45",
      "serial_number": "CTD-01",
      "qc_column": "temperature_qc",
      "interpolation": "linear",
      "max_gap_seconds": 30,
      "priority": 100,
      "is_default": true,
      "metadata": {"excluded_qc_flags": [3, 4]},
      "sensor_metadata": {"mount": "shipboard flow-through"},
      "parameter_metadata": {"instrument_role": "environmental context"}
    }
  ]
}
```

Required fields are `timestamp_column` and, for each stream, `column`,
`stream_key`, `sensor_key`, `parameter_key`, `native_unit`, and `canonical_unit`.
`native_unit` and `canonical_unit` are validated by Pelagia's explicit,
dependency-free telemetry unit registry. Unit aliases are accepted in mappings
and normalized before storage; the originally declared names remain in stream
provenance. The registry supports temperature (`K`, `degC`, `degF`), pressure
(`Pa`, `hPa`, `kPa`, `bar`, `dbar`), length (`m`, `cm`, `mm`, `km`, `ft`), speed
(`m/s`, `km/h`, `kn`), conductivity (`S/m`, `mS/cm`, `uS/cm`), voltage (`V`,
`mV`), fraction (`1`, `%`), practical salinity (`PSU`), turbidity (`NTU`),
counts, and the listed mass/amount concentrations (`mg/L`, `ug/L`, `g/m3`,
`umol/L`, `mmol/m3`).

`scale` and `offset` are required to describe the registry conversion exactly:
`canonical_value = raw_value * scale + offset`. Pelagia validates the affine
pair, including temperature offsets, before any observations are written. For
example, `degF` to `degC` requires `scale: 0.5555555555555556` and
`offset: -17.77777777777778`; `degC` to `K` requires `scale: 1.0` and
`offset: 273.15`. Mappings that name unsupported or dimensionally incompatible
units, or that supply a different scale/offset, fail instead of applying an
unverified conversion. The recorded stream metadata retains the declared units,
normalized units, registry version, scale, and offset.

Each source stream must have unique, ascending timestamps. Pelagia rejects
out-of-order input rather than silently rewriting native stream order. Blank measurements are skipped;
non-numeric, non-finite, malformed timestamp, malformed QC, or missing required
column values fail the import rather than producing a partial source.

Use `timestamp_format` values `iso8601`, `unix_seconds`, `unix_milliseconds`, or
a Python `strptime` format. `delimiter` must be exactly one character. If a
parameter has more than one stream in the import, exactly one must set
`is_default: true`; it is the stream used by ordinary parameter lookup.

```bash
pelagia import-telemetry shipboard.csv RUN_UUID \
  --mapping telemetry-mapping.json --project-key survey-2026 \
  --collections cruise-17,calibrated
```

The command prints a JSON result containing the created source asset, source,
and streams. Project selection defaults to the configured development project;
pass `--project-key` in scripts to make the target unambiguous.

HTTP imports accept only files beneath `file_browser.root_path_import_dir` or
an explicitly configured allowed root. The CLI is an operator-local surface and
can read the path supplied by the operator. Parsing and transactional `COPY` use
the immutable KVStore snapshot, so later changes at the operator path cannot
change an import already in progress. Retrying the same source bytes with the
same mapping and collections returns the existing project- and run-scoped
import instead of duplicating observations.
`POST /runs/{run_id}/telemetry/import` queues that work and returns `202` with a
job record. Include `parser_name` and `parser_version` in its JSON body to retain
the parser identity from the mapping; they default to `pelagia.delimited` and
`1` when omitted.

## Lookup, interpolation, gaps, and QC

Lookup returns one selected stream per requested parameter, including the
method, source observation time(s), QC flags, and a reason when no value can be
resolved.

Native observations are available through the paginated
`GET /runs/{run_id}/telemetry/observations?stream_id=...` route. Use optional
`start_at`, `end_at`, `limit`, and `offset` parameters for bounded inspection or
export; timestamps must include an explicit UTC offset.

```bash
pelagia lookup-telemetry RUN_UUID \
  --observed-at 2026-08-21T18:04:05.125Z \
  --parameters temperature,longitude --project-key survey-2026
```

`none` only returns an exact observation. `nearest` selects the closest usable
observation, `previous` selects the usable observation at or before the target,
and `linear` interpolates between usable bracketing observations. `linear` uses
the shortest path across the antimeridian for the `longitude` parameter. All
interpolating methods require a positive `max_gap_seconds` in the mapping.
For `nearest` and `previous`, the gap is the target-to-observation distance; for
`linear`, it is the full span between the bracketing observations. A value beyond
that bound returns `gap_exceeded`, not an extrapolated value.

Add `excluded_qc_flags` to stream metadata to exclude flag values during lookup.
An exact excluded observation returns `qc_excluded`; an interpolation whose
required neighbor is excluded also returns `qc_excluded`. Other explicit missing
outcomes include `interpolation_disabled`, `outside_stream_range`, and
`stream_not_found`. Preserve the raw QC flag meanings in metadata or your source
documentation: Pelagia stores the flag but does not impose a universal QC scale.

Circular quantities must declare their interpolation domain explicitly in stream
metadata. For longitude represented on `[-180, 180)`, add
`"circular_period": 360.0` and `"circular_minimum": -180.0`. Circular handling
does not depend on a particular project-defined parameter key.

## Timeline events

Timeline event types are project-scoped vocabulary identified by
`event_type_key`. Events are run-scoped records with a required `start_at`, an
optional inclusive `end_at`, optional telemetry `source_id`, optional text `value`,
and metadata. Both times use
the same canonical UTC handling as telemetry; an `end_at` earlier than `start_at`
is invalid. Point events omit `end_at`; interval events apply at instants from
their start through their end. A point event matches its exact recorded instant;
omitting `end_at` does not create an indefinitely open interval. Event types and events are managed through the
telemetry HTTP routes, and frame context includes events active at the frame's
capture timestamp. `GET /frames/{frame_id}/context` exposes optional
`telemetry` and `events` fields when `include_telemetry=true` and/or
`include_events=true`; the generated OpenAPI schema documents both fields.
ROI Browser and Curation use that frame context for the selected-ROI
inspector. Both ROI selection endpoints also accept repeated
`telemetry_filter` JSON query values with a `parameter_key` and inclusive
`min_value`/`max_value`; filters are resolved using the selected run's
default/priority stream and its declared interpolation and maximum-gap policy
before pagination.
Event writes record the authenticated user as `created_by`;
clients do not supply that field. List events with optional inclusive `start_at`
and `end_at` bounds (which return events overlapping that range), retrieve one
event by ID, and use `PATCH` or `DELETE` on `/runs/{run_id}/events/{event_id}`.
All event query and body timestamps must include `Z` or a numeric UTC offset.
