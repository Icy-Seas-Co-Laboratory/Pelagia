# Scientific Image Interchange Format 1.0

Status: normative specification. The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY express requirement levels. Implementation advice is explicitly marked non-normative.

## 1. Scope and authority

The format is a portable preservation and interchange package for retained scientific images. It is not an application database. Authority is divided as follows:

| Component | Authority |
|---|---|
| `data/*.sqlite` | retained encoded payloads and frame-level/source provenance |
| `manifest.json` | physical inventory, identities, shard index, structural provenance |
| `metadata.toml` | scientific and collection description |
| `history.jsonl` | append-only processing provenance |
| `checksums.sha256` | package-file integrity |
| `README.md` | durable human guidance; non-normative |

Where resilient duplication occurs, validators MUST compare the copies. Shard metadata is authoritative for shard-local facts; the manifest index is authoritative for package membership.

## 2. Package structure

A package MUST be a directory containing `manifest.json`, `metadata.toml`, `history.jsonl`, `README.md`, `checksums.sha256`, and `data/`. It MAY contain `calibration/`, `preview/`, and `tools/`. Package resource paths MUST be relative POSIX paths, MUST NOT contain `..`, and MUST NOT be required to resolve an original absolute acquisition path.

`calibration/` contains ordinary files with manifest inventory records and descriptions. No calibration file format is mandated. Metadata SHOULD identify its semantic role (camera, lens distortion, pixel scale, field of view, illumination, or clock). `preview/` contains non-authoritative derivatives; its manifest records MUST identify them as such. Automatically selected previews SHOULD use a deterministic, documented, bounded selection method and SHOULD include an index mapping each derivative to its authoritative stream UUID, retained frame ID, source-file UUID, and source frame number. Preview generation or failure MUST NOT alter authoritative frame payloads.

## 3. Identifiers and names

Dataset, acquisition, deployment, instrument, stream, source-file, and shard identities MUST use UUIDs. Human-readable names MUST NOT be interpreted as persistent identity. Shard filenames are deterministic locators, not identity. The reference naming pattern is `{stream_slug}_{sequence:06d}.sqlite`.

## 4. Versions and compatibility

`format_version` uses `MAJOR.MINOR`. Readers MUST reject a greater major version and SHOULD accept greater minor versions while preserving unknown fields. `schema_version` versions concrete JSON/TOML/SQLite schemas and is currently `1`. Library semantic versions are independent of format versions. Writers MUST NOT discard unknown manifest fields or `[extensions.*]` metadata during a read/write round trip.

## 5. Manifest schema

The UTF-8 JSON root MUST contain `format`, `format_version`, `schema_version`, `dataset_uuid`, `created_at`, `state`, `shards`, `source_files`, `software`, and `validation`. State is one of `building`, `finalizing`, `complete`, `verified`, or `modified`.

Each shard record MUST contain `shard_uuid`, `relative_path`, `byte_size`, a `file_hash` with `algorithm`, `target=shard_file`, and `value`, `stream_uuid`, `stream_name`, `first_frame`, `last_frame`, `frame_count`, `first_timestamp`, and `last_timestamp`. It SHOULD include `encoded_bytes`.

Each source record MUST contain `source_file_id`, `source_uuid`, `original_filename`, and fields when known for original relative/absolute path, byte size, typed file hash, container, codec, pixel format, dimensions, rational frame rate, expected frame count, and start/end dates. Filenames MUST NOT be assumed unique. Absolute paths are provenance only.

The manifest MAY inventory `calibration` and `previews`. Unknown root fields are compatible extensions unless a future major version says otherwise.

## 6. SQLite schema and behavior

Each shard MUST be an ordinary standalone SQLite 3 file with tables `frames`, `source_files`, `storage_formats`, and `shard_metadata`. It MUST pass `PRAGMA integrity_check`, have no required WAL/journal sidecars when finalized, and use no nonstandard SQLite extension.

`frames.frame_id` is an INTEGER PRIMARY KEY. Rows also contain source identity/number, timestamp and timestamp-provenance fields, normalized storage ID, nullable BLOB, byte size, dimensions, typed stored-BLOB and optional decoded-pixel hashes, and status. `source_file_id, source_frame_number` MUST be unique within a shard. `byte_size` MUST equal the exact BLOB length or zero for null.

Allowed status values are `valid`, `missing`, `decode_failed`, `duplicate`, `intentionally_removed`, `timestamp_invalid`, and `corrupt`. A `valid` row MUST have a BLOB and storage ID. A non-valid row MAY have a BLOB when preserving diagnostically useful bytes. Missing, dropped, failed, duplicate, and removed positions MUST have explicit rows; writers MUST NOT renumber later frames to conceal a gap. `source_frame_number` is the position in the original acquisition; `frame_id` is the retained stream sequence identity.

`source_files` contains the manifest source fields used by that shard. `storage_formats` contains codec, codec version, quality, pixel format, bit depth, encoder/version, canonical parameters JSON, and a description. A writer MUST NOT repeat a prose encoding label in every frame.

`shard_metadata` is key/value JSON and MUST include shard UUID, format/schema versions, creation time/creator, stream UUID/name, frame range/count, and timestamp range. Null is used for unknown or empty ranges.

The schema-1 columns are normative:

- `frames`: `frame_id INTEGER PRIMARY KEY`; required `source_file_id`, `source_frame_number`, `byte_size`, `status`; optional `timestamp_ns`, `source_timestamp_ns`, `timestamp_source`, `clock_source`, `timezone`, `utc_conversion`, `timestamp_precision_ns`, `synchronization_method`, `known_offset_ns`, `known_drift_ppb`; required Boolean `interpolated`; optional foreign-key `storage_id`; nullable `blob`; optional `width`, `height`, `hash`, `hash_algorithm`, `decoded_pixel_hash`, `decoded_pixel_hash_algorithm`. It has a unique constraint on `(source_file_id, source_frame_number)` and the valid-payload constraint described above.
- `source_files`: `source_file_id INTEGER PRIMARY KEY`, unique required `source_uuid`, required `original_filename`; optional `original_relative_path`, `original_absolute_path`, `byte_size`, `hash`, `hash_algorithm`, `container`, `codec`, `pixel_format`, `width`, `height`, `frame_rate_num`, `frame_rate_den`, `frame_count`, `start_timestamp`, `end_timestamp`.
- `storage_formats`: `storage_id INTEGER PRIMARY KEY`, required `codec`; optional `codec_version`, `quality`, `pixel_format`, `bit_depth`, `encoder`, `encoder_version`, `description`; required `parameters_json` defaulting to `{}`. Structured format fields are unique as a tuple excluding the human description.
- `shard_metadata`: `key TEXT PRIMARY KEY`, `value TEXT NOT NULL`, where every value is JSON text.

Schema 1 MUST provide indexes on `(source_file_id, source_frame_number)` and `timestamp_ns`. Foreign-key relationships are `frames.source_file_id → source_files.source_file_id` and `frames.storage_id → storage_formats.storage_id`.

Finalized shards MUST be treated as immutable. Repair SHOULD create a replacement shard, record provenance, replace the manifest record, and regenerate checksums. Construction MUST use a non-authoritative `.partial` name and only atomically expose and inventory a shard after transaction commit and integrity checking. Abandoned partials MUST NOT be destroyed automatically.

Implementations SHOULD provide a read-only partial-file inventory and a user-directed, recoverable quarantine operation. The reference commands are `pii shards DATASET --partials` and `pii shards DATASET --quarantine-partials RECOVERY_DIR`; neither interprets a partial as authoritative.

An implementation MAY resume an incomplete package in `building` or `finalizing` state. Resume MUST preserve finalized manifest shards, reopen only a structurally valid `.sqlite.partial`, and continue after the last durable source-frame row. A partial written by an older implementation MAY lack stream identity metadata; readers MAY infer its stream from the deterministic shard filename when unambiguous. Rows in an uncommitted SQLite transaction are not considered durable and MAY be replayed.

Non-normative implementation profile: 64 KiB pages, DELETE journal mode, FULL synchronous, a bounded negative `cache_size`, large transactions, `ANALYZE` at finalization, and no automatic `VACUUM` provide portable bulk behavior. Target shards of 5–20 GB are practical; boundaries MAY be driven by byte target, maximum rows, source boundary, or an explicit request. An individual BLOB MUST NOT be split.

## 7. Images and extraction

The BLOB is the exact retained encoded representation. Its stored-BLOB hash MUST cover those exact bytes. Extraction MUST copy bytes without decode/re-encode. `jpeg` maps to `.jpg`, `png` to `.png`; unknown codecs SHOULD map to `.bin` unless the user chooses an extension. Extraction MUST stream, use bounded memory, parameterize SQL, and confine output paths. Null-BLOB records are reported or skipped, never emitted as invented images.

## 8. Time

`timestamp_ns` is an integer count in nanoseconds in the declared time basis; its presence does not assert nanosecond accuracy. Records MAY carry source timestamp, timestamp source, clock source, timezone, UTC conversion method, precision in nanoseconds, synchronization method, known offset in nanoseconds, drift in parts per billion, and interpolation flag. Writers MUST NOT infer precision, UTC conversion, or synchronization not supported by evidence. Recognized timestamp sources include video PTS, instrument clock, acquisition-computer clock, external navigation clock, interpolated, and unknown; extensions MAY add values.

## 9. Scientific metadata

`metadata.toml` MUST be UTF-8 TOML and MUST contain `[schema]` and `[dataset]`. It SHOULD cover project/expedition/station, collection interval and region/bounds, parties and contacts, platforms/vessels, deployments, instruments/cameras/optics/sensors/illumination/acquisition settings, streams/frame rate/field of view/pixel scale, depth, calibration references, license/citation/DOI/accession/embargo/restrictions, acknowledgments, and funding.

Multiple investigators, instruments, deployments, streams, and funding sources MUST use arrays of tables. Organization-specific data MUST live below `[extensions.organization_name]`. Unknown extensions MUST be preserved. Empty or unknown values SHOULD be omitted rather than guessed.

## 10. History

`history.jsonl` is UTF-8 JSON Lines. Every nonblank line MUST independently parse as a JSON object. Public APIs MUST append and MUST NOT rewrite existing events. Each version-1 event contains `event_schema_version`, event UUID, timestamp, operation, software/version, optional git commit/operator, parameters, input/output identifier-and-hash records, environment, status, and optional message. Hash records MUST state algorithm, semantic target, and value.

Operations include `dataset_created`, `dataset_resumed`, `source_ingested`, `frames_transcoded`, `shard_created`, `shard_finalized`, `dataset_finalized`, `metadata_modified`, `dataset_verified`, `dataset_repaired`, and `dataset_exported`. New operation names MAY be added.

## 11. Integrity

Recognized algorithms include `sha256`, `blake3`, and `xxh3`. SHA-256 is the default archival algorithm. A reader MUST NOT require optional implementations merely to open a package. Every high-level hash record MUST identify its algorithm and semantic target: `source_file`, `stored_blob`, `decoded_pixels`, `shard_file`, or `package_file`.

`checksums.sha256` uses conventional lowercase hex, two spaces, and a relative POSIX path. It covers every regular finalized package file except itself, temporary files, and partial shards. Excluding the checksum list itself prevents self-reference; the manifest does not inventory the checksum file, so no second circular dependency exists. Consequently manifest, metadata, or history edits invalidate the checksum list and require regeneration, while shard hashes remain anchored in the manifest. Implementations MUST NOT append a verification event after generating checksums without regenerating them.

## 12. Validation

Quick validation checks required files, manifest structure/version, safe shard paths, existence, declared sizes and shard hashes, and package checksums. Structural additionally opens every shard, runs SQLite integrity checking, checks required schema/tables, reconciles manifest and shard range/count metadata, checks source mappings, and validates required metadata. Full additionally streams every frame, checks BLOB size/hash where supported, detects duplicate or unrepresented frame IDs and invalid mappings, performs cross-file checks, and MAY check recognizable signatures without decoding.

Archival validation includes full validation and MUST require: every source has an expected frame count; expected counts equal all explicit frame-status rows; all configured BLOB hashes are present and verifiable; manifest/shards agree; package checksums pass; no partial shard remains; and successful creation/finalization history exists. Success means the interchange package passed these mechanical checks; it does not prove source video can safely be deleted under a project's retention, legal, scientific, or backup policy. Tools MUST NOT delete source data.

## 13. Lifecycle and extensions

Image additions occur only while building. `finalizing` is transient but recoverable; `complete` means construction completed; `verified` MAY record a successful verification snapshot; `modified` signals package-level changes since a prior validation. Ordinary metadata edits MAY be allowed but MUST cause checksum regeneration and SHOULD append `metadata_modified`. Authoritative payload changes require explicit repair/rewrite provenance.

New optional tables, manifest fields, metadata extension tables, status/timestamp vocabulary values, and storage codecs MAY be introduced compatibly. Existing meanings MUST NOT change within format major version 1. Readers MUST ignore compatible unknowns where safe and preserve them when rewriting their containing document.
