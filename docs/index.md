# Pelagia documentation

Use this page as the task-oriented entry point. The repository README gives the
system overview and a quick path to a running stack; the linked pages below are
the detailed operational and format references.

## Getting started

- [Backend installation and local stack](getting-started.md)
- [Run PelagiaView](https://github.com/Icy-Seas-Co-Laboratory/PelagiaView)
- [Docker deployment](../deploy/docker/README.md)
- [Python environment](python-environment.md)
- [Oracle Builder integration](oracle-builder.md)

## Daily operations

- [End-user operations](end-user-support.md)
- [Processing status](processing-status.md)
- [Job commands](job-commands.md)
- [Curation and ML evidence](curation.md)
- [Telemetry ingestion](telemetry.md)
- [Backup and restore](backup.md)
- [Reset the system](reset-system.md)
- [Migration](migration.md)

## Data and interchange

- [Interchange usage](interchange-usage.md)
- [Interchange specification](interchange-specification.md)
- [Artifacts and manifests](artifacts.md)
- [Exports](exports.md)
- [Metadata example](metadata.example.toml)
- [Schemas](schemas/manifest.schema.json)

## Developer reference

- [System architecture](developer-reference.md#system-architecture)
- [Configuration](developer-reference.md#configuration)
- [HTTP API](developer-reference.md#http-api)
- [Jobs and workers](developer-reference.md#jobs-and-workers)
- [Storage strategy](developer-reference.md#storage-strategy)
- [Candidate ROI benchmark results](candidate-roi-performance.md)

## Repository boundaries

Pelagia owns API, storage, processing, projects, jobs, and provenance. PelagiaView
is a separate SvelteKit repository. Oracle Builder owns TensorFlow/Keras,
accelerator isolation, sealed model artifacts, and model execution. Keep those
boundaries explicit when extending the system.
