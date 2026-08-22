# AGENTS.md

## Purpose

Pelagia is a Python 3.12 backend for scientific-image ingestion, ROI-first
processing, storage, curation, background jobs, and HTTP/CLI access. Preserve
reproducibility, explicit provenance, project isolation, and inspectable data
formats. Treat claims about performance or production readiness as unverified
unless tests, benchmarks, or current documentation support them.

## Work efficiently

- Start with `rg` or `rg --files` in the relevant subsystem; do not inventory or
  read the entire repository.
- Read only the matching sections of `README.md` or `docs/`. Search headings or
  terms first, then open the smallest useful range.
- Inspect the implementation and its nearest tests together. Prefer a targeted
  test during iteration; broaden validation only when the change warrants it.
- Do not inspect generated/runtime directories such as `.venv/`, `.pelagia/`,
  caches, local data stores, logs, or `config.toml` unless the task requires it.
- Preserve unrelated working-tree changes. Do not rewrite or reformat files
  outside the requested scope.

## Hybrid model routing

Use the least expensive model that can reliably complete the task. Keep one
model responsible for the final integration; delegate only bounded, independent
work with explicit inputs and expected outputs. Do not have multiple models
inspect the same broad context by default.

- **Luna (`gpt-5.6-luna`)**: fast, high-volume, low-risk work. Use for repository
  searches, file/test discovery, mechanical edits, formatting, straightforward
  test additions, documentation cleanup, log summarization, and running known
  validation commands. Default to low or medium reasoning.
- **Terra (`gpt-5.6-terra`)**: default implementation model. Use for ordinary
  features and bug fixes, focused refactors, API/service/storage changes that
  follow established patterns, test design, code review, and integrating Luna
  results. Default to medium reasoning; use high when several subsystems interact.
- **Sol (`gpt-5.6-sol`)**: escalation and high-consequence model. Use for unclear
  architecture, scientific correctness, concurrency or transaction hazards,
  authentication/project-isolation boundaries, schema migrations, data-loss
  risk, difficult root-cause analysis, novel cross-repository design, or final
  review of a large/high-risk change. Use high or greater reasoning only when the
  risk or ambiguity justifies it.

Suggested flow for substantial work:

1. Route initial discovery and narrow inventory tasks to Luna.
2. Let Terra own the plan, implementation, focused tests, and integration.
3. Escalate only the uncertain or high-risk decision to Sol, passing a compact
   summary plus the exact files, constraints, evidence, and question.
4. Return the decision to Terra for implementation unless Sol is needed to own
   the high-risk change through validation.

Escalate upward when tests repeatedly fail without a clear local cause, the task
crosses architectural boundaries, requirements conflict, or correctness depends
on an unstated invariant. Route downward when the remaining work becomes
mechanical. Each handoff should contain findings and file/line pointers—not raw
transcripts or a full repository dump. Never use model routing to bypass the
user's requested scope, approval requirements, or destructive-action safeguards.

## Repository map

- `Pelagia/api/`: FastAPI app, authentication, schemas, and routes.
- `Pelagia/cli/`: Typer CLI commands and command wiring.
- `Pelagia/services/`: application workflows and orchestration.
- `Pelagia/processing/`: image/frame/ROI processing and model-facing logic.
- `Pelagia/storage/`: PostgreSQL and blob/KV storage implementations.
- `Pelagia/workers/`: worker registry, runtime, handlers, and progress.
- `Pelagia/domain.py`: shared domain models and invariants.
- `Pelagia/config.py`: defaults, configuration parsing, and environment mapping.
- `pelagia_interchange/`: independent Python 3.11+ archival interchange package;
  keep its standard-library-only runtime unless requirements explicitly change.
- `tests/`: backend and interchange tests, generally named after their subsystem.
- `docs/`: operational and format documentation; update the relevant page when
  changing a user-visible command, workflow, schema, or storage contract.
- `scripts/`: environment and local worker-stack management.
- `../PelagiaView/`: separate SvelteKit frontend repository. Do not edit it unless
  the task explicitly includes frontend work.

## Architectural boundaries

- Keep HTTP parsing/serialization in `api`, orchestration in `services`, durable
  persistence in `storage`, and scientific transformations in `processing`.
- Keep source frames and immutable inputs distinct from derived ROI artifacts.
  Do not silently overwrite source data or externally released products.
- Maintain user/project scoping through API, service, job, and storage layers.
- Record enough input, parameter, version, and output context for derived data to
  remain traceable and reproducible.
- Avoid adding heavy dependencies to the core package. In particular, do not
  introduce required third-party dependencies into `pelagia_interchange`.
- Oracle Builder owns TensorFlow/Keras and GPU isolation; Pelagia remains a CPU
  environment and calls that service at the established boundary.
- Use migrations and compatibility paths for persisted schema changes; never
  assume existing databases or blob stores can be discarded.

## Validation

Use the managed environment when available:

```bash
./scripts/pelagia_env sync dev       # only when setup/dependencies are needed
.venv/bin/pytest tests/test_api.py -q
.venv/bin/pytest tests/test_interchange.py -q
.venv/bin/pytest -q                  # final broad check when justified
```

Choose tests by changed subsystem (for example `test_workers.py`,
`test_processing_queue.py`, `test_project_isolation.py`, or `test_config.py`).
For environment or stack changes, use the matching script tests and non-mutating
checks such as `./scripts/pelagia_env doctor all` or
`./scripts/pelagia_stack_from_toml.sh validate scripts/pelagia_workers.example.toml`.
State clearly when PostgreSQL, codecs, Oracle Builder, or other external services
prevent a relevant test from running.

## Change expectations

- Follow existing typing, naming, error, serialization, and transaction patterns
  in adjacent code rather than introducing a new abstraction prematurely.
- Add or update focused regression tests for behavior changes and bug fixes.
- Keep configuration defaults synchronized with examples and documentation.
- Never commit credentials, machine-specific paths, runtime state, or local
  `config.toml` values.
- Comments and docs should explain scientific or operational intent and unusual
  failure modes, not restate the code.
