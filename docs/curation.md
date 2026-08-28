# ROI curation and ML evidence

Pelagia's curation workflow keeps three responsibilities explicit:

- **Oracle Builder** executes classification and self-supervised clustering models and owns model artifacts, embeddings, prototype calculations, K-nearest-neighbor calculations, and cluster fitting.
- **Pelagia** owns the project label vocabulary, human annotations and reviews, immutable model provenance, evidence records, and classification jobs.
- **PelagiaView** presents the review queue and turns an explicit reviewer action into a human assertion.

A model prediction is never written as human ground truth. Selecting **Accept prediction as human label** creates a normal human annotation and records the evidence row that suggested it. Replacing or clearing a label retains its audit history.

## Operating the workflow

1. Start Oracle Builder with a classification bundle registered under the selector configured by `oracle.default_classification_model`.
2. Run at least one Pelagia worker with the `classify` capability. The standard worker TOML assigns this capability to the existing refinement workers, so a separate process is optional.
3. Open **ML Evidence** in PelagiaView. Select one or more ready classification and/or self-supervised clustering models, constrain the refined ROI target query, review the per-model workload, and queue the resulting evidence jobs.
4. Open **Curation** to inspect the resulting evidence in context, explore feature-space outputs, and make explicit human annotation or review decisions. Curation does not enqueue ML inference.
5. Filter and sort the queue by human state, review state, evidence availability, confidence, disagreement, or telemetry ranges. Assign a project label, then verify, reject, or flag the assertion.

When a refined ROI is focused, the inspector shows telemetry resolved at its
frame timestamp when the project has an applicable stream. Use the `+` button
under **Telemetry criteria** to choose any catalog parameter and enter an
inclusive minimum, maximum, or both. Values use the parameter's canonical unit;
multiple criteria are combined with AND and are evaluated server-side before
pagination.

The Curation model panel follows active classification jobs and the most recent
result across page reloads. Pelagia resolves the exact number of eligible ROIs
before inference and reports input loading, the current Oracle batch, evidence
storage, completion, and failure through the standard job progress record. The
worker runtime renews the queue lease while a handler is active, including while
it is blocked waiting for an Oracle response. Progress advances between batches;
Oracle Builder's synchronous inference endpoint does not expose per-image
progress within a submitted batch.

Classification inputs use only the refined detection's `bbox_*` rectangle. The
worker decodes the stored ROI payload, removes the surrounding `crop_bbox_*`
padding, and records the `refined_detection_bbox_v1` crop policy and both source
rectangles in inference provenance. Invalid or inconsistent geometry fails the
job rather than silently classifying padded pixels.

Useful keyboard actions in the gallery are `1`–`0` for the first ten labels, arrow keys to move, space to verify, `F` to flag, and Escape to clear selection.

## Evidence storage

`detections_refined` is the canonical curatable ROI. Each evidence job creates one `classification_inference_runs` row (the legacy table name is retained for compatibility) and one per-ROI evidence row. Classification jobs write `classification_evidence`; clustering jobs write `clustering_evidence`. Probability, prototype, KNN, cluster assignment, similarity, and novelty/abstention summaries are indexed or retained in Postgres. Full evidence packets and Oracle result provenance are retained as JSON. Embeddings are NPY-encoded in the project's KVStore and referenced by hash from Postgres.

The clustering packet is evidence about location in a model-defined feature space, not a Pelagia label or biological taxonomy. Cluster IDs are run-local and must be interpreted together with the Oracle artifact, embedding contract, clustering method, and run provenance. A classification artifact may also return a secondary clustering packet; Pelagia stores that packet alongside the classification evidence without converting it into a human assertion.

## Feature-space exploration

PelagiaView's **Clusters** analysis page is a project-scoped ROI browser for feature-space evidence. A reviewer first selects one persisted evidence source, which is always one inference run and its recorded model artifact. Its **Similar ROIs** view performs exact cosine comparison against that source's NPY ROI vectors; it never compares vectors from different runs or artifacts. For runs above 100,000 vectors, the API returns a deterministic exact ranking of the first 100,000 vectors in stable ROI-ID order and reports that coverage to the UI rather than failing. A provenance-compatible, materialized per-run vector index remains the required path for full-source search at larger scale.

Its **Clusters** and **UMAP** views use the same project- and inference-run-scoped vectors. They produce exactly ten UMAP coordinates with a fixed random seed (`20260826`) and derive the browser's groups from HDBSCAN, rather than from a model's recorded cluster identity. Users can adjust `min_cluster_size`, `min_samples` (min_n), and `cluster_selection_epsilon` for the exploratory grouping; the response records those values. The analysis is queued only to dedicated `feature_space_analysis` CPU workers, has a hard 30-second worker deadline, permits at most two active analyses per project, and caches a completed identical request for 15 minutes (with 30-minute job cleanup). The coordinates, component extents, group labels, and membership strengths support client-side range filtering; no source vectors are changed. To retain a responsive visualization and bounded computation, it analyzes at most the deterministic first 5,000 persisted vectors, reporting whether coverage is `full_source` or `deterministic_prefix`; vectors with a different dimensionality than the largest compatible cohort and unreadable vectors are reported separately. Vectors from different model artifacts remain separate feature spaces and are never mixed. UMAP coordinates and HDBSCAN assignments are exploratory analysis results, not a label, taxonomy claim, or replacement for recorded Oracle clustering evidence.

Recorded Oracle clusters, label prototypes, and model-specific similarity remain available as ROI evidence and provenance, but do not drive the Clusters browser's grouping. They remain scoped to the model artifact and inference run that created them. A group identifier is not a reusable biological category.

The label mapping from an immutable Oracle artifact and class index to a Pelagia project label is stored in `model_class_mappings`. Labels first encountered from an Oracle result are imported into the project vocabulary and remain owned by Pelagia thereafter.

KNN neighbor identity, class, rank, and similarity are retained. The current Oracle bundle does not package deployable exemplar imagery, so PelagiaView states that limitation instead of displaying misleading or unavailable images.

## API surface

- `GET /curation/options` — ownership, labels, and classification model catalog
- `GET|POST /curation/labels` — project label vocabulary
- `GET /curation/rois` and `GET /curation/rois/{id}` — queue and detailed evidence
- `GET /curation/feature-space/sources` — project embedding spaces, scoped to inference runs
- `GET /curation/feature-space/similar/{roi_id}` — exact bounded cosine neighbors in one source
- `GET /curation/feature-space/umap` — bounded, run-scoped deterministic 10-D UMAP projection and HDBSCAN assignments; accepts `min_cluster_size`, `min_samples`, and `cluster_selection_epsilon` for exploratory grouping
- `GET /curation/feature-space/clusters` and `GET /curation/feature-space/clusters/{cluster_id}/rois` — run-local clustering browser
- `POST /curation/classification-jobs` — asynchronous Oracle classification
- `POST /curation/clustering-jobs` — asynchronous Oracle self-supervised clustering evidence
- `POST /curation/clustering-targets/preview` — preview clustering targets
- `POST /curation/annotations` — explicit human assertion
- `POST /curation/annotations/remove` — retire the current assertion while retaining history
- `POST /curation/reviews` — human verification state

`GET /curation/rois` accepts repeated `telemetry_filter` JSON query values,
for example `{"parameter_key":"temperature","min_value":2,"max_value":8}`.
The global detection endpoint accepts the same filter shape through
`GET /detections`.

Schema migration `0005_roi_curation.sql` installs the curation tables. The older generic `models` and `classification_results` records are not used by this workflow.
