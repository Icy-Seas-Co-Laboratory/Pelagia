# ROI curation and classification evidence

Pelagia's curation workflow keeps three responsibilities explicit:

- **Oracle Builder** executes classification models and owns model artifacts, embeddings, prototype calculations, and K-nearest-neighbor calculations.
- **Pelagia** owns the project label vocabulary, human annotations and reviews, immutable model provenance, evidence records, and classification jobs.
- **PelagiaView** presents the review queue and turns an explicit reviewer action into a human assertion.

A model prediction is never written as human ground truth. Selecting **Accept prediction as human label** creates a normal human annotation and records the evidence row that suggested it. Replacing or clearing a label retains its audit history.

## Operating the workflow

1. Start Oracle Builder with a classification bundle registered under the selector configured by `oracle.default_classification_model`.
2. Run at least one Pelagia worker with the `classify` capability. The standard worker TOML assigns this capability to the existing refinement workers, so a separate process is optional.
3. Open **Curation** in PelagiaView. Choose **Run selected** or **Run all** to enqueue inference.
4. Filter and sort the queue by human state, review state, evidence availability, confidence, or disagreement. Assign a project label, then verify, reject, or flag the assertion.

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

`detections_refined` is the canonical curatable ROI. Each classification job creates one `classification_inference_runs` row and one `classification_evidence` row per successfully evaluated ROI. Probability, prototype, and KNN summary values are indexed in Postgres for review queues. Full evidence packets and Oracle result provenance are retained as JSON. Embeddings are NPY-encoded in the project's KVStore and referenced by hash from Postgres.

The label mapping from an immutable Oracle artifact and class index to a Pelagia project label is stored in `model_class_mappings`. Labels first encountered from an Oracle result are imported into the project vocabulary and remain owned by Pelagia thereafter.

KNN neighbor identity, class, rank, and similarity are retained. The current Oracle bundle does not package deployable exemplar imagery, so PelagiaView states that limitation instead of displaying misleading or unavailable images.

## API surface

- `GET /curation/options` — ownership, labels, and classification model catalog
- `GET|POST /curation/labels` — project label vocabulary
- `GET /curation/rois` and `GET /curation/rois/{id}` — queue and detailed evidence
- `POST /curation/classification-jobs` — asynchronous Oracle classification
- `POST /curation/annotations` — explicit human assertion
- `POST /curation/annotations/remove` — retire the current assertion while retaining history
- `POST /curation/reviews` — human verification state

Schema migration `0005_roi_curation.sql` installs the curation tables. The older generic `models` and `classification_results` records are not used by this workflow.
