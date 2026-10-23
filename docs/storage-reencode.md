# Updating existing project datastores

Project managers and system administrators can save an image storage policy in
PelagiaView Administration, then select **Update datastores**. The API endpoint
`POST /projects/{project_id}/storage-reencode` queues a `storage_reencode` job;
`GET` on the same path reports its latest status, and
`POST /projects/{project_id}/storage-reencode/cancel` requests cancellation.
An active job is reused when the button is selected again.
Administration shows the effective saved policy before enabling the action;
the manager must check a verification box for the selected project. The status
response also reports whether a recent worker advertises `storage_reencode`,
so queued work can be distinguished from a missing worker.

`GET /projects/{project_id}/storage-usage` reports the project's physical
KVStore file bytes when available. It includes retained historical frame blobs
and excludes PostgreSQL ROI payloads and WAL, so it is not a total project
storage figure.

The job snapshots the effective project settings and a creation-time boundary.
It visits the project's original, preprocessed, and background frame payloads,
then candidate and refined ROI images and masks. Missing payloads are left
missing. The ROI tier uses the stored crop's height × width; crops at or above
`large_roi_min_pixels` use the large codec. Masks use the separate lossless mask
codec. Newly created resources are governed by the project's current settings
and are outside the submitted job's snapshot.

The worker reads bounded pages, verifies each re-encoded array's shape and dtype
(and exact values for lossless codecs), writes new KVStore blobs before updating
frame references, and commits each resource independently. PostgreSQL updates
are fenced by the row version and project ownership. Progress checkpoints allow
retries to resume; per-resource policy markers prevent repeated lossy encoding
after an interrupted attempt. The job can be paused or cancelled through queue
controls, and its current status remains visible after leaving the page.

JPEG/JPEG XL re-encoding is lossy. Previous frame blobs remain in KVStore and
their keys are recorded in frame metadata history; the job does not delete them.
Candidate and refined ROI payloads are rewritten in PostgreSQL with prior
payload hashes and codec metadata recorded in history. Their old TOAST space
becomes reusable after vacuum but the database file may not shrink. This process
can generate substantial WAL and transient disk usage; take and test a backup
and confirm disk headroom before applying it to a large project. Published
Registry assets and external source files are not rewritten.

Workers must advertise the `storage_reencode` stage. The bundled Docker stack
provides a dedicated `storage-worker` so a long rewrite does not occupy the
general processing worker; custom worker deployments must add the stage to a
worker's capabilities.
