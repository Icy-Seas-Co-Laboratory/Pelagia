# ROI statistics

| Operation | Area | Perimeter |
| --- | --- | --- |
| Candidate generation, either assembly method | Bounding-box width × height | 2 × (width + height), before mask/contour work where possible |
| Filtering materialized candidates and browsing ROIs | Bounding-box width × height | Stored external-contour length |
| Refined ROI display | Sum of external-contour areas | Sum of external-contour lengths |
| Curation and Registry selection area gates | Refined bounding-box width × height | Not applicable |

Contour geometry uses OpenCV `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`,
`contourArea`, and closed `arcLength`. Coordinates lie at pixel centers:
a filled 4 × 3 rectangle has bbox area 12 px², contour area 6 px², and
contour perimeter 10 px. Hole areas are not subtracted; inner hole boundaries
are not included in perimeter. Pixel counts remain available internally for
segmentation and overlap decisions, but are not presented as ROI area.

New records include `metadata.roi_measurements.version = 1` and explicit
`area_method`/`perimeter_method` values. Detection responses add nullable
`bbox_area`, `contour_area`, and `contour_perimeter`. The historical `area`
field is retained for compatibility. Summary listings do not fetch image objects;
for old refined records without versioned measurements, contour area is null
and the displayed perimeter is the stored historical value. ROI detail and export
paths can measure a saved mask when available. An
unavailable contour area is null, never zero or the old foreground count.
Reading does not mutate stored records. See
[export compatibility](exports.md) before comparing historical exports.

Explorer pages through both candidate and refined frame records, including
split children, and clears selections when results change. The refined ROI
Browser uses the exact refined record for images, bounds, and measurements.
Its search and ordering remain candidate-scoped, as labelled in the UI.
`GET /refined-detections?frame_id=...&limit=500&offset=0` supplies project-scoped
refined records with page metadata (limit 1–1000).

Feature-space contour-area axes use only versioned stored measurements; legacy
rows without that measurement are omitted from that axis. Opening an ROI detail
can measure its saved mask. Summary totals identify how many records have known
measurements rather than treating missing measurements as zero.
