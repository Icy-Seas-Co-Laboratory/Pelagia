"""Project-scoped exploration of persisted ROI embedding spaces."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np


MAX_EXACT_VECTOR_SCAN = 100_000
MAX_UMAP_VECTOR_SCAN = 5_000
UMAP_DIMENSIONS = 10
UMAP_RANDOM_SEED = 20_260_826


class FeatureSpaceError(ValueError):
    """Raised when a feature-space request cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class FeatureSpaceSource:
    kind: str
    inference_run_id: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.inference_run_id}"


def parse_feature_space_source(value: str) -> FeatureSpaceSource:
    kind, separator, inference_run_id = str(value or "").partition(":")
    if (
        separator != ":"
        or not inference_run_id
        or ":" in inference_run_id
        or kind not in {"classification", "clustering"}
    ):
        raise FeatureSpaceError("Feature-space source must be classification:<run-id> or clustering:<run-id>.")
    return FeatureSpaceSource(kind=kind, inference_run_id=inference_run_id)


class FeatureSpaceService:
    """Exact, bounded cosine similarity over one persisted embedding space."""

    def __init__(self, context, *, project_id: str):
        self.context = context
        self.repository = context.repository
        self.project_id = str(project_id)
        if self.repository is None:
            raise FeatureSpaceError("Feature-space exploration requires Postgres.")

    def sources(self) -> list[dict[str, Any]]:
        rows = self.repository.list_feature_space_sources(project_id=self.project_id)
        result = []
        for row in rows:
            item = dict(row)
            item["source_key"] = f"{item['source_kind']}:{item['inference_run_id']}"
            item["comparison"] = "cosine_similarity"
            item["scope"] = "single_inference_run"
            result.append(item)
        return result

    def browse_rois(self, *, source_key: str, limit: int) -> dict[str, Any]:
        """Return bounded ROI cards that can seed an exact similarity search."""

        source = parse_feature_space_source(source_key)
        rows = self.repository.list_feature_space_embeddings(
            project_id=self.project_id,
            source_kind=source.kind,
            inference_run_id=source.inference_run_id,
            limit=limit,
        )
        ids = [str(row["refined_detection_id"]) for row in rows if row.get("embedding_payload_ref")]
        summaries = self.repository.list_feature_space_roi_summaries(
            project_id=self.project_id, roi_ids=ids
        )
        by_id = {str(row["id"]): dict(row) for row in summaries}
        return {
            "items": [by_id[roi_id] for roi_id in ids if roi_id in by_id],
            "source_key": source.key,
            "limit": limit,
        }

    def umap_rois(
        self,
        *,
        source_key: str,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
    ) -> dict[str, Any]:
        """Project one run to deterministic UMAP coordinates and HDBSCAN groups.

        The projection and its ephemeral cluster assignments are scoped to this
        exact, bounded embedding cohort. They are exploration aids, not durable
        biological labels or replacements for Oracle Builder evidence.
        """

        source = parse_feature_space_source(source_key)
        if min_cluster_size < 2:
            raise FeatureSpaceError("HDBSCAN min_cluster_size must be at least 2.")
        if min_samples is not None and min_samples < 1:
            raise FeatureSpaceError("HDBSCAN min_samples must be at least 1.")
        if cluster_selection_epsilon < 0.0:
            raise FeatureSpaceError("HDBSCAN cluster_selection_epsilon cannot be negative.")
        total_vector_count = self.repository.count_feature_space_embeddings(
            project_id=self.project_id,
            source_kind=source.kind,
            inference_run_id=source.inference_run_id,
        )
        scan_limit = min(total_vector_count, MAX_UMAP_VECTOR_SCAN)
        rows = self.repository.list_feature_space_embeddings(
            project_id=self.project_id,
            source_kind=source.kind,
            inference_run_id=source.inference_run_id,
            limit=scan_limit,
        )
        kvstore = self.context.kvstore_for_project(self.project_id, initialize=False)
        if kvstore is None:
            raise FeatureSpaceError("The project embedding store is unavailable.")

        readable: list[tuple[str, np.ndarray]] = []
        unreadable_count = 0
        for row in rows:
            payload_ref = row.get("embedding_payload_ref")
            if not payload_ref:
                unreadable_count += 1
                continue
            try:
                values = np.asarray(
                    np.load(io.BytesIO(kvstore.get_store(str(payload_ref))), allow_pickle=False),
                    dtype="float64",
                ).reshape(-1)
            except (KeyError, OSError, TypeError, ValueError):
                unreadable_count += 1
                continue
            if values.size == 0 or not np.isfinite(values).all():
                unreadable_count += 1
                continue
            readable.append((str(row["refined_detection_id"]), values))

        by_dimension: dict[int, list[tuple[str, np.ndarray]]] = {}
        for item in readable:
            by_dimension.setdefault(item[1].size, []).append(item)
        if not by_dimension:
            raise FeatureSpaceError("No readable embeddings are available in the selected source.")
        # A stable, largest compatible cohort avoids silently padding/truncating
        # vectors and keeps repeat requests deterministic when mixed artifacts exist.
        dimension, compatible = min(
            by_dimension.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(compatible) < 5:
            raise FeatureSpaceError(
                "UMAP and HDBSCAN require at least five compatible embeddings."
            )
        matrix = np.vstack([values for _, values in compatible])
        try:
            coordinates = self._reduce_umap(matrix)
            labels, probabilities = self._cluster_hdbscan(
                coordinates,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
            )
        except (FloatingPointError, ValueError, TypeError) as exc:
            raise FeatureSpaceError("UMAP/HDBSCAN could not be computed safely for the selected source.") from exc
        if coordinates.shape != (len(compatible), UMAP_DIMENSIONS) or not np.isfinite(coordinates).all():
            raise FeatureSpaceError("UMAP returned invalid coordinates for the selected source.")
        if labels.shape != (len(compatible),):
            raise FeatureSpaceError("HDBSCAN returned invalid cluster assignments for the selected source.")
        component_ranges = [
            {
                "component": index + 1,
                "minimum": float(coordinates[:, index].min()),
                "maximum": float(coordinates[:, index].max()),
            }
            for index in range(UMAP_DIMENSIONS)
        ]
        ids = [roi_id for roi_id, _ in compatible]
        summaries = self.repository.list_feature_space_roi_summaries(
            project_id=self.project_id, roi_ids=ids
        )
        summaries_by_id = {str(row["id"]): dict(row) for row in summaries}
        items = []
        for index, roi_id in enumerate(ids):
            summary = summaries_by_id.get(roi_id)
            if summary is not None:
                label = int(labels[index])
                summary["umap_coordinates"] = [float(value) for value in coordinates[index]]
                summary["hdbscan_label"] = label
                summary["hdbscan_cluster_id"] = None if label < 0 else f"hdbscan:{label}"
                summary["hdbscan_membership_strength"] = float(probabilities[index])
                items.append(summary)
        return {
            "items": items,
            "source_key": source.key,
            "projection": "umap",
            "clustering": "hdbscan",
            "component_count": UMAP_DIMENSIONS,
            "random_seed": UMAP_RANDOM_SEED,
            "umap_parameters": {"n_neighbors": min(15, len(compatible) - 1), "min_dist": 0.1, "metric": "cosine"},
            "hdbscan_parameters": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "metric": "euclidean",
            },
            "component_ranges": component_ranges,
            "total_vector_count": total_vector_count,
            "scanned_vector_count": len(rows),
            "projection_scope": "full_source" if len(rows) >= total_vector_count else "deterministic_prefix",
            "readable_embedding_count": len(readable),
            "compatible_embedding_count": len(compatible),
            "incompatible_embedding_count": len(readable) - len(compatible),
            "unreadable_embedding_count": unreadable_count,
            "vector_dimension": dimension,
            "cluster_count": len({int(label) for label in labels if int(label) >= 0}),
            "noise_count": int(np.count_nonzero(labels < 0)),
        }

    @staticmethod
    def _reduce_umap(matrix: np.ndarray) -> np.ndarray:
        try:
            from umap import UMAP
        except ImportError as exc:
            raise FeatureSpaceError(
                "UMAP support requires the feature-space dependencies (umap-learn and hdbscan)."
            ) from exc
        return np.asarray(
            UMAP(
                n_components=UMAP_DIMENSIONS,
                n_neighbors=min(15, len(matrix) - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=UMAP_RANDOM_SEED,
                n_jobs=1,
                # Spectral initialization cannot produce ten axes from a small
                # cohort; seeded random initialization keeps those valid runs
                # deterministic while preserving the fixed output dimension.
                init="random",
            ).fit_transform(matrix),
            dtype="float64",
        )

    @staticmethod
    def _cluster_hdbscan(
        coordinates: np.ndarray,
        *,
        min_cluster_size: int,
        min_samples: int | None,
        cluster_selection_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from hdbscan import HDBSCAN
        except ImportError as exc:
            raise FeatureSpaceError(
                "HDBSCAN support requires the feature-space dependencies (umap-learn and hdbscan)."
            ) from exc
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",
            prediction_data=False,
        )
        labels = np.asarray(model.fit_predict(coordinates), dtype="int64")
        probabilities = np.asarray(getattr(model, "probabilities_", np.zeros(len(labels))), dtype="float64")
        return labels, probabilities

    def similar_rois(
        self,
        *,
        roi_id: str,
        source_key: str,
        limit: int,
        minimum: float,
    ) -> dict[str, Any]:
        source = parse_feature_space_source(source_key)
        if not -1.0 <= minimum <= 1.0:
            raise FeatureSpaceError("Minimum similarity must be between -1 and 1.")
        if source.kind == "clustering":
            return self._cluster_local_similar_rois(
                roi_id=roi_id,
                source=source,
                limit=limit,
                minimum=minimum,
            )
        total_vector_count = self.repository.count_feature_space_embeddings(
            project_id=self.project_id,
            source_kind=source.kind,
            inference_run_id=source.inference_run_id,
        )
        scan_limit = min(total_vector_count, MAX_EXACT_VECTOR_SCAN)
        rows = self.repository.list_feature_space_embeddings(
            project_id=self.project_id,
            source_kind=source.kind,
            inference_run_id=source.inference_run_id,
            limit=scan_limit,
        )
        kvstore = self.context.kvstore_for_project(self.project_id, initialize=False)
        if kvstore is None:
            raise FeatureSpaceError("The project embedding store is unavailable.")

        ids: list[str] = []
        vectors: list[np.ndarray] = []
        unreadable_count = 0
        for row in rows:
            payload_ref = row.get("embedding_payload_ref")
            if not payload_ref:
                continue
            try:
                values = np.asarray(
                    np.load(io.BytesIO(kvstore.get_store(str(payload_ref))), allow_pickle=False),
                    dtype="float32",
                ).reshape(-1)
            except (KeyError, OSError, TypeError, ValueError):
                unreadable_count += 1
                continue
            norm = float(np.linalg.norm(values))
            if not np.isfinite(values).all() or norm <= 0:
                unreadable_count += 1
                continue
            ids.append(str(row["refined_detection_id"]))
            vectors.append(values / norm)

        if roi_id not in ids:
            raise FeatureSpaceError("This ROI has no readable embedding in the selected source.")
        reference_index = ids.index(roi_id)
        reference = vectors[reference_index]
        compatible = [index for index, vector in enumerate(vectors) if vector.shape == reference.shape]
        if not compatible:
            raise FeatureSpaceError("No compatible embeddings are available in the selected source.")
        scored = ((ids[index], float(np.dot(vectors[index], reference))) for index in compatible)
        ranked = sorted(
            (item for item in scored if item[1] >= minimum),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
        summaries = self.repository.list_feature_space_roi_summaries(
            project_id=self.project_id, roi_ids=[item[0] for item in ranked]
        )
        by_id = {str(row["id"]): dict(row) for row in summaries}
        items = []
        for detection_id, score in ranked:
            row = by_id.get(detection_id)
            if row is not None:
                row["similarity"] = score
                row["is_reference"] = detection_id == roi_id
                items.append(row)
        return {
            "items": items,
            "reference_roi_id": roi_id,
            "source_key": source.key,
            "comparison": "cosine_similarity",
            "minimum": minimum,
            "candidate_count": len(rows),
            "total_vector_count": total_vector_count,
            "scanned_vector_count": len(rows),
            "search_scope": "full_source" if len(rows) >= total_vector_count else "deterministic_prefix",
            "readable_embedding_count": len(vectors),
            "unreadable_embedding_count": unreadable_count,
            "limit": limit,
        }

    def _cluster_local_similar_rois(
        self,
        *,
        roi_id: str,
        source: FeatureSpaceSource,
        limit: int,
        minimum: float,
    ) -> dict[str, Any]:
        """Use persisted cluster membership instead of an unsafe full-run scan.

        Clustering evidence stores similarity to its assigned centroid, not a
        pairwise ROI similarity.  Preserve that distinction in the response.
        """

        assignment = self.repository.get_feature_space_cluster_assignment(
            project_id=self.project_id,
            inference_run_id=source.inference_run_id,
            refined_detection_id=roi_id,
        )
        if assignment is None:
            raise FeatureSpaceError("This ROI has no assigned cluster in the selected evidence run.")
        cluster_id = str(assignment["cluster_id"])
        result = self.repository.list_feature_space_cluster_members(
            project_id=self.project_id,
            inference_run_id=source.inference_run_id,
            cluster_id=cluster_id,
            limit=limit,
            offset=0,
            minimum=minimum,
        )
        items = []
        for value in result["items"]:
            row = dict(value)
            row["is_reference"] = str(row["id"]) == roi_id
            items.append(row)
        return {
            "items": items,
            "reference_roi_id": roi_id,
            "source_key": source.key,
            "comparison": "cluster_centroid_similarity",
            "minimum": minimum,
            "candidate_count": result["total"],
            "total_vector_count": result["total"],
            "scanned_vector_count": result["total"],
            "search_scope": "cluster_local",
            "readable_embedding_count": None,
            "unreadable_embedding_count": 0,
            "limit": limit,
            "cluster_id": cluster_id,
        }

    def clusters(self, *, source_key: str) -> dict[str, Any]:
        source = parse_feature_space_source(source_key)
        if source.kind == "clustering":
            rows = self.repository.list_feature_space_clusters(
                project_id=self.project_id, inference_run_id=source.inference_run_id
            )
            organization_kind = "self_supervised_clusters"
        else:
            rows = self.repository.list_feature_space_label_prototypes(
                project_id=self.project_id, inference_run_id=source.inference_run_id
            )
            organization_kind = "label_prototypes"
        return {
            "items": rows,
            "source_key": source.key,
            "organization_kind": organization_kind,
            "group_ids": "run_local",
        }

    def cluster_members(
        self,
        *,
        source_key: str,
        cluster_id: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        source = parse_feature_space_source(source_key)
        if source.kind == "clustering":
            result = self.repository.list_feature_space_cluster_members(
                project_id=self.project_id,
                inference_run_id=source.inference_run_id,
                cluster_id=cluster_id,
                limit=limit,
                offset=offset,
                minimum=-1.0,
            )
            organization_kind = "self_supervised_clusters"
        else:
            try:
                result = self.repository.list_feature_space_label_prototype_members(
                    project_id=self.project_id,
                    inference_run_id=source.inference_run_id,
                    prototype_id=cluster_id,
                    limit=limit,
                    offset=offset,
                )
            except ValueError as exc:
                raise FeatureSpaceError(str(exc)) from exc
            organization_kind = "label_prototypes"
        result.update(
            source_key=source.key,
            cluster_id=cluster_id,
            organization_kind=organization_kind,
            group_ids="run_local",
        )
        return result
