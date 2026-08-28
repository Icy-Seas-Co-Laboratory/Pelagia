from __future__ import annotations

import io

import numpy as np
import pytest

from Pelagia.services.feature_space import (
    MAX_EXACT_VECTOR_SCAN,
    MAX_UMAP_VECTOR_SCAN,
    FeatureSpaceError,
    FeatureSpaceService,
    parse_feature_space_source,
)


def _npy_bytes(values: list[float]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


class _Store:
    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs

    def get_store(self, key: str) -> bytes:
        return self.blobs[key]


class _Repository:
    def __init__(self):
        self.embedding_calls: list[dict] = []
        self.summary_calls: list[list[str]] = []
        self.vector_count = 3
        self.embedding_rows = [
            {"refined_detection_id": "roi-reference", "embedding_payload_ref": "reference"},
            {"refined_detection_id": "roi-near", "embedding_payload_ref": "near"},
            {"refined_detection_id": "roi-far", "embedding_payload_ref": "far"},
        ]

    def list_feature_space_sources(self, *, project_id: str):
        assert project_id == "project-1"
        return [{"source_kind": "classification", "inference_run_id": "run-class", "embedding_count": 3}]

    def list_feature_space_embeddings(self, **values):
        self.embedding_calls.append(values)
        return self.embedding_rows

    def count_feature_space_embeddings(self, **values):
        return self.vector_count

    def list_feature_space_roi_summaries(self, *, project_id: str, roi_ids: list[str]):
        assert project_id == "project-1"
        self.summary_calls.append(roi_ids)
        return [
            {
                "id": roi_id,
                "asset_filename": f"{roi_id}.png",
                "bbox_w": 20,
                "bbox_h": 10,
                "roi_shape": [12, 24],
                "area": 123,
            }
            for roi_id in roi_ids
        ]

    def list_feature_space_clusters(self, **values):
        return [{"cluster_id": "cluster-a", "roi_count": 2, **values}]

    def list_feature_space_label_prototypes(self, **values):
        return [
            {
                "cluster_id": "label-prototype:4",
                "cluster_name": "Copepod",
                "roi_count": 2,
                **values,
            }
        ]

    def get_feature_space_cluster_assignment(self, **values):
        return {"cluster_id": "cluster-a"}

    def list_feature_space_cluster_members(self, **values):
        return {"items": [{"id": "roi-near", **values}], "total": 1, "limit": values["limit"], "offset": values["offset"]}

    def list_feature_space_label_prototype_members(self, **values):
        return {
            "items": [{"id": "roi-near", "cluster_name": "Copepod", **values}],
            "total": 1,
            "limit": values["limit"],
            "offset": values["offset"],
        }


class _Context:
    def __init__(self):
        self.repository = _Repository()
        self.store = _Store(
            {
                "reference": _npy_bytes([1.0, 0.0]),
                "near": _npy_bytes([0.8, 0.2]),
                "far": _npy_bytes([0.0, 1.0]),
            }
        )

    def kvstore_for_project(self, project_id: str, *, initialize: bool = True):
        assert project_id == "project-1"
        assert not initialize
        return self.store


def test_feature_space_similarity_is_exact_and_scoped_to_one_run():
    context = _Context()
    service = FeatureSpaceService(context, project_id="project-1")

    sources = service.sources()
    result = service.similar_rois(
        roi_id="roi-reference", source_key="classification:run-class", limit=2, minimum=0.0
    )

    assert sources[0]["source_key"] == "classification:run-class"
    assert sources[0]["scope"] == "single_inference_run"
    assert [item["id"] for item in result["items"]] == ["roi-reference", "roi-near"]
    assert result["items"][0]["is_reference"] is True
    assert result["items"][1]["similarity"] == pytest.approx(0.9701425)
    assert context.repository.embedding_calls == [
        {
            "project_id": "project-1",
            "source_kind": "classification",
            "inference_run_id": "run-class",
            "limit": 3,
        }
    ]
    assert result["search_scope"] == "full_source"
    assert result["scanned_vector_count"] == result["total_vector_count"] == 3


def test_feature_space_browse_rois_returns_source_scoped_reference_candidates():
    context = _Context()
    service = FeatureSpaceService(context, project_id="project-1")

    result = service.browse_rois(source_key="classification:run-class", limit=2)

    assert [item["id"] for item in result["items"]] == ["roi-reference", "roi-near", "roi-far"]
    assert result["source_key"] == "classification:run-class"
    assert result["limit"] == 2
    assert context.repository.embedding_calls == [
        {
            "project_id": "project-1",
            "source_kind": "classification",
            "inference_run_id": "run-class",
            "limit": 2,
        }
    ]


def test_feature_space_similarity_returns_a_deterministic_prefix_for_large_runs():
    context = _Context()
    context.repository.vector_count = MAX_EXACT_VECTOR_SCAN + 1
    service = FeatureSpaceService(context, project_id="project-1")

    result = service.similar_rois(
        roi_id="roi-reference", source_key="classification:run-class", limit=2, minimum=0.0
    )

    assert result["search_scope"] == "deterministic_prefix"
    assert result["total_vector_count"] == MAX_EXACT_VECTOR_SCAN + 1
    assert result["scanned_vector_count"] == 3
    assert context.repository.embedding_calls == [
        {
            "project_id": "project-1",
            "source_kind": "classification",
            "inference_run_id": "run-class",
            "limit": MAX_EXACT_VECTOR_SCAN,
        }
    ]


def test_feature_space_clustering_similarity_uses_recorded_cluster_membership():
    context = _Context()
    service = FeatureSpaceService(context, project_id="project-1")

    result = service.similar_rois(
        roi_id="roi-reference", source_key="clustering:run-cluster", limit=20, minimum=0.25
    )

    assert result["comparison"] == "cluster_centroid_similarity"
    assert result["cluster_id"] == "cluster-a"
    assert result["candidate_count"] == 1
    assert result["items"][0]["id"] == "roi-near"
    assert context.repository.embedding_calls == []


def test_feature_space_umap_returns_run_scoped_coordinates_and_hdbscan_assignments(monkeypatch):
    context = _Context()
    context.repository.vector_count = 5
    context.repository.embedding_rows = [
        {"refined_detection_id": f"roi-{index}", "embedding_payload_ref": f"pca-{index}"}
        for index in range(5)
    ]
    context.store.blobs.update(
        {f"pca-{index}": _npy_bytes(values) for index, values in enumerate([
            [0, 0, 0, 0], [1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4],
        ])}
    )

    service = FeatureSpaceService(context, project_id="project-1")
    monkeypatch.setattr(service, "_reduce_umap", lambda matrix: np.arange(len(matrix) * 10, dtype="float64").reshape(len(matrix), 10))
    hdbscan_calls: list[dict[str, object]] = []

    def cluster(coordinates, **kwargs):
        hdbscan_calls.append(kwargs)
        return np.array([0, 0, 1, -1, 1]), np.array([.9, .8, .7, 0, .6])

    monkeypatch.setattr(service, "_cluster_hdbscan", cluster)
    result = service.umap_rois(
        source_key="classification:run-class",
        min_cluster_size=7,
        min_samples=3,
        cluster_selection_epsilon=0.15,
    )

    assert result["source_key"] == "classification:run-class"
    assert result["projection"] == "umap"
    assert result["clustering"] == "hdbscan"
    assert result["random_seed"] == 20_260_826
    assert result["projection_scope"] == "full_source"
    assert result["compatible_embedding_count"] == 5
    assert len(result["items"]) == 5
    assert len(result["items"][0]["umap_coordinates"]) == 10
    assert result["items"][0]["hdbscan_cluster_id"] == "hdbscan:0"
    assert result["items"][3]["hdbscan_cluster_id"] is None
    assert result["cluster_count"] == 2
    assert result["noise_count"] == 1
    assert result["hdbscan_parameters"] == {
        "min_cluster_size": 7,
        "min_samples": 3,
        "cluster_selection_epsilon": 0.15,
        "metric": "euclidean",
    }
    assert hdbscan_calls == [{"min_cluster_size": 7, "min_samples": 3, "cluster_selection_epsilon": 0.15}]
    assert len(result["component_ranges"]) == 10
    assert result["items"][0]["bbox_w"] == 20
    assert result["items"][0]["bbox_h"] == 10


def test_feature_space_umap_excludes_incompatible_and_unreadable_embeddings(monkeypatch):
    context = _Context()
    context.repository.vector_count = 7
    context.repository.embedding_rows = [
        {"refined_detection_id": f"roi-{index}", "embedding_payload_ref": key}
        for index, key in enumerate(["pca-0", "pca-1", "pca-2", "pca-3", "pca-4", "wrong-shape", "missing"])
    ]
    context.store.blobs.update(
        {
            "pca-0": _npy_bytes([0, 0, 0, 0]), "pca-1": _npy_bytes([1, 0, 0, 0]),
            "pca-2": _npy_bytes([0, 1, 0, 0]), "pca-3": _npy_bytes([0, 0, 1, 0]),
            "pca-4": _npy_bytes([0, 0, 0, 1]),
            "wrong-shape": _npy_bytes([1, 2, 3]),
        }
    )

    service = FeatureSpaceService(context, project_id="project-1")
    monkeypatch.setattr(service, "_reduce_umap", lambda matrix: np.zeros((len(matrix), 10)))
    monkeypatch.setattr(service, "_cluster_hdbscan", lambda coordinates, **kwargs: (np.full(len(coordinates), -1), np.zeros(len(coordinates))))
    result = service.umap_rois(
        source_key="classification:run-class"
    )

    assert result["compatible_embedding_count"] == 5
    assert result["incompatible_embedding_count"] == 1
    assert result["unreadable_embedding_count"] == 1
    assert [item["id"] for item in result["items"]] == ["roi-0", "roi-1", "roi-2", "roi-3", "roi-4"]


def test_feature_space_umap_uses_a_smaller_deterministic_projection_cap(monkeypatch):
    context = _Context()
    context.repository.vector_count = MAX_UMAP_VECTOR_SCAN + 1
    context.repository.embedding_rows = [
        {"refined_detection_id": f"roi-{index}", "embedding_payload_ref": f"pca-{index}"}
        for index in range(5)
    ]
    context.store.blobs.update(
        {f"pca-{index}": _npy_bytes([index, 0, 0, 0]) for index in range(5)}
    )

    service = FeatureSpaceService(context, project_id="project-1")
    monkeypatch.setattr(service, "_reduce_umap", lambda matrix: np.zeros((len(matrix), 10)))
    monkeypatch.setattr(service, "_cluster_hdbscan", lambda coordinates, **kwargs: (np.full(len(coordinates), -1), np.zeros(len(coordinates))))
    result = service.umap_rois(
        source_key="classification:run-class"
    )

    assert result["projection_scope"] == "deterministic_prefix"
    assert context.repository.embedding_calls[-1]["limit"] == MAX_UMAP_VECTOR_SCAN


def test_feature_space_umap_requires_five_compatible_vectors():
    context = _Context()
    with pytest.raises(FeatureSpaceError, match="at least five compatible"):
        FeatureSpaceService(context, project_id="project-1").umap_rois(
            source_key="classification:run-class"
        )


def test_feature_space_uses_label_prototypes_for_classification_runs():
    context = _Context()
    service = FeatureSpaceService(context, project_id="project-1")

    with pytest.raises(FeatureSpaceError):
        parse_feature_space_source("classification:run-class:another-run")

    prototypes = service.clusters(source_key="classification:run-class")
    prototype_members = service.cluster_members(
        source_key="classification:run-class", cluster_id="label-prototype:4", limit=20, offset=0
    )
    clusters = service.clusters(source_key="clustering:run-cluster")
    members = service.cluster_members(
        source_key="clustering:run-cluster", cluster_id="cluster-a", limit=20, offset=0
    )

    assert prototypes["organization_kind"] == "label_prototypes"
    assert prototypes["items"][0]["cluster_name"] == "Copepod"
    assert prototype_members["organization_kind"] == "label_prototypes"
    assert prototype_members["items"][0]["cluster_name"] == "Copepod"
    assert clusters["source_key"] == "clustering:run-cluster"
    assert clusters["organization_kind"] == "self_supervised_clusters"
    assert members["source_key"] == "clustering:run-cluster"
    assert members["cluster_id"] == "cluster-a"
    assert members["organization_kind"] == "self_supervised_clusters"
