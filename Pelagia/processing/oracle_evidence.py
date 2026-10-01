"""Translate Oracle V2 image evidence into Pelagia's persisted ROI evidence fields."""

from __future__ import annotations

from typing import Any

import numpy as np


IMAGE_OUTPUT_SCHEMA = "oracle_builder.inference.v2.image_output"
IMAGE_OUTPUT_VERSION = "1.0.0"
_IMAGE_FIELDS = {
    "class_probabilities", "primary_decision", "embedding", "embedding_normalized",
    "knn", "prototype_similarity", "secondary_heads", "cluster_evidence",
}


def json_image_output(value: Any) -> Any:
    """Retain numeric V2 output arrays as inspectable typed JSON values."""
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.tolist()}
    if isinstance(value, dict):
        return {key: json_image_output(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_image_output(item) for item in value]
    return value


def normalize_image_output(output: dict[str, Any], *, expected_task: str) -> dict[str, Any]:
    """Validate a V2 result and expose fields used by existing evidence writers."""
    if output.get("schema_name") != IMAGE_OUTPUT_SCHEMA or output.get("schema_version") != IMAGE_OUTPUT_VERSION:
        raise ValueError("Oracle Builder returned an unsupported image output schema")
    if output.get("type") != "image_analysis" or output.get("model_task") != expected_task:
        raise ValueError(f"Oracle Builder did not return {expected_task} image evidence")
    if _IMAGE_FIELDS.difference(output):
        raise ValueError("Oracle Builder image output is missing required fields")
    probability_packet = output.get("class_probabilities")
    probabilities = [] if probability_packet is None else probability_packet.get("classes")
    if not isinstance(probabilities, list):
        raise ValueError("Oracle Builder class probabilities must be an indexed list")
    decision = output.get("primary_decision")
    if expected_task == "classification" and (not isinstance(decision, dict) or not probabilities):
        raise ValueError("Oracle Builder classification output is missing a decision or probabilities")
    if expected_task != "classification" and (decision is not None or probability_packet is not None):
        raise ValueError("Non-classification evidence must not contain class predictions")
    embedding_packet = output.get("embedding")
    embedding = None
    normalized = output.get("embedding_normalized")
    if embedding_packet is not None:
        if not isinstance(embedding_packet, dict):
            raise ValueError("Oracle Builder embedding must be a typed object")
        embedding = embedding_packet.get("values")
        if (not isinstance(embedding, np.ndarray) or embedding.ndim != 1
                or not np.issubdtype(embedding.dtype, np.number)
                or not np.isfinite(embedding).all()):
            raise ValueError("Oracle Builder embedding must be a finite numeric vector")
        if int(embedding_packet.get("dimension", -1)) != embedding.size or embedding_packet.get("dtype") != str(embedding.dtype):
            raise ValueError("Oracle Builder embedding dimension or dtype does not match its values")
        if not isinstance(embedding_packet.get("normalized"), bool):
            raise ValueError("Oracle Builder embedding normalization is missing")
        if normalized is not None and normalized != embedding_packet["normalized"]:
            raise ValueError("Oracle Builder embedding normalization fields disagree")
        normalized = embedding_packet["normalized"]
    elif expected_task in {"embedding", "clustering"}:
        raise ValueError(f"Oracle Builder {expected_task} output is missing an embedding")
    if expected_task == "clustering" and not isinstance(output.get("cluster_evidence"), dict):
        raise ValueError("Oracle Builder clustering output is missing cluster evidence")
    evidence = {
        "knn": output.get("knn"),
        "prototype": output.get("prototype_similarity"),
    }
    return {
        "type": expected_task,
        "probabilities": probabilities,
        "decision": decision or {},
        "embedding": embedding,
        "embedding_normalized": normalized,
        "evidence": evidence,
        "clustering_evidence": output.get("cluster_evidence"),
    }
