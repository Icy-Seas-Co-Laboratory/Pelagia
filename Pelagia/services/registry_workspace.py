from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .registry_transfer import RegistryTransferError, _jsonb


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _performance_metrics(value: Any) -> dict[str, str | int | float]:
    """Collect explicitly named model metrics without interpreting arbitrary numbers."""
    metric_groups = {
        "metrics", "performance", "performance_metrics", "evaluation", "evaluation_metrics",
        "validation_metrics", "test_metrics", "training_metrics", "scores",
    }
    metric_names = {
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "sensitivity",
        "f1", "f1_score", "auc", "auroc", "average_precision", "map", "loss", "error_rate",
        "r2", "rmse", "mae", "mse", "iou", "dice", "top1", "top5",
    }
    result: dict[str, str | int | float] = {}

    def walk(item: Any, path: list[str], inside_group: bool = False) -> None:
        if len(result) >= 200 or not isinstance(item, dict):
            return
        for key, child in item.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            child_path = [*path, str(key)]
            grouped = inside_group or normalized in metric_groups or normalized.endswith("_metrics")
            if isinstance(child, dict):
                walk(child, child_path, grouped)
            elif (
                (grouped or normalized in metric_names)
                and isinstance(child, (int, float, str))
                and not isinstance(child, bool)
            ):
                result[" · ".join(child_path)] = child

    walk(value, [])
    return result


class RegistryWorkspaceService:
    """Project- and user-scoped Registry operations over a loaded PostgreSQL workspace."""

    def __init__(self, repository, project_id: str, owner_username: str):
        self.repository = repository
        self.schema = repository.schema
        self.project_id = project_id
        self.owner_username = owner_username

    def _workspace(self, cursor, *, required: bool = True, for_update: bool = False) -> dict[str, Any] | None:
        cursor.execute(
            f"""SELECT * FROM {self.schema}.registry_workspaces
            WHERE project_id=%s AND owner_username=%s AND is_active=true AND status='loaded'
            ORDER BY loaded_at DESC LIMIT 1 {"FOR UPDATE" if for_update else ""}""",
            (self.project_id, self.owner_username),
        )
        row = cursor.fetchone()
        if row is None and required:
            raise RegistryTransferError("No Registry dataset is loaded for this project and user")
        return row

    def active_workspace(self) -> dict[str, Any] | None:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            return self._workspace(cursor, required=False)

    def list_workspaces(self) -> list[dict[str, Any]]:
        """Return the non-purged workspaces visible to the current project user."""
        with self.repository.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT w.id workspace_id,w.dataset_id,w.revision_id,w.parent_revision_id,
                w.dataset_type,w.name,w.title,w.description,w.dataset_version version,
                w.dataset_lifecycle lifecycle,w.source_path,w.source_size_bytes,w.status,w.is_active,
                w.dirty_at,w.loaded_at,w.exported_at,w.last_export_path,w.contract_schema_version,
                (SELECT count(*) FROM {self.schema}.registry_items i WHERE i.workspace_id=w.id) item_count,
                (SELECT count(DISTINCT a.item_id) FROM {self.schema}.registry_annotations a
                  WHERE a.workspace_id=w.id AND a.is_current AND a.status IN ('accepted','candidate')) labeled_count
                FROM {self.schema}.registry_workspaces w
                WHERE w.project_id=%s AND w.owner_username=%s AND w.status!='purged'
                ORDER BY w.is_active DESC,w.loaded_at DESC,w.id""",
                (self.project_id, self.owner_username),
            )
            result = [dict(row) for row in cursor.fetchall()]
            for workspace in result:
                workspace["workspace_id"] = str(workspace["workspace_id"])
            return result

    def activate_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Make an existing loaded workspace active within the current user/project scope."""
        with self.repository.connect() as connection, connection.cursor() as cursor:
            lock_key = f"registry:{self.project_id}:{self.owner_username}"
            cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (lock_key,))
            cursor.execute(
                f"""SELECT id FROM {self.schema}.registry_workspaces
                WHERE id=%s AND project_id=%s AND owner_username=%s AND status='loaded' FOR UPDATE""",
                (workspace_id, self.project_id, self.owner_username),
            )
            if cursor.fetchone() is None:
                raise KeyError(workspace_id)
            cursor.execute(
                f"UPDATE {self.schema}.registry_workspaces SET is_active=false WHERE project_id=%s AND owner_username=%s AND is_active=true",
                (self.project_id, self.owner_username),
            )
            cursor.execute(
                f"UPDATE {self.schema}.registry_workspaces SET is_active=true WHERE id=%s",
                (workspace_id,),
            )
            connection.commit()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            workspace_id = workspace["id"]
            cursor.execute(
                f"""SELECT count(*) total,
                count(*) FILTER (WHERE a.annotation_id IS NOT NULL AND a.status IN ('accepted','candidate')) labeled,
                count(*) FILTER (WHERE a.annotation_id IS NULL OR a.status NOT IN ('accepted','candidate')) unlabeled,
                count(*) FILTER (WHERE r.decision='verified') verified,
                count(*) FILTER (WHERE r.decision='needs_review') needs_review
                FROM {self.schema}.registry_items i
                LEFT JOIN {self.schema}.registry_annotations a ON a.workspace_id=i.workspace_id
                  AND a.item_id=i.item_id AND a.is_current
                LEFT JOIN LATERAL (
                  SELECT decision FROM {self.schema}.registry_reviews rr
                  WHERE rr.workspace_id=a.workspace_id AND rr.annotation_id=a.annotation_id
                  ORDER BY rr.created_at DESC LIMIT 1
                ) r ON true WHERE i.workspace_id=%s""",
                (workspace_id,),
            )
            stats = dict(cursor.fetchone())
            cursor.execute(
                f"SELECT count(*) count FROM {self.schema}.registry_labels WHERE workspace_id=%s AND deprecated_at IS NULL",
                (workspace_id,),
            )
            stats["active_labels"] = cursor.fetchone()["count"]
            cursor.execute(
                f"""SELECT l.label_id,coalesce(l.display_name,l.name) display_name,count(*) count
                FROM {self.schema}.registry_annotations a JOIN {self.schema}.registry_labels l
                  ON l.workspace_id=a.workspace_id AND l.label_id=a.label_id
                WHERE a.workspace_id=%s AND a.is_current AND a.status IN ('accepted','candidate')
                GROUP BY l.label_id,l.display_name,l.name ORDER BY count(*) DESC""",
                (workspace_id,),
            )
            return {
                "workspace_id": str(workspace_id), "dataset_id": workspace["dataset_id"],
                "revision_id": workspace["revision_id"], "parent_revision_id": workspace["parent_revision_id"],
                "dataset_type": workspace["dataset_type"], "name": workspace["name"],
                "title": workspace["title"], "description": workspace["description"],
                "version": workspace["dataset_version"], "lifecycle": workspace["dataset_lifecycle"],
                "path": workspace["source_path"], "dirty": workspace["dirty_at"] is not None,
                "stats": stats, "label_distribution": list(cursor.fetchall()),
                "physical_scale": {"available": False, "calibrated_items": 0, "total_items": stats["total"]},
            }

    def details(self) -> dict[str, Any]:
        summary = self.summary()
        sources = {source["source_key"]: source for source in self.evidence_sources()}
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            workspace_id = workspace["id"]
            counts = {}
            for name, table in {
                "items": "registry_items", "assets": "registry_assets", "events": "registry_dataset_events",
                "inference_runs": "registry_inference_runs", "model_evidence": "registry_model_evidence",
                "labels": "registry_labels", "descriptor_definitions": "registry_descriptors",
            }.items():
                cursor.execute(f"SELECT count(*) count FROM {self.schema}.{table} WHERE workspace_id=%s", (workspace_id,))
                counts[name] = cursor.fetchone()["count"]
            cursor.execute(
                f"""SELECT coalesce(media_type,'unknown') media_type,coalesce(encoding,'unknown') encoding,count(*) count
                FROM {self.schema}.registry_assets WHERE workspace_id=%s GROUP BY media_type,encoding ORDER BY count(*) DESC""",
                (workspace_id,),
            )
            formats = list(cursor.fetchall())
            cursor.execute(
                f"SELECT * FROM {self.schema}.registry_dataset_events WHERE workspace_id=%s ORDER BY created_at DESC LIMIT 12",
                (workspace_id,),
            )
            events = list(cursor.fetchall())
            cursor.execute(
                f"SELECT * FROM {self.schema}.registry_inference_runs WHERE workspace_id=%s ORDER BY created_at DESC",
                (workspace_id,),
            )
            inference_sources = []
            for row in cursor.fetchall():
                run = dict(row)
                source_key = f"registry:{run['inference_run_id']}"
                name = run.get("name") or run.get("model_artifact_id") or run["inference_run_id"]
                evidence = dict(sources.get(source_key) or {
                    "source_key": source_key, "source_kind": "registry", "source_name": name,
                    "item_count": 0, "embedding_count": 0, "confidence_count": 0,
                    "knn_count": 0, "prototype_count": 0,
                    "capabilities": {"confidence": False, "knn": False, "prototype": False, "embedding": False},
                })
                cursor.execute(
                    f"""SELECT avg(prediction_confidence) mean_confidence,
                    min(prediction_confidence) min_confidence,max(prediction_confidence) max_confidence,
                    avg(nearest_neighbor_similarity) mean_knn_similarity,
                    avg(top_k_label_agreement) mean_knn_agreement,
                    avg(weighted_label_support) mean_weighted_label_support,
                    avg(label_margin) mean_label_margin
                    FROM {self.schema}.registry_model_evidence
                    WHERE workspace_id=%s AND inference_run_id=%s""",
                    (workspace_id, run["inference_run_id"]),
                )
                evidence["aggregates"] = dict(cursor.fetchone())
                inference_sources.append({
                    "source_key": source_key, "source_kind": "registry", "name": name,
                    "evidence": evidence, "performance_metrics": _performance_metrics({"run": run}),
                    "run": run,
                })
        return {
            "dataset": {
                "dataset_id": workspace["dataset_id"], "revision_id": workspace["revision_id"],
                "parent_revision_id": workspace["parent_revision_id"], "dataset_type": workspace["dataset_type"],
                "name": workspace["name"], "title": workspace["title"], "description": workspace["description"],
                "version": workspace["dataset_version"], "lifecycle": workspace["dataset_lifecycle"],
                "metadata": workspace["metadata"],
            },
            "summary": summary,
            "database": {"path": "postgresql", "size_bytes": 0, "schema": {"name": workspace["contract_schema_name"], "version": workspace["contract_schema_version"]},
                         "table_count": len(counts), "tables": list(counts), "counts": counts},
            "asset_formats": formats, "recent_events": events, "inference_sources": inference_sources,
        }

    def labels(self, include_deprecated: bool = True) -> list[dict[str, Any]]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            cursor.execute(
                f"""SELECT l.*,
                count(a.annotation_id) FILTER (WHERE a.is_current AND a.status IN ('accepted','candidate')) item_count,
                count(a.annotation_id) FILTER (WHERE a.is_current AND r.decision='verified') verified_count,
                count(a.annotation_id) FILTER (WHERE a.is_current AND coalesce(r.decision,'')!='verified') unverified_count,
                (SELECT count(*) FROM {self.schema}.registry_model_evidence me
                  WHERE me.workspace_id=l.workspace_id AND me.predicted_label_id=l.label_id) ml_count
                FROM {self.schema}.registry_labels l
                LEFT JOIN {self.schema}.registry_annotations a ON a.workspace_id=l.workspace_id AND a.label_id=l.label_id
                LEFT JOIN LATERAL (SELECT decision FROM {self.schema}.registry_reviews rr
                  WHERE rr.workspace_id=a.workspace_id AND rr.annotation_id=a.annotation_id ORDER BY created_at DESC LIMIT 1) r ON true
                WHERE l.workspace_id=%s GROUP BY l.workspace_id,l.label_id
                ORDER BY (l.deprecated_at IS NOT NULL),coalesce(l.display_name,l.name)""",
                (workspace["id"],),
            )
            result = []
            for row in cursor.fetchall():
                item = dict(row)
                metadata = item.get("metadata") or {}
                registry = metadata.get("registry", {})
                item["display_name"] = item.get("display_name") or registry.get("display_name") or item["name"]
                item["deprecated_at"] = item.get("deprecated_at") or registry.get("deprecated_at")
                item["standard_concept_id"] = registry.get("standard_concept_id")
                vocabulary = registry.get("vocabulary", {})
                item["standard_vocabulary_key"] = f"{vocabulary.get('id')}@{vocabulary.get('version')}" if vocabulary.get("id") and vocabulary.get("version") else None
                item["preferred"] = bool(item["standard_concept_id"])
                if include_deprecated or not item["deprecated_at"]:
                    result.append(item)
            return result

    def add_label(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True)
            label_id = str(uuid.uuid4())
            cursor.execute(f"SELECT coalesce(max(class_index),-1)+1 value FROM {self.schema}.registry_labels WHERE workspace_id=%s", (workspace["id"],))
            class_index = cursor.fetchone()["value"] if workspace["dataset_type"] == "classification" else None
            metadata = data.get("metadata") or {}
            cursor.execute(
                f"""INSERT INTO {self.schema}.registry_labels
                (workspace_id,label_id,origin,class_index,name,display_name,parent_label_id,rank,description,metadata,created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (workspace["id"], label_id, "classification" if class_index is not None else "workspace", class_index,
                 data["name"].strip(), data.get("display_name"), data.get("parent_label_id"), data.get("rank"),
                 data.get("description"), _jsonb(metadata), utc_now()),
            )
            self._dirty(cursor, workspace["id"])
            connection.commit()
        return next(value for value in self.labels() if value["label_id"] == label_id)

    def update_label(self, label_id: str, data: dict[str, Any]) -> dict[str, Any]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True)
            cursor.execute(f"SELECT * FROM {self.schema}.registry_labels WHERE workspace_id=%s AND label_id=%s", (workspace["id"], label_id))
            row = cursor.fetchone()
            if not row:
                raise KeyError(label_id)
            deprecated_at = utc_now() if data.get("deprecated") is True else (None if data.get("deprecated") is False else row["deprecated_at"])
            cursor.execute(
                f"""UPDATE {self.schema}.registry_labels SET name=%s,display_name=%s,description=%s,deprecated_at=%s
                WHERE workspace_id=%s AND label_id=%s""",
                (data.get("name", row["name"]), data.get("display_name", row["display_name"]),
                 data.get("description", row["description"]), deprecated_at, workspace["id"], label_id),
            )
            self._dirty(cursor, workspace["id"]); connection.commit()
        return next(value for value in self.labels() if value["label_id"] == label_id)

    def reassign_label(self, source_label_id: str, target_label_id: str, actor: str,
                       *, deprecate_source: bool = True) -> dict[str, Any]:
        if source_label_id == target_label_id:
            raise RegistryTransferError("Source and target labels must be different")
        operation_id = str(uuid.uuid4()); created_at = utc_now(); count = 0
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True); wid = workspace["id"]
            cursor.execute(f"SELECT * FROM {self.schema}.registry_labels WHERE workspace_id=%s AND label_id=ANY(%s) FOR UPDATE",
                           (wid, [source_label_id, target_label_id]))
            labels = {row["label_id"]: row for row in cursor.fetchall()}
            if source_label_id not in labels or target_label_id not in labels:
                raise RegistryTransferError("Source or target label does not exist")
            if labels[target_label_id]["deprecated_at"]:
                raise RegistryTransferError("Target label is deprecated")
            cursor.execute(f"""SELECT * FROM {self.schema}.registry_annotations
              WHERE workspace_id=%s AND label_id=%s AND is_current AND status IN ('accepted','candidate') FOR UPDATE""",
              (wid, source_label_id))
            for previous in cursor.fetchall():
                cursor.execute(f"UPDATE {self.schema}.registry_annotations SET is_current=false WHERE workspace_id=%s AND annotation_id=%s",
                               (wid, previous["annotation_id"]))
                annotation_id = str(uuid.uuid4())
                cursor.execute(f"""INSERT INTO {self.schema}.registry_annotations
                  (workspace_id,annotation_id,item_id,label_id,origin,created_at,annotator,method,status,is_current,
                   parent_annotation_id,parameters,notes,metadata)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,'bulk-reassign','accepted',true,%s,%s,%s,%s)""",
                  (wid, annotation_id, previous["item_id"], target_label_id, previous["origin"], created_at, actor,
                   previous["annotation_id"], _jsonb({}), "Bulk label reassignment",
                   _jsonb({"registry_operation_id": operation_id, "source_label_id": source_label_id,
                           "target_label_id": target_label_id})))
                count += 1
            if deprecate_source:
                cursor.execute(f"UPDATE {self.schema}.registry_labels SET deprecated_at=%s WHERE workspace_id=%s AND label_id=%s",
                               (created_at, wid, source_label_id))
            self._dirty(cursor, wid); connection.commit()
        return {"source_label_id": source_label_id, "target_label_id": target_label_id,
                "reassigned_count": count, "source_deprecated": deprecate_source, "operation_id": operation_id}

    def tags(self) -> list[dict[str, Any]]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            cursor.execute(
                f"""SELECT d.descriptor_id tag_id,CASE d.scope WHEN 'target' THEN 'target_tags' ELSE 'image_tags' END scope,
                d.name,d.parent_descriptor_id parent_tag_id,d.concept_id,d.concept_type,d.selectable,
                d.exclusive_within_parent,d.preferred,d.metadata,d.created_at,d.deprecated_at,
                count(a.annotation_id) FILTER (WHERE a.is_current AND a.status='accepted') item_count
                FROM {self.schema}.registry_descriptors d LEFT JOIN {self.schema}.registry_descriptor_annotations a
                  ON a.workspace_id=d.workspace_id AND a.descriptor_id=d.descriptor_id
                WHERE d.workspace_id=%s GROUP BY d.workspace_id,d.descriptor_id ORDER BY d.scope,d.preferred DESC,d.name""",
                (workspace["id"],),
            )
            return list(cursor.fetchall())

    def add_tag(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True); tag_id = str(uuid.uuid4())
            cursor.execute(
                f"""INSERT INTO {self.schema}.registry_descriptors
                (workspace_id,descriptor_id,scope,name,parent_descriptor_id,concept_type,selectable,exclusive_within_parent,preferred,metadata,created_at)
                VALUES (%s,%s,%s,%s,%s,'custom',true,%s,false,%s,%s)""",
                (workspace["id"], tag_id, data["scope"].removesuffix("_tags"), data["name"].strip(),
                 data.get("parent_tag_id"), bool(data.get("exclusive_within_parent")), _jsonb({"registry": {"origin": "user"}}), utc_now()),
            )
            self._dirty(cursor, workspace["id"]); connection.commit()
        return next(value for value in self.tags() if value["tag_id"] == tag_id)

    def list_items(self, *, limit: int, offset: int, annotation_state: str = "all", review: str = "all",
                   label_ids: list[str] | None = None, search: str | None = None, sort: str = "original",
                   evidence_source: str | None = None, item_ids: list[str] | None = None, **filters) -> dict[str, Any]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor); params: list[Any] = [workspace["id"]]; clauses = ["i.workspace_id=%s"]
            if annotation_state == "labeled": clauses.append("a.annotation_id IS NOT NULL AND a.status IN ('accepted','candidate')")
            elif annotation_state == "unlabeled": clauses.append("(a.annotation_id IS NULL OR a.status NOT IN ('accepted','candidate'))")
            if review in {"verified", "rejected", "needs_review"}: clauses.append("r.decision=%s"); params.append(review)
            elif review == "unreviewed": clauses.append("a.annotation_id IS NOT NULL AND r.decision IS NULL")
            if label_ids: clauses.append("a.label_id=ANY(%s)"); params.append(label_ids)
            if search: clauses.append("(i.source_key ILIKE %s OR i.item_id ILIKE %s)"); params.extend([f"%{search}%", f"%{search}%"])
            if item_ids is not None:
                if not item_ids:
                    return {"items": [], "total": 0, "limit": limit, "offset": offset}
                clauses.append("i.item_id=ANY(%s)"); params.append(item_ids)
            if evidence_source:
                if filters.get("ml_state") == "with": clauses.append("me.item_id IS NOT NULL")
                elif filters.get("ml_state") == "without": clauses.append("me.item_id IS NULL")
                for key, operator in (("confidence_min", ">="), ("confidence_max", "<="),
                                      ("knn_similarity_min", ">="), ("knn_agreement_min", ">=")):
                    value = filters.get(key)
                    if value is not None:
                        column = {"confidence_min":"prediction_confidence", "confidence_max":"prediction_confidence",
                                  "knn_similarity_min":"nearest_neighbor_similarity", "knn_agreement_min":"top_k_label_agreement"}[key]
                        clauses.append(f"me.{column}{operator}%s"); params.append(value)
                if filters.get("ml_disagreement"):
                    clauses.append("me.predicted_label_id IS NOT NULL AND a.label_id IS NOT NULL AND me.predicted_label_id!=a.label_id")
            where = " AND ".join(clauses)
            order = {"original":"i.ordinal", "annotation":"(a.annotation_id IS NOT NULL) DESC,i.ordinal",
                     "label":"coalesce(l.display_name,l.name,''),i.ordinal", "random":"md5(i.item_id)",
                     "image_area_asc":"coalesce((image.shape->>0)::int,0)*coalesce((image.shape->>1)::int,0),i.ordinal",
                     "image_area_desc":"coalesce((image.shape->>0)::int,0)*coalesce((image.shape->>1)::int,0) DESC,i.ordinal",
                     "ml_confidence_desc":"me.prediction_confidence DESC NULLS LAST,i.ordinal",
                     "ml_confidence_asc":"me.prediction_confidence NULLS LAST,i.ordinal",
                     "ml_knn_desc":"me.nearest_neighbor_similarity DESC NULLS LAST,i.ordinal",
                     "ml_knn_asc":"me.nearest_neighbor_similarity NULLS LAST,i.ordinal",
                     "ml_agreement_asc":"me.top_k_label_agreement NULLS LAST,i.ordinal"}.get(sort, "i.ordinal")
            source_run = evidence_source.removeprefix("registry:") if evidence_source else None
            params_with_source = [source_run, *params]
            base = f"""FROM {self.schema}.registry_items i
              JOIN {self.schema}.registry_assets image ON image.workspace_id=i.workspace_id AND image.asset_id=i.image_asset_id
              LEFT JOIN {self.schema}.registry_annotations a ON a.workspace_id=i.workspace_id AND a.item_id=i.item_id AND a.is_current
              LEFT JOIN {self.schema}.registry_labels l ON l.workspace_id=a.workspace_id AND l.label_id=a.label_id
              LEFT JOIN LATERAL (SELECT decision FROM {self.schema}.registry_reviews rr WHERE rr.workspace_id=a.workspace_id
                AND rr.annotation_id=a.annotation_id ORDER BY created_at DESC LIMIT 1) r ON true
              LEFT JOIN {self.schema}.registry_model_evidence me ON me.workspace_id=i.workspace_id AND me.item_id=i.item_id
                AND me.inference_run_id=%s WHERE {where}"""
            cursor.execute("SELECT count(DISTINCT i.item_id) total " + base, params_with_source)
            total = cursor.fetchone()["total"]
            cursor.execute(f"""SELECT i.item_id,i.source_key,i.sample_weight,i.metadata,image.shape,image.encoding,image.media_type,
              a.annotation_id,a.label_id,a.annotator,a.created_at annotation_created_at,a.status annotation_status,a.method annotation_source,
              l.name label_name,coalesce(l.display_name,l.name) label_display_name,l.origin label_origin,r.decision review_decision,
              CASE WHEN me.inference_run_id IS NULL THEN NULL ELSE 'registry:'||me.inference_run_id END evidence_source,
              me.predicted_label_id ml_predicted_label_id,me.prediction_confidence,me.nearest_neighbor_similarity,
              me.top_k_label_agreement,me.weighted_label_support,me.label_margin,
              CASE WHEN me.embedding_array_id IS NULL THEN 0 ELSE 1 END embedding_available,
              nullif(image.shape->>0,'')::int image_height,nullif(image.shape->>1,'')::int image_width
              {base} ORDER BY {order} LIMIT %s OFFSET %s""", [*params_with_source, limit, offset])
            items = list(cursor.fetchall())
            self._attach_tags(cursor, workspace["id"], items)
            for item in items:
                h, w = item.get("image_height") or 0, item.get("image_width") or 0
                item["pixel_area"] = h * w; item["longest_side"] = max(h, w)
            return {"items": items, "total": total, "limit": limit, "offset": offset}

    def _attach_tags(self, cursor, workspace_id, items: list[dict[str, Any]]) -> None:
        by_item = {item["item_id"]: [] for item in items}
        for item in items: item["descriptor_indicators"] = by_item[item["item_id"]]
        if not by_item: return
        cursor.execute(
            f"""SELECT a.item_id,d.descriptor_id tag_id,CASE d.scope WHEN 'target' THEN 'target_tags' ELSE 'image_tags' END scope,d.name
            FROM {self.schema}.registry_descriptor_annotations a JOIN {self.schema}.registry_descriptors d
              ON d.workspace_id=a.workspace_id AND d.descriptor_id=a.descriptor_id
            WHERE a.workspace_id=%s AND a.item_id=ANY(%s) AND a.is_current AND a.status='accepted' ORDER BY d.scope,d.name""",
            (workspace_id, list(by_item)),
        )
        for row in cursor.fetchall(): by_item[row["item_id"]].append(dict(row))

    def item_detail(self, item_id: str) -> dict[str, Any]:
        result = self.list_items(limit=1, offset=0, item_ids=[item_id])
        # list_items deliberately accepts extra filters; enforce the identity here.
        item = next((value for value in result["items"] if value["item_id"] == item_id), None)
        if item is None:
            with self.repository.connect() as connection, connection.cursor() as cursor:
                workspace = self._workspace(cursor)
                cursor.execute(f"SELECT * FROM {self.schema}.registry_items WHERE workspace_id=%s AND item_id=%s", (workspace["id"], item_id))
                item = cursor.fetchone()
        if item is None: raise KeyError(item_id)
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            cursor.execute(f"""SELECT a.*,l.name label_name,coalesce(l.display_name,l.name) label_display_name,l.metadata label_metadata,l.origin label_origin
              FROM {self.schema}.registry_annotations a JOIN {self.schema}.registry_labels l ON l.workspace_id=a.workspace_id AND l.label_id=a.label_id
              WHERE a.workspace_id=%s AND a.item_id=%s ORDER BY a.created_at DESC""", (workspace["id"], item_id))
            item["annotations"] = list(cursor.fetchall())
            cursor.execute(f"""SELECT d.descriptor_id tag_id,CASE d.scope WHEN 'target' THEN 'target_tags' ELSE 'image_tags' END scope,d.name,
              d.parent_descriptor_id parent_tag_id,d.concept_type,d.preferred,d.metadata,a.annotation_id,a.annotator,a.created_at
              FROM {self.schema}.registry_descriptor_annotations a JOIN {self.schema}.registry_descriptors d ON d.workspace_id=a.workspace_id AND d.descriptor_id=a.descriptor_id
              WHERE a.workspace_id=%s AND a.item_id=%s AND a.is_current AND a.status='accepted'""", (workspace["id"], item_id))
            item["tags"] = list(cursor.fetchall())
        return item

    def image(self, item_id: str, size: int | None = None) -> tuple[bytes, str]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            cursor.execute(f"""SELECT a.payload,a.media_type,a.encoding FROM {self.schema}.registry_items i
              JOIN {self.schema}.registry_assets a ON a.workspace_id=i.workspace_id AND a.asset_id=i.image_asset_id
              WHERE i.workspace_id=%s AND i.item_id=%s""", (workspace["id"], item_id))
            row = cursor.fetchone()
        if not row or row["payload"] is None: raise KeyError(item_id)
        payload = bytes(row["payload"]); media = row["media_type"] or "application/octet-stream"
        if size is None: return payload, media
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None: return payload, media
        scale = min(1.0, size / max(image.shape[:2])); resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 86])
        return (encoded.tobytes(), "image/jpeg") if ok else (payload, media)

    def assign(self, item_ids: list[str], label_id: str, actor: str, *, remove: bool = False) -> list[dict[str, Any]]:
        created = []; operation_id = str(uuid.uuid4())
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True); wid = workspace["id"]
            for item_id in dict.fromkeys(item_ids):
                cursor.execute(f"SELECT * FROM {self.schema}.registry_annotations WHERE workspace_id=%s AND item_id=%s AND is_current FOR UPDATE", (wid, item_id))
                previous = cursor.fetchone()
                if remove and not previous: continue
                resolved_label = previous["label_id"] if remove else label_id
                cursor.execute(f"UPDATE {self.schema}.registry_annotations SET is_current=false WHERE workspace_id=%s AND item_id=%s AND is_current", (wid, item_id))
                annotation_id = str(uuid.uuid4()); status = "deprecated" if remove else "accepted"
                cursor.execute(f"""INSERT INTO {self.schema}.registry_annotations
                  (workspace_id,annotation_id,item_id,label_id,origin,created_at,annotator,method,status,is_current,parent_annotation_id,metadata)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,true,%s,%s)""",
                  (wid, annotation_id, item_id, resolved_label, "classification" if workspace["dataset_type"] == "classification" else "workspace",
                   utc_now(), actor, "manual-removal" if remove else "manual", status,
                   previous["annotation_id"] if previous else None, _jsonb({"registry_operation_id": operation_id})))
                created.append({"item_id": item_id, "annotation_id": annotation_id, "operation_id": operation_id})
            self._dirty(cursor, wid); connection.commit()
        return created

    def review(self, item_ids: list[str], decision: str, reviewer: str) -> list[dict[str, Any]]:
        created = []
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True); wid = workspace["id"]
            for item_id in dict.fromkeys(item_ids):
                cursor.execute(f"SELECT * FROM {self.schema}.registry_annotations WHERE workspace_id=%s AND item_id=%s AND is_current", (wid, item_id))
                annotation = cursor.fetchone()
                if not annotation or annotation["status"] == "deprecated": continue
                review_id = str(uuid.uuid4())
                cursor.execute(f"""INSERT INTO {self.schema}.registry_reviews
                  (workspace_id,review_id,annotation_id,origin,reviewer,decision,created_at,metadata)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                  (wid, review_id, annotation["annotation_id"], annotation["origin"], reviewer, decision, utc_now(), _jsonb({})))
                created.append({"item_id": item_id, "review_id": review_id})
            self._dirty(cursor, wid); connection.commit()
        return created

    def set_tag(self, item_ids: list[str], tag_id: str, assigned: bool, actor: str) -> list[dict[str, Any]]:
        created = []
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True); wid = workspace["id"]
            for item_id in dict.fromkeys(item_ids):
                cursor.execute(f"SELECT * FROM {self.schema}.registry_descriptor_annotations WHERE workspace_id=%s AND item_id=%s AND descriptor_id=%s AND is_current FOR UPDATE", (wid, item_id, tag_id))
                previous = cursor.fetchone(); desired = "accepted" if assigned else "deprecated"
                if previous and previous["status"] == desired: continue
                cursor.execute(f"UPDATE {self.schema}.registry_descriptor_annotations SET is_current=false WHERE workspace_id=%s AND item_id=%s AND descriptor_id=%s AND is_current", (wid, item_id, tag_id))
                annotation_id = str(uuid.uuid4())
                cursor.execute(f"""INSERT INTO {self.schema}.registry_descriptor_annotations
                  (workspace_id,annotation_id,item_id,descriptor_id,created_at,annotator,status,is_current,parent_annotation_id,metadata)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,true,%s,%s)""",
                  (wid, annotation_id, item_id, tag_id, utc_now(), actor, desired, previous["annotation_id"] if previous else None, _jsonb({})))
                created.append({"item_id": item_id, "tag_id": tag_id, "annotation_id": annotation_id, "assigned": assigned})
            self._dirty(cursor, wid); connection.commit()
        return created

    def undo(self, actor: str) -> dict[str, Any]:
        """Append compensating annotations for the most recent bulk label operation."""
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True); wid = workspace["id"]
            cursor.execute(f"""SELECT metadata->>'registry_operation_id' operation_id,max(created_at) created_at
              FROM {self.schema}.registry_annotations WHERE workspace_id=%s AND metadata ? 'registry_operation_id'
              GROUP BY metadata->>'registry_operation_id' ORDER BY created_at DESC LIMIT 1""", (wid,))
            operation = cursor.fetchone()
            if not operation: raise RegistryTransferError("There is no Registry label operation to undo")
            cursor.execute(f"""SELECT * FROM {self.schema}.registry_annotations
              WHERE workspace_id=%s AND metadata->>'registry_operation_id'=%s AND is_current FOR UPDATE""",
              (wid, operation["operation_id"]))
            current_rows = list(cursor.fetchall()); created = []
            for current in current_rows:
                cursor.execute(f"SELECT * FROM {self.schema}.registry_annotations WHERE workspace_id=%s AND annotation_id=%s",
                               (wid, current["parent_annotation_id"]))
                previous = cursor.fetchone()
                cursor.execute(f"UPDATE {self.schema}.registry_annotations SET is_current=false WHERE workspace_id=%s AND annotation_id=%s",
                               (wid, current["annotation_id"]))
                if previous:
                    annotation_id = str(uuid.uuid4())
                    cursor.execute(f"""INSERT INTO {self.schema}.registry_annotations
                      (workspace_id,annotation_id,item_id,label_id,origin,created_at,annotator,method,confidence,status,is_current,
                       parent_annotation_id,parameters,notes,metadata) VALUES (%s,%s,%s,%s,%s,%s,%s,'undo',%s,%s,true,%s,%s,%s,%s)""",
                      (wid, annotation_id, current["item_id"], previous["label_id"], previous["origin"], utc_now(), actor,
                       previous["confidence"], previous["status"], current["annotation_id"], _jsonb(previous["parameters"]),
                       f"Undo {operation['operation_id']}", _jsonb({"registry_undo_of": operation["operation_id"]})))
                    created.append(annotation_id)
            self._dirty(cursor, wid); connection.commit()
            return {"undone_operation_id": operation["operation_id"], "compensating_annotations": created,
                    "unlabeled_items": len(current_rows) - len(created)}

    def evidence_sources(self) -> list[dict[str, Any]]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            cursor.execute(f"""SELECT r.inference_run_id,r.name,r.model_artifact_id,count(e.evidence_id) item_count,
              count(e.embedding_array_id) embedding_count,count(e.prediction_confidence) confidence_count,
              count(*) FILTER (WHERE e.packet ? 'neighbors') knn_count
              FROM {self.schema}.registry_inference_runs r LEFT JOIN {self.schema}.registry_model_evidence e
              ON e.workspace_id=r.workspace_id AND e.inference_run_id=r.inference_run_id
              WHERE r.workspace_id=%s GROUP BY r.workspace_id,r.inference_run_id""", (workspace["id"],))
            return [{"source_key": f"registry:{r['inference_run_id']}", "source_kind": "registry",
                     "source_name": r["name"] or r["model_artifact_id"] or r["inference_run_id"],
                     "item_count": r["item_count"], "embedding_count": r["embedding_count"], "confidence_count": r["confidence_count"],
                     "knn_count": r["knn_count"], "prototype_count": 0,
                     "capabilities": {"confidence": bool(r["confidence_count"]), "knn": bool(r["knn_count"]),
                                      "prototype": False, "embedding": bool(r["embedding_count"])} } for r in cursor.fetchall()]

    def item_evidence(self, item_id: str, source_key: str | None) -> list[dict[str, Any]]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor); params = [workspace["id"], item_id]; clause = ""
            if source_key: clause = "AND e.inference_run_id=%s"; params.append(source_key.removeprefix("registry:"))
            cursor.execute(f"""SELECT e.*,r.name FROM {self.schema}.registry_model_evidence e JOIN {self.schema}.registry_inference_runs r
              ON r.workspace_id=e.workspace_id AND r.inference_run_id=e.inference_run_id
              WHERE e.workspace_id=%s AND e.item_id=%s {clause}""", params)
            result = []
            for row in cursor.fetchall():
                packet = row["packet"] or {}; result.append({**row, "source_key": f"registry:{row['inference_run_id']}",
                    "source_kind": "registry", "source_name": row["name"] or row["inference_run_id"],
                    "embedding_available": int(bool(row["embedding_array_id"])), "neighbors": packet.get("neighbors", [])})
            return result

    def neighbor_items(self, item_id: str, source_key: str) -> dict[str, Any]:
        evidence = self.item_evidence(item_id, source_key)
        neighbors = evidence[0].get("neighbors", []) if evidence else []
        ids = list(dict.fromkeys([item_id, *(str(row.get("uuid")) for row in neighbors if row.get("uuid"))]))
        result = self.list_items(limit=max(1, len(ids)), offset=0, evidence_source=source_key, item_ids=ids)
        scores = {str(row.get("uuid")): row.get("similarity") for row in neighbors}; scores[item_id] = 1.0
        positions = {value: index for index, value in enumerate(ids)}
        result["items"].sort(key=lambda row: positions.get(row["item_id"], len(ids)))
        for row in result["items"]: row["similarity"] = scores.get(row["item_id"])
        result.update(total=len(result["items"]), reference_item_id=item_id, evidence_source=source_key, mode="recorded-knn")
        return result

    def similar_items(self, item_id: str, source_key: str, *, limit: int, offset: int, minimum: float) -> dict[str, Any]:
        run_id = source_key.removeprefix("registry:")
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor)
            cursor.execute(f"""SELECT e.item_id,a.payload FROM {self.schema}.registry_model_evidence e
              JOIN {self.schema}.registry_evidence_arrays a ON a.workspace_id=e.workspace_id AND a.array_id=e.embedding_array_id
              WHERE e.workspace_id=%s AND e.inference_run_id=%s ORDER BY e.item_id""", (workspace["id"], run_id))
            ids, vectors = [], []
            for row in cursor.fetchall():
                try: vector = np.asarray(np.load(io.BytesIO(bytes(row["payload"])), allow_pickle=False), dtype=np.float32).reshape(-1)
                except (ValueError, TypeError, OSError): continue
                norm = float(np.linalg.norm(vector))
                if norm > 0 and np.isfinite(vector).all(): ids.append(row["item_id"]); vectors.append(vector / norm)
        if item_id not in ids or not vectors: raise RegistryTransferError("This evidence source has no readable embedding for the item")
        matrix = np.vstack(vectors); scores = matrix @ matrix[ids.index(item_id)]
        ranked = sorted(((ids[index], float(score)) for index, score in enumerate(scores) if score >= minimum), key=lambda value: value[1], reverse=True)
        page = ranked[offset:offset + limit]; page_ids = [value[0] for value in page]
        result = self.list_items(limit=max(1, len(page_ids)), offset=0, evidence_source=source_key, item_ids=page_ids)
        positions = {value: index for index, value in enumerate(page_ids)}; score_map = dict(page)
        result["items"].sort(key=lambda row: positions.get(row["item_id"], len(page_ids)))
        for row in result["items"]: row["similarity"] = score_map[row["item_id"]]
        result.update(total=len(ranked), limit=limit, offset=offset, reference_item_id=item_id, evidence_source=source_key)
        return result

    def purge(self, workspace_id: str | None = None) -> dict[str, Any]:
        with self.repository.connect() as connection, connection.cursor() as cursor:
            workspace = self._workspace(cursor, for_update=True) if workspace_id is None else None
            resolved = str(workspace["id"]) if workspace else workspace_id
            cursor.execute(f"""SELECT id FROM {self.schema}.registry_workspaces
              WHERE id=%s AND project_id=%s AND owner_username=%s FOR UPDATE""",
              (resolved, self.project_id, self.owner_username))
            if not cursor.fetchone(): raise RegistryTransferError("Registry workspace was not found")
            deleted = 0
            for table in (
                "registry_reviews", "registry_descriptor_annotations", "registry_mask_annotations",
                "registry_model_evidence", "registry_annotations", "registry_items", "registry_labels",
                "registry_descriptors", "registry_inference_runs", "registry_evidence_arrays",
                "registry_dataset_events", "registry_contract_records", "registry_assets",
            ):
                cursor.execute(f"DELETE FROM {self.schema}.{table} WHERE workspace_id=%s", (resolved,))
                deleted += max(0, cursor.rowcount)
            cursor.execute(f"""UPDATE {self.schema}.registry_workspaces SET status='purged',is_active=false
              WHERE id=%s AND project_id=%s AND owner_username=%s RETURNING id""", (resolved, self.project_id, self.owner_username))
            cursor.fetchone()
            connection.commit()
            return {"workspace_id": resolved, "status": "purged", "deleted_rows": deleted}

    def _dirty(self, cursor, workspace_id) -> None:
        cursor.execute(f"UPDATE {self.schema}.registry_workspaces SET dirty_at=NOW() WHERE id=%s", (workspace_id,))
