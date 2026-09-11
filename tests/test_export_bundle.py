from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from Pelagia.services.exports.bundle import ExportBundleWriter, verify_bundle
from Pelagia.services.exports import products
from Pelagia.services.exports.xlsx import write_xlsx
from Pelagia.services.exports.snapshot import prepare_export_snapshot, assert_snapshot_matches
from Pelagia.workers.stages import export as export_stage


def test_bundle_registers_direct_product_files_and_verifies(tmp_path: Path) -> None:
    export_id = str(uuid4())
    destination = tmp_path / "result.zip"
    with ExportBundleWriter(export_id, destination) as bundle:
        direct = bundle.root / "products" / "roi-statistics" / "assets" / str(uuid4()) / "statistics.json"
        direct.parent.mkdir(parents=True)
        direct.write_bytes(b"[]\n")
        result = bundle.finalize(
            project_id=str(uuid4()), request={}, products={}, versions={}, readme="# Export\n",
        )
    assert verify_bundle(destination)["valid"] is True
    assert "products/roi-statistics" in " ".join(result["manifest"]["files"])
    with zipfile.ZipFile(destination) as archive:
        assert f"{export_id}/checksums.sha256" in archive.namelist()


def test_bundle_allows_output_larger_than_an_advisory_estimate(tmp_path: Path) -> None:
    export_id = str(uuid4())
    destination = tmp_path / "result.zip"
    with ExportBundleWriter(export_id, destination) as bundle:
        bundle.write_bytes("products/example.bin", b"too-large")
        bundle.finalize(project_id=str(uuid4()), request={}, products={}, versions={}, readme="# Export\n")
    assert verify_bundle(destination)["valid"] is True


def test_xlsx_writer_emits_native_numbers_and_safe_formula_strings() -> None:
    contents = write_xlsx({"Measurements": [{"area_px2": 12.5, "note": "=not-a-formula"}]})
    with zipfile.ZipFile(__import__("io").BytesIO(contents)) as archive:
        xml = archive.read("xl/worksheets/sheet1.xml").decode()
    assert '<v>12.5</v>' in xml
    assert "'=not-a-formula" in xml


def test_xlsx_writer_partitions_oversized_logical_table(monkeypatch) -> None:
    import Pelagia.services.exports.xlsx as xlsx

    monkeypatch.setattr(xlsx, "MAX_XLSX_ROWS", 3)
    contents = xlsx.write_xlsx({"ROI statistics": [{"roi_id": index} for index in range(3)]})
    with zipfile.ZipFile(__import__("io").BytesIO(contents)) as archive:
        workbook = archive.read("xl/workbook.xml").decode()
        first = archive.read("xl/worksheets/sheet1.xml").decode()
        second = archive.read("xl/worksheets/sheet2.xml").decode()
    assert "ROI statistics_001" in workbook
    assert "ROI statistics_002" in workbook
    assert first.count("<row ") == 3  # header plus two data rows
    assert second.count("<row ") == 2  # header plus remainder


def test_product_orchestrator_uses_uuid_root_and_config_owned_destination(tmp_path: Path, monkeypatch) -> None:
    export_id, project_id = str(uuid4()), str(uuid4())
    seen: list[Path] = []

    def fake_raw(**kwargs):
        root = kwargs["output_root"]
        path = root / "products" / "roi-statistics" / "assets" / str(uuid4()) / "statistics.json"
        path.parent.mkdir(parents=True); path.write_text("[]\n")
        seen.append(root)
        return {"product": "roi-statistics", "schema_version": "1.0", "asset_count": 0, "roi_count": 0}

    monkeypatch.setattr(products, "write_raw_roi_statistics", fake_raw)
    context = SimpleNamespace(
        config=SimpleNamespace(artifacts=SimpleNamespace(local_root=tmp_path / "artifacts")),
        repository=object(), kvstore_for_project=lambda project: None,
    )
    reporter = SimpleNamespace(checkpoint=lambda: None, update=lambda *args, **kwargs: None)
    result = products.build_export_bundle(
        job={"project_id": project_id}, context=context,
        artifact={"id": export_id, "request": {"products": ["raw_roi_statistics"]}}, reporter=reporter,
    )
    assert Path(result["path"]) == tmp_path / "artifacts" / "exports" / project_id / f"{export_id}.zip"
    assert seen and seen[0].name == export_id
    assert verify_bundle(Path(result["path"]))["valid"]


def test_export_work_progress_uses_frozen_input_granularity(tmp_path: Path, monkeypatch) -> None:
    export_id, project_id, asset_id = str(uuid4()), str(uuid4()), str(uuid4())
    progress_updates = []

    def fake_evidence(**kwargs):
        callback = kwargs["progress_callback"]
        callback(1, 2, "Exported first ROI")
        callback(2, 2, "Exported second ROI")
        root = kwargs["output_root"]
        path = root / "products" / "roi-evidence" / "complete.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
        return {"product": "roi-evidence", "schema_version": "1.0", "roi_count": 2}

    monkeypatch.setattr(products, "write_roi_evidence", fake_evidence)
    monkeypatch.setattr(products, "assert_snapshot_matches", lambda *_args, **_kwargs: {"roi_ids": []})
    artifact = {
        "id": export_id,
        "request": {"products": ["roi_evidence"]},
        "input_snapshot": {"roi_items": [{"roi_id": str(uuid4()), "asset_id": asset_id}, {"roi_id": str(uuid4()), "asset_id": asset_id}]},
    }
    assert products.estimate_export_work_units(artifact) == 5  # read + write each ROI, plus publication
    context = SimpleNamespace(
        config=SimpleNamespace(artifacts=SimpleNamespace(local_root=tmp_path / "artifacts")),
        repository=object(), kvstore_for_project=lambda project: None,
    )
    reporter = SimpleNamespace(
        checkpoint=lambda: None,
        update=lambda completed, **kwargs: progress_updates.append((completed, kwargs)),
    )
    products.build_export_bundle(job={"project_id": project_id}, context=context, artifact=artifact, reporter=reporter)
    assert any(completed == 1 and update["message"] == "Exported first ROI" for completed, update in progress_updates)
    assert any(completed == 2 and update["message"] == "Exported second ROI" for completed, update in progress_updates)
    assert progress_updates[-1][0] == 4


def test_snapshot_freezes_roi_membership_and_detects_changed_payload(monkeypatch) -> None:
    project_id, roi_id, asset_id, frame_id = map(lambda _value: str(uuid4()), range(4))
    row = {
        "id": roi_id, "source_asset_id": asset_id, "frame_id": frame_id,
        "roi_payload": b"immutable", "source_asset_checksum": "sha256:asset",
    }
    monkeypatch.setattr("Pelagia.services.exports.snapshot.registry_generation._selected_rows", lambda *_args: iter([dict(row)]))
    request = {"products": ["raw_roi_statistics"], "asset_ids": [asset_id], "roi_stage": "refined"}
    snapshot, estimate = prepare_export_snapshot(SimpleNamespace(), project_id=project_id, request=request)
    assert estimate["counts"]["roi_count"] == 1
    assert estimate["advisory"] is True
    assert snapshot["roi_items"][0]["roi_id"] == roi_id
    assert assert_snapshot_matches(SimpleNamespace(), project_id=project_id, snapshot=snapshot)["roi_ids"] == [roi_id]
    row["roi_payload"] = b"changed"
    with __import__("pytest").raises(ValueError, match="changed after export submission"):
        assert_snapshot_matches(SimpleNamespace(), project_id=project_id, snapshot=snapshot)


def test_export_worker_records_retry_pending_attempt(monkeypatch) -> None:
    export_id, project_id = str(uuid4()), str(uuid4())
    calls = []

    class Repository:
        def get_export_artifact(self, *_args, **_kwargs):
            return {"id": export_id, "request": {"products": ["raw_roi_statistics"]}}

        def begin_export_attempt(self, *_args, **kwargs):
            calls.append(("begin", kwargs))

        def update_export_attempt_product(self, *_args, **kwargs):
            calls.append(("product", kwargs))

        def finish_export_attempt(self, *_args, **kwargs):
            calls.append(("finish", kwargs))

        def update_export_artifact(self, *_args, **kwargs):
            calls.append(("artifact", kwargs))
            return {"id": export_id, "request": {"products": ["raw_roi_statistics"]},
                    "input_snapshot": {"roi_items": [], "telemetry_items": []}}

    def fail(**_kwargs):
        raise RuntimeError("temporary storage outage")

    monkeypatch.setattr("Pelagia.services.exports.products.build_export_bundle", fail)
    monkeypatch.setattr("Pelagia.services.exports.snapshot.prepare_export_snapshot", lambda *_args, **_kwargs: (
        {"roi_items": [], "telemetry_items": []}, {"advisory": True},
    ))
    context = SimpleNamespace(repository=Repository(), config=SimpleNamespace())
    with __import__("pytest").raises(RuntimeError, match="temporary storage outage"):
        export_stage.handle({"id": str(uuid4()), "project_id": project_id, "attempt_count": 1,
                             "max_attempts": 2, "payload": {"export_id": export_id}}, context)
    assert calls[0][0] == "begin"
    assert calls[-1] == ("finish", {"project_id": project_id, "attempt_number": 1,
                                     "status": "failed", "error": "temporary storage outage", "retry_pending": True})
