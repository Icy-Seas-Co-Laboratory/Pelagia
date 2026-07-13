from types import SimpleNamespace

from Pelagia.config import CoreConfig
from Pelagia.services.project_settings import (
    invalidate_project_settings,
    merge_project_settings,
    resolve_project_storage_settings,
    storage_settings_payload,
)


class _Repository:
    def __init__(self, project):
        self.project = project

    def get_project(self, project_id):
        return self.project


def _context(project):
    return SimpleNamespace(config=CoreConfig(), repository=_Repository(project))


def test_project_storage_settings_override_global_defaults():
    settings = {
        "storage": {
            "frame": {"encoding": "jpeg-xl", "quality": 72},
            "roi": {"encoding": "png"},
        }
    }

    resolved = resolve_project_storage_settings(_context({"settings": settings}), "project-1")

    assert resolved.frame_encoding == "jxl"
    assert resolved.frame_quality == 72
    assert resolved.roi_encoding == "png"
    assert resolved.frame_encoding_source == "project"
    assert resolved.roi_encoding_source == "project"


def test_explicit_storage_overrides_take_precedence_over_project_settings():
    project = {
        "settings": {
            "storage": {
                "frame": {"encoding": "png", "quality": 10},
                "roi": {"encoding": "zstd"},
            }
        }
    }

    resolved = resolve_project_storage_settings(
        _context(project),
        "project-1",
        frame_encoding="jpg",
        frame_quality=65,
        roi_encoding="auto",
    )

    assert (resolved.frame_encoding, resolved.frame_quality, resolved.roi_encoding) == ("jpg", 65, "auto")
    assert resolved.frame_encoding_source == "override"
    assert resolved.frame_quality_source == "override"
    assert resolved.roi_encoding_source == "override"


def test_legacy_project_metadata_remains_a_frame_storage_fallback():
    project = {"metadata": {"processing": {"frame_storage": {"image_encoding": "raw"}}}}

    resolved = resolve_project_storage_settings(_context(project), "project-1")

    assert resolved.frame_encoding == "raw"
    assert resolved.frame_encoding_source == "legacy-project"
    assert resolved.roi_encoding == "zstd"


def test_project_storage_setting_patches_merge_nested_sections():
    existing = {"storage": {"frame": {"encoding": "png", "quality": 90}, "roi": {"encoding": "zstd"}}}
    patch = storage_settings_payload(frame_encoding="jpeg-xs", roi_encoding="auto")

    merged = merge_project_settings(existing, patch)

    assert merged["storage"]["frame"] == {"encoding": "jxs", "quality": 90}
    assert merged["storage"]["roi"] == {"encoding": "auto"}


def test_project_settings_resolution_caches_and_can_be_invalidated():
    repository = _Repository({"settings": {"storage": {"frame": {"encoding": "png"}}}})
    repository.calls = 0
    original_get_project = repository.get_project

    def get_project(project_id):
        repository.calls += 1
        return original_get_project(project_id)

    repository.get_project = get_project
    context = SimpleNamespace(config=CoreConfig(), repository=repository)

    assert resolve_project_storage_settings(context, "project-1").frame_encoding == "png"
    assert resolve_project_storage_settings(context, "project-1").frame_encoding == "png"
    assert repository.calls == 1

    invalidate_project_settings(context, "project-1")
    resolve_project_storage_settings(context, "project-1")
    assert repository.calls == 2
