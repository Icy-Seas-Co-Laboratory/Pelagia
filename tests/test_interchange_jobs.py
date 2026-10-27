from types import SimpleNamespace

import pytest

from Pelagia.domain import PipelineStage
from Pelagia.workers.stages import interchange


def test_interchange_worker_calls_creator_and_protects_destination(tmp_path, monkeypatch):
    source = tmp_path / 'videos'
    source.mkdir()
    output = tmp_path / 'archive'
    calls = []

    class Reporter:
        def __init__(self, *args, **kwargs): pass
        def start(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass

    def create(source_path, output_path, **kwargs):
        calls.append((source_path, output_path, kwargs))
        kwargs['progress']('processing')
        return SimpleNamespace(frames=12, acquisition_segments=2, previews=1)

    monkeypatch.setattr(interchange, 'JobProgressReporter', Reporter)
    monkeypatch.setattr('pelagia_interchange.ingest_video_directory', create)
    job = {'payload': {'source_path': str(source), 'output_path': str(output), 'jpeg_quality': 80}}
    result = interchange.handle(job, None)
    assert result['frames'] == 12
    assert calls[0][2]['jpeg_quality'] == 80
    assert output.is_dir()
    with pytest.raises(FileExistsError):
        interchange.handle(job, None)
    assert len(calls) == 1


def test_interchange_worker_rejects_overlapping_paths(tmp_path):
    source = tmp_path / 'videos'
    source.mkdir()
    with pytest.raises(ValueError):
        interchange.handle({'payload': {'source_path': str(source), 'output_path': str(source / 'archive')}}, None)
    assert not (source / 'archive').exists()


def test_interchange_worker_is_registered():
    from Pelagia.workers.registry import default_stage_handlers
    assert dict(default_stage_handlers())[PipelineStage.INTERCHANGE_CREATE] is interchange.handle
