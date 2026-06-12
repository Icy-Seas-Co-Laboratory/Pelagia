from pathlib import Path

import numpy as np
import pytest

from Pelagia.config import CoreConfig
from Pelagia.processing.detection_refinement import IdentityRoiRefinementModel
from Pelagia.processing.oracle_unet_refiner import (
    OracleBuilderUnetRefinementModel,
    load_oracle_builder_unet_run,
    normalize_oracle_unet_batch,
    oracle_unet_prediction_to_probability_mask,
    refinement_model_from_config,
    refinement_options_from_config,
)


ORACLE_RUN_DIR = Path(__file__).resolve().parents[2] / "oracle-builder" / "runs" / "unet-test"


class FakePredictor:
    def __init__(self):
        self.seen = None

    def predict(self, batch):
        self.seen = np.array(batch, copy=True)
        return batch[..., 1, None] * 0.75


def test_oracle_unet_batch_normalization_matches_training_contract():
    batch = np.zeros((1, 2, 2, 2), dtype=np.uint8)
    batch[..., 0] = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    batch[..., 1] = np.array([[0, 255], [1, 0]], dtype=np.uint8)

    normalized = normalize_oracle_unet_batch(batch)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[0, :, :, 0], [[0.0, 128 / 255], [1.0, 64 / 255]])
    np.testing.assert_array_equal(normalized[0, :, :, 1], [[0.0, 1.0], [1.0, 0.0]])


def test_oracle_unet_prediction_output_accepts_keras_channel_shape():
    prediction = np.ones((2, 4, 4, 1), dtype=np.float32) * 1.5

    mask = oracle_unet_prediction_to_probability_mask(
        prediction,
        expected_batch_size=2,
        output_shape=(4, 4, 1),
    )

    assert mask.shape == (2, 4, 4)
    assert mask.dtype == np.float32
    assert float(mask.max()) == 1.0


def test_oracle_unet_adapter_uses_validated_run_shape_with_fake_predictor():
    if not ORACLE_RUN_DIR.exists():
        pytest.skip(f"oracle-builder test run not available at {ORACLE_RUN_DIR}")
    run = load_oracle_builder_unet_run(ORACLE_RUN_DIR)
    predictor = FakePredictor()
    model = OracleBuilderUnetRefinementModel(ORACLE_RUN_DIR, predictor=predictor, run=run)
    batch = np.zeros((1, run.input_shape[0], run.input_shape[1], 2), dtype=np.uint8)
    batch[:, 20:30, 40:50, 1] = 255

    prediction = model.predict(batch)

    assert prediction.shape == (1, run.output_shape[0], run.output_shape[1])
    assert float(prediction[:, 20:30, 40:50].max()) == 0.75
    assert predictor.seen is not None
    assert float(predictor.seen[..., 0].max()) == 0.0
    assert float(predictor.seen[..., 1].max()) == 1.0


def test_oracle_unet_artifact_metadata_matches_pelagia_refinement_options():
    if not ORACLE_RUN_DIR.exists():
        pytest.skip(f"oracle-builder test run not available at {ORACLE_RUN_DIR}")
    run = load_oracle_builder_unet_run(ORACLE_RUN_DIR)
    model = OracleBuilderUnetRefinementModel(ORACLE_RUN_DIR, predictor=FakePredictor(), run=run)

    options = model.refinement_options(overlap_fraction=0.5)

    assert run.input_shape == (256, 256, 2)
    assert run.output_shape == (256, 256, 1)
    assert options.tile_size == 256
    assert options.overlap_fraction == 0.5


def test_roi_refinement_config_helpers_resolve_packaged_default_model():
    config = CoreConfig.load(local_config_path=None, use_env=False)

    model = refinement_model_from_config(config)
    options = refinement_options_from_config(config)

    assert not isinstance(model, IdentityRoiRefinementModel)
    assert model.method_name.startswith("keras_artifact:builtin:model/roi_refinement/example_model")
    assert options.tile_size == 256
    assert options.output_threshold == 0.5
    assert options.overlap_reconciliation_enabled is True


def test_oracle_unet_savedmodel_artifact_predicts_when_ml_stack_is_available():
    if not ORACLE_RUN_DIR.exists():
        pytest.skip(f"oracle-builder test run not available at {ORACLE_RUN_DIR}")
    pytest.importorskip("tensorflow")
    run = load_oracle_builder_unet_run(ORACLE_RUN_DIR)
    model = OracleBuilderUnetRefinementModel(ORACLE_RUN_DIR, artifact="savedmodel", run=run)
    batch = np.zeros((1, run.input_shape[0], run.input_shape[1], 2), dtype=np.float32)

    prediction = model.predict(batch)

    assert prediction.shape == (1, run.output_shape[0], run.output_shape[1])
    assert prediction.dtype == np.float32
