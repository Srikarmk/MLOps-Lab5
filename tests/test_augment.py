"""The augmentation check: the lab's chosen configuration, retrained with augmentation off and on."""

import json

import pytest

from tuning import augment
from tuning.models import BASELINE, CNNHyperModel


def write_lab_run(run_dir, final_max_epochs):
    run_dir.mkdir(parents=True)
    chosen = {**BASELINE, "blocks": 2, "filters_1": 16, "filters_2": 16}
    del chosen["filters_3"]
    baseline_runs = [{"seed": 42, "test_accuracy": 0.3, "val_accuracy": 0.3, "best_epoch": 2}]
    (run_dir / "results.json").write_text(json.dumps({
        "seed": 42, "final_max_epochs": final_max_epochs, "patience": 3,
        "baseline": {"runs": baseline_runs}, "tuned": {"hyperparameters": chosen}}))
    return chosen


def test_the_chosen_configuration_is_rebuilt_exactly():
    values = {**BASELINE, "augment": True}
    model = CNNHyperModel().build(augment.fixed_hyperparameters(values))
    assert [layer.filters for layer in model.layers if layer.name.startswith("conv_")] == [32, 64, 64]
    assert {"augment_flip", "augment_shift"} <= {layer.name for layer in model.layers}


def test_latest_lab_run_skips_comparisons_and_needs_a_lab_run(tmp_path):
    with pytest.raises(SystemExit, match="Run the lab first"):
        augment.latest_lab_run(tmp_path)
    write_lab_run(tmp_path / "20260101-000000", 2)
    write_lab_run(tmp_path / "20260102-000000", 2)
    (tmp_path / "20260103-000000-compare").mkdir()
    assert augment.latest_lab_run(tmp_path).name == "20260102-000000"


@pytest.mark.parametrize("lab_epochs, compared_with_baseline", [(2, True), (30, False)])
def test_quick_check_runs_both_versions(small_data, tmp_path, lab_epochs, compared_with_baseline):
    write_lab_run(tmp_path / "20260101-000000", lab_epochs)
    output = augment.main(["--quick", "--runs-dir", str(tmp_path)])
    assert output["not_augmented"]["runs"][0]["seed"] == output["augmented"]["runs"][0]["seed"] == 42
    assert output["augmented_minus_not_augmented"]["seeds"] == 1
    # the baseline is only compared when the lab trained its final models the same way (here: 2 epochs)
    assert ("augmented_minus_baseline" in output) == compared_with_baseline
    assert (tmp_path / "20260101-000000" / "augment.json").exists()
