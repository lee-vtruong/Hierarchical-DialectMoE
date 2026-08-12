from pathlib import Path

from scripts.run_h16_validation import discover_epochs as discover_validation_epochs
from scripts.select_h16_checkpoint import discover_epochs as discover_selection_epochs


def test_h16_validation_scripts_exist():
    assert Path("scripts/run_h16_validation.py").is_file()
    assert Path("scripts/select_h16_checkpoint.py").is_file()


def test_h16_discovers_all_numeric_checkpoints(tmp_path):
    for name in ["epoch_0.pt", "epoch_29.pt", "epoch_30.pt", "epoch_59.pt"]:
        (tmp_path / name).touch()
    (tmp_path / "epoch_best.pt").touch()

    expected = [0, 29, 30, 59]
    assert discover_validation_epochs(tmp_path) == expected
    assert discover_selection_epochs(tmp_path) == expected
