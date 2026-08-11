from pathlib import Path


def test_h16_validation_scripts_exist():
    assert Path("scripts/run_h16_validation.py").is_file()
    assert Path("scripts/select_h16_checkpoint.py").is_file()

