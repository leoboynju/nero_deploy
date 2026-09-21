from pathlib import Path

from nero_deploy.config import load_config


def test_default_config_has_three_cameras() -> None:
    config = load_config(Path(__file__).parents[1] / "config/nero.yaml")
    assert set(config["cameras"]) >= {"left_wrist", "right_wrist", "third_person"}
    assert config["policy"]["port"] == 8000
