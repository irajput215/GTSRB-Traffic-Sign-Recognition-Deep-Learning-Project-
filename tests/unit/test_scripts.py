"""Unit tests for the standalone scripts.

The scripts are not part of the installed package, so they are loaded by path.
Testing them matters because they are what CI actually gates on: a quality gate
that can be satisfied by a missing or malformed result file is worse than no gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit


def load_script(name: str) -> ModuleType:
    """Import a script from ``scripts/`` by path."""
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quality() -> ModuleType:
    return load_script("check_model_quality")


@pytest.fixture(scope="module")
def configs() -> ModuleType:
    return load_script("validate_configs")


def write_result(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "model_name": "compact_cnn",
        "epochs_trained": 3,
        "best_metrics": {"val_accuracy": 0.9, "val_macro_f1": 0.85},
        "checkpoint_path": None,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestModelQualityGate:
    def test_passes_when_above_the_floor(self, quality: ModuleType, tmp_path: Path) -> None:
        result = write_result(tmp_path / "train_result.json")
        args = quality.parse_args(
            ["--result", str(result), "--monitor", "val_accuracy", "--minimum", "0.6"]
        )
        assert quality.check(args) == []

    def test_fails_when_below_the_floor(self, quality: ModuleType, tmp_path: Path) -> None:
        result = write_result(tmp_path / "train_result.json")
        args = quality.parse_args(
            ["--result", str(result), "--monitor", "val_accuracy", "--minimum", "0.95"]
        )
        failures = quality.check(args)
        assert any("below the required minimum" in message for message in failures)

    def test_fails_on_a_missing_result_file(self, quality: ModuleType, tmp_path: Path) -> None:
        """A missing result must not be treated as a pass."""
        args = quality.parse_args(["--result", str(tmp_path / "absent.json"), "--minimum", "0.6"])
        assert quality.check(args)

    def test_fails_on_malformed_json(self, quality: ModuleType, tmp_path: Path) -> None:
        result = tmp_path / "broken.json"
        result.write_text("{not json", encoding="utf-8")
        args = quality.parse_args(["--result", str(result), "--minimum", "0.6"])
        assert any("not valid JSON" in message for message in quality.check(args))

    def test_fails_when_the_metric_is_absent(self, quality: ModuleType, tmp_path: Path) -> None:
        result = write_result(tmp_path / "train_result.json", best_metrics={})
        args = quality.parse_args(["--result", str(result), "--minimum", "0.6"])
        assert any("is missing from best_metrics" in message for message in quality.check(args))

    def test_enforces_the_epoch_floor(self, quality: ModuleType, tmp_path: Path) -> None:
        result = write_result(tmp_path / "train_result.json", epochs_trained=0)
        args = quality.parse_args(
            ["--result", str(result), "--minimum", "0.6", "--min-epochs", "2"]
        )
        assert any("epoch(s) completed" in message for message in quality.check(args))

    def test_requires_the_checkpoint_to_exist(self, quality: ModuleType, tmp_path: Path) -> None:
        result = write_result(
            tmp_path / "train_result.json", checkpoint_path=str(tmp_path / "missing.pt")
        )
        args = quality.parse_args(
            ["--result", str(result), "--minimum", "0.6", "--require-artifacts"]
        )
        assert any("does not exist" in message for message in quality.check(args))

    def test_accepts_an_existing_checkpoint(self, quality: ModuleType, tmp_path: Path) -> None:
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"not a real checkpoint")
        result = write_result(tmp_path / "train_result.json", checkpoint_path=str(checkpoint))
        args = quality.parse_args(
            ["--result", str(result), "--minimum", "0.6", "--require-artifacts"]
        )
        assert quality.check(args) == []

    def test_main_returns_nonzero_on_failure(
        self, quality: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = write_result(tmp_path / "train_result.json")
        exit_code = quality.main(
            ["--result", str(result), "--monitor", "val_accuracy", "--minimum", "0.99"]
        )
        assert exit_code == 1
        assert "FAIL" in capsys.readouterr().err

    def test_main_returns_zero_on_success(
        self, quality: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = write_result(tmp_path / "train_result.json")
        exit_code = quality.main(
            ["--result", str(result), "--monitor", "val_accuracy", "--minimum", "0.5"]
        )
        assert exit_code == 0
        assert "PASS" in capsys.readouterr().out


class TestValidateConfigs:
    def test_shipped_configs_pass(self, configs: ModuleType, config_dir: Path) -> None:
        assert configs.validate(config_dir) == []

    def test_missing_layer_is_reported(self, configs: ModuleType, tmp_path: Path) -> None:
        failures = configs.validate(tmp_path)
        assert any("missing config layer" in message for message in failures)

    def test_malformed_layer_is_reported(self, configs: ModuleType, tmp_path: Path) -> None:
        for name in configs.LAYERS:
            (tmp_path / name).write_text("{}", encoding="utf-8")
        (tmp_path / "train.yaml").write_text("training:\n  epochs: -5\n", encoding="utf-8")
        failures = configs.validate(tmp_path)
        assert any("epochs" in message for message in failures)

    def test_main_returns_nonzero_on_failure(
        self, configs: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert configs.main([str(tmp_path)]) == 1
        assert "FAIL" in capsys.readouterr().err

    def test_main_returns_zero_for_valid_configs(
        self, configs: ModuleType, config_dir: Path
    ) -> None:
        assert configs.main([str(config_dir)]) == 0
