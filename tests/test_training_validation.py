import json

import pytest

from improved_tds.experiments import validation
from improved_tds.experiments.collect_posture_database import _fractions as database_fractions
from improved_tds.experiments.train_sb3 import _fractions as training_fractions
from improved_tds.experiments.validation import (
    ValidationEarlyStoppingCallback,
    validation_result_is_better,
)


def _result(success_rate: float, error: float) -> dict[str, float]:
    return {"success_rate": success_rate, "median_abs_error": error}


def test_full_range_fraction_parsers_include_both_endpoints() -> None:
    assert database_fractions("0 0.5 1") == (0.0, 0.5, 1.0)
    assert training_fractions("0 0.5 1") == (0.0, 0.5, 1.0)
    with pytest.raises(Exception):
        database_fractions("-0.1 1")


def test_validation_best_policy_prefers_success_rate() -> None:
    assert validation_result_is_better(_result(0.96, 0.02), _result(0.95, 0.001))
    assert not validation_result_is_better(_result(0.94, 0.001), _result(0.95, 0.02))


def test_validation_best_policy_uses_error_only_for_ties() -> None:
    assert validation_result_is_better(_result(0.95, 0.004), _result(0.95, 0.006))
    assert not validation_result_is_better(_result(0.95, 0.008), _result(0.95, 0.006))


def test_validation_best_policy_prefers_worst_target_coverage() -> None:
    broad = {
        **_result(0.80, 0.02),
        "min_target_success_rate": 0.50,
        "target_coverage_fraction": 1.0,
    }
    narrow = {
        **_result(0.90, 0.01),
        "min_target_success_rate": 0.0,
        "target_coverage_fraction": 0.8,
    }
    assert validation_result_is_better(broad, narrow)
    assert not validation_result_is_better(narrow, broad)


def test_callback_saves_best_and_stops_after_two_successes(tmp_path, monkeypatch) -> None:
    results = iter(
        [
            {
                "episodes": 100,
                "successes": 95,
                "success_rate": 0.95,
                "median_abs_error": 0.01,
                "mean_abs_error": 0.02,
            },
            {
                "episodes": 100,
                "successes": 96,
                "success_rate": 0.96,
                "median_abs_error": 0.008,
                "mean_abs_error": 0.01,
            },
        ]
    )
    monkeypatch.setattr(validation, "score_policy_on_grid", lambda *args: next(results))

    class DummyLogger:
        def record(self, *args) -> None:
            pass

    class DummyModel:
        logger = DummyLogger()

        def __init__(self) -> None:
            self.saved: list[str] = []

        def save(self, path) -> None:
            self.saved.append(str(path))

    class DummyEnv:
        closed = False

        def close(self) -> None:
            self.closed = True

    model = DummyModel()
    env = DummyEnv()
    report = tmp_path / "validation_history.json"
    callback = ValidationEarlyStoppingCallback(
        validation_env=env,
        poses=[0],
        targets=[0.1],
        option_key="target",
        eval_freq=100_000,
        success_threshold=0.95,
        consecutive_evaluations=2,
        best_policy_path=tmp_path / "best_policy",
        report_path=report,
        verbose=0,
    )
    callback.model = model

    callback.num_timesteps = 100_000
    assert callback._on_step()
    callback.num_timesteps = 200_000
    assert not callback._on_step()

    assert model.saved == [
        str(tmp_path / "best_pose_coverage_policy"),
        str(tmp_path / "best_policy"),
        str(tmp_path / "best_success_policy"),
        str(tmp_path / "best_pose_coverage_policy"),
        str(tmp_path / "best_policy"),
        str(tmp_path / "best_success_policy"),
    ]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["stopped_early"] is True
    assert payload["success_streak"] == 2
    assert payload["best"]["timesteps"] == 200_000

    callback._on_training_end()
    assert env.closed


def test_success_and_coverage_policies_can_select_different_steps(tmp_path, monkeypatch):
    results=iter([
        dict(episodes=100,successes=30,success_rate=.3,median_abs_error=.1,
             mean_abs_error=.1,min_target_success_rate=.1,target_coverage_fraction=1.),
        dict(episodes=100,successes=70,success_rate=.7,median_abs_error=.02,
             mean_abs_error=.02,min_target_success_rate=0.,target_coverage_fraction=.8),
    ])
    monkeypatch.setattr(validation,'score_policy_on_grid',lambda *args:next(results))
    class Model:
        class logger:
            @staticmethod
            def record(*args): pass
        def __init__(self): self.saved=[]
        def save(self,path): self.saved.append(str(path))
    model=Model()
    cb=ValidationEarlyStoppingCallback(validation_env=object(),poses=[0],targets=[0,1],
        option_key='target',eval_freq=100,success_threshold=.95,consecutive_evaluations=2,
        best_policy_path=tmp_path/'best_policy',report_path=tmp_path/'history.json',verbose=0)
    cb.model=model
    cb.num_timesteps=100
    cb._on_step()
    cb.num_timesteps=200
    cb._on_step()
    assert cb.best['success_rate']==.7
    assert cb.best_coverage['success_rate']==.3
    assert model.saved.count(str(tmp_path/'best_pose_coverage_policy'))==1
    assert model.saved.count(str(tmp_path/'best_success_policy'))==2
