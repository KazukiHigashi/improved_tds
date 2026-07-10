from __future__ import annotations

import ast
from pathlib import Path


def test_core_does_not_import_optional_simulation_or_training_packages() -> None:
    root = Path(__file__).parents[1] / "src" / "improved_tds"
    core_directories = [root / name for name in ("control", "data", "evaluation", "synergy", "tools")]
    forbidden = {"gymnasium", "mujoco", "stable_baselines3"}
    violations = []
    for directory in core_directories:
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in forbidden:
                        violations.append(f"{path.name}: {name}")
    assert not violations

