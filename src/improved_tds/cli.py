from __future__ import annotations

import argparse
from importlib import import_module


COMMANDS = {
    "convert-legacy": "improved_tds.experiments.convert_legacy",
    "collect-config": "improved_tds.experiments.collect_config",
    "collect": "improved_tds.experiments.collect",
    "collect-policy": "improved_tds.experiments.collect_policy",
    "evaluate-transfer": "improved_tds.experiments.evaluate_transfer",
    "fit-tds": "improved_tds.experiments.fit_tds",
    "calibrate": "improved_tds.experiments.calibrate",
    "evaluate": "improved_tds.experiments.evaluate",
    "train-sb3": "improved_tds.experiments.train_sb3",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="improved-tds")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, remaining = parser.parse_known_args(argv)
    import_module(COMMANDS[args.command]).main(remaining)


if __name__ == "__main__":
    main()
