from __future__ import annotations

import argparse

from improved_tds.data.legacy import LegacySynergyConverter


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert legacy synergy_dataset.npy to NPZ")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--tool-family", default="scissors")
    parser.add_argument("--instance-id", default="legacy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    dataset = LegacySynergyConverter.convert(
        args.input,
        tool_family=args.tool_family,
        tool_instance_id=args.instance_id,
        seed=args.seed,
    )
    dataset.save(args.output)
    print(f"converted {len(dataset.trajectories)} terminal samples to {args.output}")


if __name__ == "__main__":
    main()

