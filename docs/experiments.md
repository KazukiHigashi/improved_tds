# Experiment workflow

Install simulation and developer dependencies with `python -m pip install -e '.[dev]'`.
Seeds are explicit in collection/training commands.

```bash
python -m improved_tds.experiments.collect \
  --env-id TDS-Trigger-v0 --episodes 20 --seed 0 \
  --output runs/trigger/data.npz

python -m improved_tds.experiments.fit_tds \
  --dataset runs/trigger/data.npz --method pca \
  --output runs/trigger/tds_pca.npz

python -m improved_tds.experiments.fit_tds \
  --dataset runs/trigger/data.npz --method pls \
  --output runs/trigger/tds_pls.npz

python -m improved_tds.experiments.calibrate \
  --dataset runs/trigger/data.npz --model runs/trigger/tds_pls.npz \
  --method isotonic --shots 6 --output runs/trigger/calibration.npz

python -m improved_tds.experiments.evaluate \
  --dataset runs/trigger/data.npz --model runs/trigger/tds_pls.npz \
  --calibration runs/trigger/calibration.npz

python -m improved_tds.experiments.evaluate_transfer \
  --manifest configs/scissors/instances.json --method pls --shots 3,6,12 \
  --output runs/scissors/transfer.json
```

SB3 DDPG+HER is optional: install `.[training]` and run
`python -m improved_tds.experiments.train_sb3`. Fast unit/simulation tests run with
`pytest`; long RL training is intentionally not part of the fast suite.

