# Scissors: palm-following sphere for grasp and success

The green sphere in the original MuJoCo single initial pose supplies the
reference geometry. At environment construction, before a pose-pool reset,
record its centre in palm-local coordinates:

\[
  r_0 = R_{p,0}^{\mathsf T}(c_{g,0}-p_{p,0}).
\]

At each later simulation step, including any authored initial pose, display
and evaluate the sphere at

\[
  c_g(t)=p_p(t)+R_p(t)r_0.
\]

The original sphere radius is unchanged (`grasp_threshold`, 0.05 m in the
current tool-manip-gym Scissors variants). In the canonical XML initial pose,
the resulting palm-local centre is `[0.07006666, -0.05188849, 0.0665]` m for
Scissors 3–5; it is measured from the model, not hard-coded. The measured point is the centre
of `scissors_hinge_2:joint` (`scissors2:center`), **not** the decorative
`scissors` marker. `is_on_palm` and `grasp_maintained` are true exactly when
`||p_hinge-c_g|| < grasp_threshold`. Contact counts, displacement, rotation,
drop, and tip are diagnostics only. The RL instantaneous success condition is
`is_on_palm AND angle_error <= angle_tolerance`; the existing five-step stable
success condition is unchanged. Numerical failure is still an invalid episode.

The Scissors reach reward uses the palm-sphere result for grasp shaping and
success bonus. An optional contact-only auxiliary term is separately controlled
by `--scissors-contact-auxiliary-weight` and defaults to zero. It never changes
grasp or success labels. `--scissors-interaction-weight 0` is used in the
paired auxiliary-reward pilot, so the pilot isolates the contact term.

The archived 2026-09-25 direct-TDS HTML and success-posture databases retain
their original contact-gated labels and are not overwritten. Any policy or
database trained under this rule must live in a new result directory. A
one-seed, eight-pose diagnostic of the archived PCA/PLS axes under the new
rule is saved in `analysis/direct_positive_range_palm_20260925`; its scores
are exploratory and do not replace the multi-seed TDS experiment.

For Scissors 3, at the archived positive score endpoint all 8/8 test resets
held the hinge inside the moving sphere for two seconds (PCA and PLS), even
though the old contact-gated report recorded no held point. Expanding the
positive score to three times that endpoint reached about 0.99 rad and
retained 7/8 PCA and 6/8 PLS trials. Scissors 4 had no two-second held point
in this isolated direct-posture probe; Scissors 5 gained held points at larger
scores. The plotted pooled endpoint span is not continuous calibration
coverage or an inverse-map validation.

Focused tests:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
& '..\fdms_reconfiguration\.venv\Scripts\python.exe' -m pytest tests/test_scissors_reach.py tests/test_scissors_smooth.py tests/test_simple_synergy.py tests/test_reach_tds_direct.py tests/test_training_validation.py -q --basetemp .codex_tmp/palm_pytest_validation_20260925
```

Paired pilot command (new output directory required):

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
& '..\fdms_reconfiguration\.venv\Scripts\python.exe' scripts/run_palm_reward_ablation.py
```

The pilot uses 100,000 transitions, four parallel environments, one paired
seed per variant, and the same 32-pose × five-target deterministic validation
grid. It compares contact auxiliary weights 0 and 0.25 while setting the
separate interaction reward to zero. Its purpose is to screen whether adding
the existing contact signal has a measurable benefit under the new palm-only
success rule. A paper claim requires independent seeds and a longer horizon.

The completed one-seed pilot (`runs/scissors_palm_aux_ablation_20260926/summary.md`)
reported stable-success rates of 43.8% versus 62.5% (Scissors 3), 63.1% versus
84.4% (Scissors 4), and 48.1% versus 91.9% (Scissors 5), for weights 0 versus
0.25 respectively. Mean angle error also decreased in all three variants.
However, final-step success did not improve in Scissors 5 (33.1% versus
32.5%). The full-training launcher and low-level CLI therefore retain zero
as the default, with 0.25 available as an explicit ablation setting. These
pilot results cannot establish multi-seed robustness or final 3M-step
superiority.
