# Self-Pruning Neural Network — Tredence AI Engineer Case Study

A feed-forward neural network for CIFAR-10 that learns to prune itself *during* training. Each weight is multiplied by a learnable sigmoid gate; an L1 penalty on the gates drives most of them to zero, yielding a sparse network with no post-training pruning step.

## Files

- `self_pruning_nn.py` — required single-file deliverable: `PrunableLinear` module, `PrunableMLP` definition, training + evaluation loop, λ sweep.
- `report.md` — required report: explanation of the L1-on-sigmoid-gates intuition, results table across three λ values, and a gate-value distribution plot.
- `results/metrics.json` — raw numbers backing the report.
- `results/gate_distribution.png` — histogram of final gate values for the best run.
- `tests/test_prunable_linear.py` — smoke tests verifying the gated-weight mechanism and gradient flow.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full experiment

```bash
python self_pruning_nn.py
```

Defaults: 5,000 stratified CIFAR-10 training samples, full 10k test set, 10 epochs per λ ∈ {1e-5, 5e-5, 2e-4}, Adam with separate learning rates (lr_weights=1e-3, lr_gates=5e-2 — see `report.md` for why), batch size 128, seed 42, gradient clipping at norm 5.0. Auto-selects MPS (Apple Silicon) or CPU.

Produces `results/metrics.json` and `results/gate_distribution.png`.

## Run the smoke tests

```bash
pytest tests/ -v
```

## Report

See [`report.md`](report.md) for the analysis, results table, and plot.
