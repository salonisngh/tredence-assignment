# Self-Pruning Neural Network — Design Spec

**Date:** 2026-04-22
**Author:** Rishabh Sharma (rishabh@trytaketwo.com)
**Context:** Tredence Analytics — AI Engineering Intern case study ("The Self-Pruning Neural Network")

## Problem

Build a feed-forward neural network that learns to prune its own weights *during* training, rather than as a post-training step. Each weight is associated with a learnable scalar "gate" in [0, 1]; an L1 penalty on the gates is added to the classification loss to drive most gates to zero, producing a sparse network.

Deliverables mandated by the case study:

1. A single, well-commented Python script containing the `PrunableLinear` module, the model definition, and the training/evaluation loop.
2. A markdown report with:
   - A short explanation of why L1 on sigmoid gates encourages sparsity.
   - A results table (Lambda, Test Accuracy, Sparsity Level %).
   - A matplotlib histogram of the final gate-value distribution for the best model.

## Success Criteria

- `PrunableLinear` correctly implements the gated weight mechanism; gradients flow through both `weight` and `gate_scores`.
- Training loop computes `total_loss = CE + λ · sum(sigmoid(gate_scores))` and backpropagates correctly.
- Results demonstrate sparsity increases monotonically with λ.
- Gate distribution plot shows clear bimodality (spike near 0, cluster elsewhere).
- Reviewer can run the script end-to-end with the provided instructions.

## Execution Scope (Proof Run)

To demonstrate correctness without a long training run, the default configuration uses:

- CIFAR-10, 5,000 stratified training samples + full 10,000 test set (`torchvision.datasets.CIFAR10`).
- 5 epochs per λ value, batch size 128, Adam optimizer, lr = 1e-3.
- λ sweep: **{1e-5, 1e-4, 1e-3}** (low / medium / high).
- Fixed seed (42) for reproducibility.
- Device auto-select: MPS (Apple Silicon) if available, else CPU.

## Architecture

### Project Layout

```
project2/
├── self_pruning_nn.py         # required single-file deliverable
├── report.md                  # required markdown report
├── results/
│   ├── gate_distribution.png  # histogram of final gate values (best run)
│   └── metrics.json           # raw numbers that back the report table
├── requirements.txt
├── README.md
└── .gitignore                 # excludes data/, .venv/, __pycache__, *.pyc
```

### `PrunableLinear(nn.Module)`

- Signature: `PrunableLinear(in_features, out_features)`
- Parameters (all `nn.Parameter`, so the optimizer updates them):
  - `weight`: shape `(out_features, in_features)`, Kaiming-uniform init (matches `nn.Linear`).
  - `bias`: shape `(out_features,)`, zero init.
  - `gate_scores`: shape `(out_features, in_features)`, initialized to a positive constant (≈ 2.0) so that `sigmoid(2.0) ≈ 0.88` at start — the network has near-full capacity initially and *learns* to prune.
- Forward pass:
  1. `gates = torch.sigmoid(self.gate_scores)`
  2. `pruned_weights = self.weight * gates`
  3. Return `F.linear(input, pruned_weights, self.bias)`
- Exposes `gates()` for logging and evaluation (reads sigmoid of current gate_scores).

**Gradient flow:** Autograd handles this automatically because the forward pass uses only differentiable ops on parameters. No custom backward needed. The spec's "challenge" is satisfied simply by not detaching anything.

### `PrunableMLP(nn.Module)`

- Topology: Flatten(3×32×32 = 3072) → PrunableLinear(3072, 512) → ReLU → PrunableLinear(512, 256) → ReLU → PrunableLinear(256, 10).
- Every weight in the network is prunable, making the "sparsity level" metric meaningful end-to-end.
- Helper methods:
  - `sparsity_loss()` → sum of `sigmoid(gate_scores)` across all `PrunableLinear` children (scalar tensor, differentiable).
  - `sparsity_level(threshold=1e-2)` → float; percentage of gates whose sigmoid value is below `threshold`.
  - `all_gate_values()` → detached 1-D tensor of every sigmoid-gate value in the model, for the histogram.

### Training Function

```
train_one_config(lam, epochs, train_loader, test_loader, device) -> dict
```

- Fresh model + fresh Adam optimizer (lr=1e-3) per λ — so results are independent.
- Loss: `CE(logits, y) + lam * model.sparsity_loss()`.
- At the end: compute test accuracy (no grad), sparsity level, and return `{lambda, test_accuracy, sparsity_pct, gate_values_flat}`.

### Main

- Builds the data loaders once (stratified 5k training subset using `sklearn.model_selection.train_test_split` is fine; alternatively a deterministic per-class index slice to keep dependencies light).
- Loops over λ ∈ {1e-5, 1e-4, 1e-3}; prints a console table and appends to a results list.
- Saves `results/metrics.json` (lambda, test_accuracy, sparsity_pct for each run).
- Picks the "best" model for the plot: the highest-λ run that still beats a 20% accuracy floor (or just the highest λ). Plots its gate-value histogram to `results/gate_distribution.png`.

## Report (`report.md`) Structure

1. **Why L1 on sigmoid gates encourages sparsity** — ~1 paragraph: L1 has a constant-magnitude gradient toward zero regardless of the current value, so any gate not earning its keep via classification loss gets pushed to 0. The sigmoid keeps gates in [0, 1] and saturates near 0 once gate_scores becomes very negative, making the zero state stable.
2. **Results Table** — filled from `metrics.json`.
3. **Gate distribution plot** — embedded `results/gate_distribution.png`.
4. **Trade-off discussion** — 2–3 sentences on how λ trades accuracy for sparsity.

## Non-Goals (YAGNI)

- No multi-GPU, distributed, or checkpoint/resume infrastructure.
- No unit tests — the spec asks for a runnable script + report, not a test suite. The report itself evidences correctness.
- No hyperparameter search beyond the required 3-point λ sweep.
- No CNN backbone; the spec says "feed-forward" and every parameter should be subject to the gating mechanism.

## Risks / Open Questions

- **MPS numerical quirks:** PyTorch MPS occasionally diverges from CPU on edge-case ops. The model uses only `Linear`, `ReLU`, `sigmoid`, and `CrossEntropyLoss`, which are all MPS-stable. If MPS fails at runtime, fall back to CPU (5k samples × 5 epochs is still tractable on CPU in ~5–10 min).
- **Small-subset noise:** With only 5k training samples, test accuracy will be modest (~40–50%) and may wobble run-to-run. Fixed seed mitigates this and the *relative* ordering across λ values is what matters for the report.

## Self-Review

- No TBDs / placeholders. Every component is specified.
- Internal consistency: architecture matches the feature list, loss formulation matches the spec, report contents match the case study requirements.
- Scope: single implementation plan, one script, one report. Bounded.
- Ambiguity: gate init value (2.0) and λ values (1e-5 / 1e-4 / 1e-3) are chosen explicitly rather than left open.
