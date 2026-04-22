"""
Self-Pruning Neural Network — Tredence AI Engineer case study.

Trains a feed-forward MLP on CIFAR-10 whose weights are multiplied by
learnable sigmoid gates. An L1 penalty on the gates is added to the
classification loss so the optimizer drives most gates toward zero,
yielding a sparse network that is pruned *during* training rather than
afterward.

Run: `python self_pruning_nn.py`
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """A Linear layer whose weights are element-wise gated by a learnable
    sigmoid parameter. Replaces `torch.nn.Linear` per the case-study spec.

    The forward pass computes `F.linear(x, weight * sigmoid(gate_scores), bias)`.
    Both `weight` and `gate_scores` are `nn.Parameter`s and receive gradients.
    """

    # Initial gate_scores value. sigmoid(2.0) ≈ 0.88, so the network starts
    # with nearly full capacity and *learns* to prune, rather than beginning
    # crippled with gates near 0.5 (which is what a zero init would give).
    INIT_GATE_SCORE = 2.0

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Match torch.nn.Linear's Kaiming-uniform init for fair comparison.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), self.INIT_GATE_SCORE)
        )
        self._reset_weight()

    def _reset_weight(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Bias init follows torch.nn.Linear: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def gates(self) -> torch.Tensor:
        """Return the current sigmoid-gate values (differentiable)."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pruned_weights = self.weight * self.gates()
        return F.linear(x, pruned_weights, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class PrunableMLP(nn.Module):
    """Flatten(3072) -> PrunableLinear -> ReLU -> PrunableLinear -> ReLU
    -> PrunableLinear(10). Every weight in the network is prunable, which
    makes the reported sparsity-level metric meaningful end-to-end.
    """

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        self.layers = nn.ModuleList(
            PrunableLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # no activation on final logits
                x = F.relu(x)
        return x

    def sparsity_loss(self) -> torch.Tensor:
        """Sum of sigmoid(gate_scores) across all PrunableLinear layers.
        Differentiable; this is what gets multiplied by λ in the total loss.
        """
        return torch.stack(
            [layer.gates().sum() for layer in self.prunable_layers()]
        ).sum()

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Percentage of gates whose sigmoid value is below `threshold`.
        Higher = more aggressively pruned."""
        with torch.no_grad():
            flat = self.all_gate_values()
            return 100.0 * (flat < threshold).float().mean().item()

    def all_gate_values(self) -> torch.Tensor:
        """Flat 1-D tensor of every sigmoid-gate value in the model."""
        with torch.no_grad():
            return torch.cat(
                [layer.gates().flatten() for layer in self.prunable_layers()]
            )


import os
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_indices(labels: list[int], per_class: int, seed: int = SEED) -> list[int]:
    """Deterministically pick `per_class` indices for each label."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_class[y].append(idx)
    selected: list[int] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    return selected


def make_dataloaders(
    train_subset_total: int = 5_000,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 loaders: a stratified `train_subset_total`-sample train set
    and the full 10k test set. Normalizes with the canonical CIFAR-10 stats."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    per_class = train_subset_total // 10
    idx = stratified_indices(full_train.targets, per_class=per_class)
    train_subset = Subset(full_train, idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test, batch_size=256, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def evaluate(model: PrunableMLP, test_loader: DataLoader, device: torch.device) -> float:
    """Return top-1 test accuracy in percent."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total


def _split_gate_and_weight_params(model: PrunableMLP):
    """Return (weight_params, gate_params) so the optimizer can use a higher
    learning rate on gate_scores. Adam normalizes gradient magnitude, so a
    single shared lr makes gate movement extremely slow — gates only reach
    the sparsity threshold within a reasonable training budget if they have
    their own (larger) learning rate.
    """
    weight_params, gate_params = [], []
    for name, p in model.named_parameters():
        (gate_params if name.endswith("gate_scores") else weight_params).append(p)
    return weight_params, gate_params


def train_one_config(
    lam: float,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr_weights: float = 1e-3,
    lr_gates: float = 5e-2,
    grad_clip: float = 5.0,
    log_prefix: str = "",
) -> dict:
    """Train one model for one λ value and return final metrics + gate values.

    Returns dict with keys: lambda, test_accuracy, sparsity_pct, gate_values_flat.
    """
    set_all_seeds(SEED)  # identical init across λ values → fair comparison
    model = PrunableMLP().to(device)
    weight_params, gate_params = _split_gate_and_weight_params(model)
    optimizer = torch.optim.Adam(
        [
            {"params": weight_params, "lr": lr_weights},
            {"params": gate_params, "lr": lr_gates},
        ]
    )
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"ce": 0.0, "sparsity": 0.0, "n": 0}
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            classification = ce_loss(logits, yb)
            sparsity = model.sparsity_loss()
            loss = classification + lam * sparsity

            optimizer.zero_grad()
            loss.backward()
            # Clip for numerical stability; MPS in particular can produce
            # occasional NaN spikes on this workload without it.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bs = yb.size(0)
            running["ce"] += classification.item() * bs
            running["sparsity"] += sparsity.item() * bs
            running["n"] += bs

        train_ce = running["ce"] / running["n"]
        sp_pct = model.sparsity_level()
        print(
            f"{log_prefix}epoch {epoch}/{epochs}: "
            f"train_ce={train_ce:.4f} sparsity_sum={running['sparsity']/running['n']:.1f} "
            f"sparsity_pct={sp_pct:.2f}%"
        )

    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.sparsity_level()
    gate_vals = model.all_gate_values().cpu().numpy()
    return {
        "lambda": lam,
        "test_accuracy": final_test_acc,
        "sparsity_pct": final_sparsity,
        "gate_values_flat": gate_vals,
    }


import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive; avoids blocking when running headless
import matplotlib.pyplot as plt


RESULTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "results"
LAMBDAS = (1e-5, 5e-5, 2e-4)  # low / medium / high
EPOCHS = 10
TRAIN_SUBSET = 5_000


def plot_gate_distribution(runs: list[dict], out_path: Path) -> None:
    """3-panel histogram of final sigmoid-gate values, one panel per λ.

    Y-axis is log-scaled so the bimodal structure (tall spike at 0, small
    surviving cluster near 1) is visible even when >90% of gates are pruned.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(13, 4), sharey=True)
    for ax, r in zip(axes, runs):
        ax.hist(
            r["gate_values_flat"],
            bins=60,
            range=(0.0, 1.0),
            color="#3366cc",
            edgecolor="black",
            linewidth=0.2,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Gate value (sigmoid of gate_scores)")
        ax.set_title(
            f"λ={r['lambda']:.0e}\n"
            f"sparsity={r['sparsity_pct']:.1f}%  "
            f"acc={r['test_accuracy']:.1f}%"
        )
        ax.axvline(
            1e-2, color="red", linestyle="--", linewidth=1,
            label="sparsity threshold (1e-2)",
        )
        ax.legend(loc="upper center", fontsize=8)
    axes[0].set_ylabel("Count (log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = make_dataloaders(
        train_subset_total=TRAIN_SUBSET, batch_size=128
    )
    print(
        f"Train samples: {len(train_loader.dataset)} | "
        f"Test samples: {len(test_loader.dataset)}"
    )

    runs: list[dict] = []
    for lam in LAMBDAS:
        print(f"\n{'='*60}\nTraining with λ = {lam}\n{'='*60}")
        result = train_one_config(
            lam=lam,
            epochs=EPOCHS,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            log_prefix=f"[λ={lam}] ",
        )
        runs.append(result)

    # Console summary table
    print("\n=== Results ===")
    print(f"{'Lambda':>10} | {'Test Acc (%)':>13} | {'Sparsity (%)':>13}")
    print("-" * 44)
    for r in runs:
        print(
            f"{r['lambda']:>10.0e} | {r['test_accuracy']:>13.2f} | {r['sparsity_pct']:>13.2f}"
        )

    # Persist metrics (strip numpy gate arrays — they go to the plot only)
    summary = [
        {"lambda": r["lambda"], "test_accuracy": r["test_accuracy"], "sparsity_pct": r["sparsity_pct"]}
        for r in runs
    ]
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "config": {
                    "train_subset_total": TRAIN_SUBSET,
                    "epochs": EPOCHS,
                    "lambdas": list(LAMBDAS),
                    "seed": SEED,
                    "device": str(device),
                },
                "runs": summary,
            },
            f,
            indent=2,
        )
    print(f"Wrote {metrics_path}")

    plot_path = RESULTS_DIR / "gate_distribution.png"
    plot_gate_distribution(runs, plot_path)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
