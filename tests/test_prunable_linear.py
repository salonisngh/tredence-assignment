"""Smoke tests for PrunableLinear (evaluation criterion #1)."""
import torch

from self_pruning_nn import PrunableLinear


def test_output_shape():
    layer = PrunableLinear(in_features=4, out_features=3)
    x = torch.randn(8, 4)
    y = layer(x)
    assert y.shape == (8, 3)


def test_gate_scores_is_parameter_with_correct_shape():
    layer = PrunableLinear(in_features=4, out_features=3)
    assert isinstance(layer.gate_scores, torch.nn.Parameter)
    assert layer.gate_scores.shape == layer.weight.shape


def test_gradients_flow_to_weight_and_gate_scores():
    """The 'Challenge' in the spec: gradients must reach both tensors."""
    torch.manual_seed(0)
    layer = PrunableLinear(in_features=4, out_features=3)
    x = torch.randn(2, 4)
    loss = layer(x).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.gate_scores.grad is not None
    assert torch.any(layer.weight.grad != 0)
    assert torch.any(layer.gate_scores.grad != 0)


def test_closed_gate_zeros_weight_contribution():
    """If all gate_scores are very negative, sigmoid≈0, so the layer
    output should equal only the bias (approximately)."""
    layer = PrunableLinear(in_features=4, out_features=3)
    with torch.no_grad():
        layer.gate_scores.fill_(-20.0)  # sigmoid(-20) ≈ 2e-9
        layer.bias.fill_(0.5)
    x = torch.randn(2, 4)
    y = layer(x)
    assert torch.allclose(y, torch.full_like(y, 0.5), atol=1e-5)


from self_pruning_nn import PrunableMLP


def test_mlp_output_shape_matches_cifar10():
    model = PrunableMLP()
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_mlp_sparsity_loss_is_sum_of_all_gates():
    model = PrunableMLP()
    expected = sum(layer.gates().sum() for layer in model.prunable_layers())
    assert torch.allclose(model.sparsity_loss(), expected)


def test_mlp_sparsity_level_threshold():
    model = PrunableMLP()
    with torch.no_grad():
        # Force every gate_score very negative -> sigmoid ≈ 0 -> 100% sparse.
        for layer in model.prunable_layers():
            layer.gate_scores.fill_(-20.0)
    assert model.sparsity_level(threshold=1e-2) == 100.0
