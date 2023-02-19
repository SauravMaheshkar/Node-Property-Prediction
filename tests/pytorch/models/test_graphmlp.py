"""Test PyTorch GraphMLP model"""

import pytest
import torch

from src.pytorch.models.graphmlp import GraphMLP, GraphMLPBlock

BATCH_SIZE = 1


@pytest.mark.pytorch
def test_graphmlp_block() -> None:
    """Test GraphMLPBlock"""

    # create graphmlp block
    graphmlp_block = GraphMLPBlock(num_features=1_433, hidden_dim=256, dropout_prob=0.1)

    # test forward pass
    temp_array = torch.rand(BATCH_SIZE, 1_433)
    output = graphmlp_block(temp_array)

    # test output shape
    assert output.shape == (BATCH_SIZE, 256)


@pytest.mark.pytorch
def test_graphmlp() -> None:
    """Test GraphMLP"""

    # create graphmlp model
    graphmlp = GraphMLP(num_features=1_433, num_classes=7)

    # test forward pass
    temp_array = torch.rand(BATCH_SIZE, 1_433)
    output, _ = graphmlp(temp_array)

    # test output shape
    assert output.shape == (BATCH_SIZE, 7)
