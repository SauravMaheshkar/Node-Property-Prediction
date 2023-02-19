"""Test TensorFlow GraphMLP model"""

import pytest

import tensorflow as tf
from src.tensorflow.models.graphmlp import GraphMLP, GraphMLPBlock

BATCH_SIZE = 1


@pytest.mark.tensorflow
def test_graphmlp_block() -> None:
    """Test GraphMLPBlock"""

    # create graphmlp block
    graphmlp_block = GraphMLPBlock(hidden_dim=256, dropout_prob=0.1)

    # test forward pass
    temp_array = tf.random.normal((BATCH_SIZE, 1_433))
    output = graphmlp_block(temp_array)

    # test output shape
    assert output.shape == (BATCH_SIZE, 256)


@pytest.mark.tensorflow
def test_graphmlp() -> None:
    """Test GraphMLP"""

    # create graphmlp model
    graphmlp = GraphMLP(num_classes=7)

    # test forward pass
    temp_array = tf.random.normal((BATCH_SIZE, 1_433))
    output, _ = graphmlp(temp_array, training=True)

    # test output shape
    assert output.shape == (BATCH_SIZE, 7)
