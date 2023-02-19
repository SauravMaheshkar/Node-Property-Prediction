"""Test JAX GraphMLP model"""

import pytest

from jax import random
from src.jax.models.graphmlp import GraphMLP, GraphMLPBlock
from src.jax.utils import ones

BATCH_SIZE = 1


@pytest.mark.jax
def test_graphmlp_block() -> None:
    """Test GraphMLPBlock"""

    seed = random.PRNGKey(0)
    main_key, params_key, dropout_key = random.split(key=seed, num=3)

    # create temp array
    temp_array = ones(key=main_key, shape=(BATCH_SIZE, 1_433))

    # create graphmlp block
    graphmlp_block = GraphMLPBlock(hidden_dim=256, dropout_prob=0.1)
    variables = graphmlp_block.init(
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )

    # test forward pass
    output = graphmlp_block.apply(
        variables=variables,
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )

    # test output shape
    assert output.shape == (BATCH_SIZE, 256)


@pytest.mark.jax
def test_graphmlp() -> None:
    """Test GraphMLP"""

    seed = random.PRNGKey(0)
    main_key, params_key, dropout_key = random.split(key=seed, num=3)

    # create temp array
    temp_array = ones(key=main_key, shape=(BATCH_SIZE, 1_433))

    # create graphmlp model
    graphmlp = GraphMLP(num_classes=7)
    variables = graphmlp.init(
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=True,
    )

    # test forward pass
    output, _ = graphmlp.apply(
        variables=variables,
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=True,
    )

    # test output shape
    assert output.shape == (BATCH_SIZE, 7)
