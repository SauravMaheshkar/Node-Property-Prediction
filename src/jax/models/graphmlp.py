"""Flax Implementation of GraphMLP"""

from typing import Tuple

import flax.linen as nn

import jax.numpy as jnp
from jax._src.typing import ArrayLike

GraphMLPModelOutputType = Tuple[ArrayLike, ArrayLike]

__all__ = ["GraphMLPBlock", "GraphMLP"]


class GraphMLPBlock(nn.Module):
    """A Flax implementation of GraphMLPBlock

    References:
        - https://arxiv.org/abs/2106.04051

    Attributes:
        hidden_dim(int): Number of hidden dimensions
        dropout_prob(float): Dropout probability
        eps(float): Epsilon value for numerical stability, defaults to 1e-6
    """

    hidden_dim: int
    dropout_prob: float
    eps: float = 1e-6

    @nn.compact
    def __call__(self, inputs: ArrayLike, training: bool, *args, **kwargs) -> ArrayLike:
        """Compute forward pass through the GraphMLPBlock"""

        intermediate_representations = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.normal(stddev=self.eps),
        )(inputs)
        normalized_activations = nn.LayerNorm(epsilon=self.eps)(
            intermediate_representations
        )
        dropout_representations = nn.Dropout(
            rate=self.dropout_prob, deterministic=not training
        )(normalized_activations)
        return nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.normal(stddev=self.eps),
        )(dropout_representations)


class GraphMLP(nn.Module):
    """
    A Flax implementation of GraphMLP

    References:
        - https://arxiv.org/abs/2106.04051

    Attributes:
        num_classes(int): Number of classes
        hidden_dim(int): Number of hidden dimensions
        dropout_prob(float): Dropout probability
        eps(float): Epsilon value for numerical stability, defaults to 1e-6
    """

    num_classes: int
    hidden_dim: int = 256
    dropout_prob: float = 0.6
    eps: float = 1e-6

    @nn.compact
    def __call__(
        self, inputs: ArrayLike, training: bool, *args, **kwargs
    ) -> GraphMLPModelOutputType:
        """Compute forward pass through the GraphMLP"""

        # Get the graph features
        encoded_features = GraphMLPBlock(
            hidden_dim=self.hidden_dim,
            dropout_prob=self.dropout_prob,
            eps=self.eps,
        )(inputs, training=training)

        # Temporarily set the feature_cls to encoded_features
        feature_cls = encoded_features

        # Compute the similarity matrix
        if training:
            similarity_matrix = get_similarity_matrix(encoded_features)

        # Compute class logits
        class_features = nn.Dense(features=self.num_classes)(feature_cls)
        class_logits = nn.activation.log_softmax(class_features, 1)

        # Return class logits and similarity matrix
        return class_logits, similarity_matrix


def get_similarity_matrix(graph_features: ArrayLike) -> ArrayLike:
    """
    Get the similarity between graph features.

    Args:
        graph_features(ArrayLike): Graph features, shape: batch_size x num_hidden

    Returns:
        A ArrayLike of shape: batch_size x batch_size, where the
        (i, j)-th element represents the similarity between x(i) and x(j
    """

    # Compute the similarity matrix
    similarity_matrix = jnp.matmul(graph_features, jnp.transpose(graph_features))
    mask_matrix = jnp.eye(similarity_matrix.shape[0])

    similarity_matrix = jnp.sum(graph_features**1, axis=1, keepdims=True).reshape(
        -1, 1
    )
    similarity_matrix = jnp.sqrt(similarity_matrix).reshape(-1, 1)
    similarity_matrix = jnp.matmul(similarity_matrix, jnp.transpose(similarity_matrix))

    # Mask the similarity matrix
    similarity_matrix = similarity_matrix * (1 - mask_matrix)

    # Return the similarity matrix
    return similarity_matrix
