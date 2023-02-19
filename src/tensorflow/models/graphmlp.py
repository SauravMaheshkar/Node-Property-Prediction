"""Tensorflow Implementation of GraphMLP"""

from typing import Tuple

import tensorflow as tf

__all__ = ["GraphMLPBlock", "GraphMLP"]

GraphMLPModelOutputType = Tuple[tf.Tensor, tf.Tensor]


class GraphMLPBlock(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of GraphMLPBlock

    References:
        - https://arxiv.org/abs/2106.04051

    Attributes:
        hidden_dim(int): Number of hidden dimensions
        dropout_prob(float): Dropout probability
        eps(float): Epsilon value for numerical stability, defaults to 1e-6
    """

    def __init__(
        self, hidden_dim: int, dropout_prob: float, eps: float = 1e-6, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.eps = eps

        # Linear layers
        self.linear_layer_1 = tf.keras.layers.Dense(
            units=hidden_dim,
            activation=tf.keras.activations.gelu,
            kernel_initializer="glorot_uniform",
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=self.eps),
        )
        self.linear_layer_2 = tf.keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer="glorot_uniform",
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=self.eps),
        )

        # Dropout and LayerNorm
        self.dropout = tf.keras.layers.Dropout(rate=dropout_prob)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=self.eps)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Compute forward pass through the GraphMLPBlock"""

        intermediate_representations = self.linear_layer_1(inputs)
        normalized_activations = self.layernorm(intermediate_representations)
        dropout_representations = self.dropout(normalized_activations)
        return self.linear_layer_2(dropout_representations)

    def get_config(self) -> dict:
        """Get the config for the GraphMLPBlock"""

        config = super().get_config()
        config.update({"eps": self.eps})
        return config


class GraphMLP(tf.keras.Model):
    """
    A Tensorflow implementation of GraphMLP

    References:
        - https://arxiv.org/abs/2106.04051

    Attributes:
        num_classes(int): Number of classes
        hidden_dim(int): Number of hidden dimensions, defaults to 256
        dropout_prob(float): Dropout probability, defaults to 0.6
        eps(float): Epsilon value for numerical stability, defaults to 1e-6
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        dropout_prob: float = 0.6,
        eps: float = 1e-6,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.eps = eps
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        # GraphMLPBlock
        self.mlp_encoder = GraphMLPBlock(
            hidden_dim=self.hidden_dim, dropout_prob=self.dropout_prob, eps=self.eps
        )

        # Classifier Layer
        self.classifier = tf.keras.layers.Dense(
            units=num_classes,
            activation=tf.keras.activations.softmax,
        )

    def call(
        self, inputs: tf.Tensor, training=None, mask=None
    ) -> GraphMLPModelOutputType:
        """Compute a forward pass through the GraphMLP Model"""

        # Get the graph features
        encoded_features = self.mlp_encoder(inputs)

        # Temporarily set the feature_cls to encoded_features
        feature_cls = encoded_features

        # Compute the similarity matrix
        if training:
            similarity_matrix = self.get_similarity_matrix(encoded_features)

        # Compute class logits
        class_logits = self.classifier(feature_cls)

        # Return class logits and similarity matrix
        return class_logits, similarity_matrix

    def get_similarity_matrix(self, graph_features: tf.Tensor) -> tf.Tensor:
        """
        Get the similarity between graph features.

        Args:
            graph_features(tf.Tensor): Graph features, shape: batch_size x num_hidden

        Returns:
            A tf.Tensor of shape: batch_size x batch_size, where the
            (i, j)-th element represents the similarity between x(i) and x(j
        """

        # Create similarity and mask matrices
        self.similarity_matrix = tf.matmul(
            graph_features, graph_features, transpose_b=True
        )
        mask_matrix = tf.eye(self.similarity_matrix.shape[0])

        self.similarity_matrix = tf.math.reduce_sum(graph_features**2, 1)
        self.similarity_matrix = tf.reshape(self.similarity_matrix, [-1, 1])
        self.similarity_matrix = tf.sqrt(self.similarity_matrix)
        self.similarity_matrix = tf.reshape(self.similarity_matrix, [1, -1])
        self.similarity_matrix = tf.matmul(
            self.similarity_matrix, self.similarity_matrix, transpose_b=True
        )

        # Mask the similarity matrix
        self.similarity_matrix = tf.math.multiply(
            (1 - mask_matrix), self.similarity_matrix
        )
        return self.similarity_matrix

    def get_config(self) -> dict:
        """Get the config for the GraphMLP model"""

        config = super().get_config()
        config.update(
            {
                "eps": self.eps,
                "hidden_dim": self.hidden_dim,
                "dropout_prob": self.dropout_prob,
            }
        )
        return config
