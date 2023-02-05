"""PyTorch Implementation of GraphMLP"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout, LayerNorm

__all__ = ["GraphMLPBlock", "GraphMLP"]

GraphMLPModelOutputType = Tuple[torch.Tensor, torch.Tensor]


class GraphMLPBlock(torch.nn.Module):
    """
    A PyTorch implementation of GraphMLPBlock

    References:
        - https://arxiv.org/abs/2106.04051

    Attributes:
        num_features(int): Number of features in the input
        hidden_dim(int): Number of hidden dimensions
        dropout_prob(float): Dropout probability
        eps(float): Epsilon value for numerical stability, defaults to 1e-6
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        dropout_prob: float,
        eps: float = 1e-6,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.eps = eps

        # Linear layers
        self.linear_layer_1 = torch.nn.Linear(
            in_features=num_features, out_features=hidden_dim
        )
        self.linear_layer_2 = torch.nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim
        )

        # Activation, Dropout and LayerNorm
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(p=dropout_prob)
        self.layernorm = LayerNorm(normalized_shape=hidden_dim, eps=self.eps)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights of various layers"""

        # Initialize weights of linear layers
        nn.init.xavier_uniform_(self.linear_layer_1.weight)
        nn.init.xavier_uniform_(self.linear_layer_2.weight)

        # Initialize biases of linear layers
        nn.init.normal_(self.linear_layer_1.bias, std=self.eps)
        nn.init.normal_(self.linear_layer_2.bias, std=self.eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through the GraphMLPBlock"""

        intermediate_representations = self.linear_layer_1(inputs)
        activations = self.act_fn(intermediate_representations)
        normalized_activations = self.layernorm(activations)
        dropout_representations = self.dropout(normalized_activations)
        return self.linear_layer_2(dropout_representations)


class GraphMLP(torch.nn.Module):
    """
    A PyTorch implementation of GraphMLP

    References:
        - https://arxiv.org/abs/2106.04051

    Attributes:
        num_features(int): Number of features in the input
        num_classes(int): Number of classes in the dataset
        eps(float): Epsilon value for numerical stability, defaults to 1e-6
        hidden_dim(int): Number of hidden dimensions, defaults to 256
        dropout_prob(float): Dropout probability, defaults to 0.6
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        eps: float = 1e-6,
        hidden_dim: int = 256,
        dropout_prob: float = 0.6,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.eps = eps
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        # GraphMLPBlock
        self.mlp_encoder = GraphMLPBlock(
            num_features=num_features,
            hidden_dim=self.hidden_dim,
            dropout_prob=self.dropout_prob,
            eps=self.eps,
        )

        # Classifier Layer
        self.classifier = torch.nn.Linear(
            in_features=self.hidden_dim, out_features=num_classes
        )

    def forward(self, batch) -> GraphMLPModelOutputType:
        """Compute a forward pass through the GraphMLP Model"""

        # Get the graph features
        encoded_features = self.mlp_encoder(batch)

        # Temporarily set the feature_cls to encoded_features
        feature_cls = encoded_features

        # Compute similarity matrix
        if self.training:
            similarity_matrix = self.get_similarity_matrix(encoded_features)

        # Compute class logits
        class_features = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_features, dim=1)

        # Return class logits and similarity matrix
        return class_logits, similarity_matrix

    def get_similarity_matrix(self, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Get the similarity between graph features.

        Args:
            graph_features(torch.Tensor): Graph features, shape: batch_size x num_hidden

        Returns:
            A torch.Tensor of shape: batch_size x batch_size, where the
            (i, j)-th element represents the similarity between x(i) and x(j
        """

        # Create similarity and mask matrices
        self.similarity_matrix = graph_features @ graph_features.T
        mask_matrix = torch.eye(self.similarity_matrix.shape[0])

        x_sum = torch.sum(graph_features**2, 1).reshape(-1, 1)
        x_sum = torch.sqrt(x_sum).reshape(-1, 1)
        x_sum = x_sum @ x_sum.T
        self.similarity_matrix = self.similarity_matrix * (x_sum ** (-1))

        # Mask the similarity matrix
        self.similarity_matrix = (1 - mask_matrix) * self.similarity_matrix
        return self.similarity_matrix
