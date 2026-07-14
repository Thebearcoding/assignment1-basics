from __future__ import annotations

import torch
from torch import Tensor, nn

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((d_out, d_in), device=device, dtype=dtype))
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]