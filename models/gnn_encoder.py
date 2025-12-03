import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=1, hidden=64, out_dim=64, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden, hidden))
        self.layers.append(GCNConv(hidden, out_dim))

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = torch.relu(h)
        return h
