import torch
import torch.nn as nn

class EdgeSelector(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1)
        )

    def forward(self, edge_list, node_embeddings):
        scores = []
        for (u, v) in edge_list:
            h = torch.cat([node_embeddings[u], node_embeddings[v]], dim=-1)
            s = self.edge_mlp(h)
            scores.append(s)
        scores = torch.cat(scores).squeeze()
        probs = torch.softmax(scores, dim=0)
        return probs
