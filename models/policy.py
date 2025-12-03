import torch
import torch.nn as nn
from models.edge_selector import EdgeSelector
from utils.graph_utils import get_edge_list

class Policy(nn.Module):
    def __init__(self, gnn, hidden_dim=64):
        super().__init__()
        self.gnn = gnn
        self.edge_sel = EdgeSelector(hidden_dim)

    def forward(self, A):
        edge_index = A.nonzero().t().contiguous()
        x = torch.ones((A.shape[0], 1))  # trivial node features
        node_emb = self.gnn(x, edge_index)

        edge_list = get_edge_list(A)
        probs = self.edge_sel(edge_list, node_emb)
        return edge_list, probs

    def sample_action(self, A):
        edge_list, probs = self.forward(A)
        idx = torch.multinomial(probs, 2, replacement=False)
        e1 = edge_list[idx[0]]
        e2 = edge_list[idx[1]]
        pattern = int(torch.randint(0, 2, (1,)))
        return (e1, e2, pattern), probs[idx].sum()
