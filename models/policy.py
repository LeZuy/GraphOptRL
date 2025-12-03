import torch
import torch.nn as nn
from utils.graph_utils import get_edge_list
from models.transformer_encoder import GraphTransformer
from models.positional_encoding import laplacian_pe, fiedler_vector

class Policy(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # input_dim = (#PE dims)
        input_dim = 8 + 1 + 1   # example: LaplacianPE(8) + Fiedler + degree

        self.encoder = GraphTransformer(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            num_heads=4
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, G):
        # Build positional features
        pe = laplacian_pe(G, k=8)
        fv = fiedler_vector(G)
        deg = torch.ones((G.number_of_nodes(), 1))

        x = torch.cat([pe, fv, deg], dim=-1)   # shape [n, in_dim]

        node_emb = self.encoder(x)

        # Compute edge scores
        edge_list = get_edge_list(G)
        scores = []
        for (u, v) in edge_list:
            h = torch.cat([node_emb[u], node_emb[v]])
            s = self.edge_mlp(h)
            scores.append(s)

        scores = torch.cat(scores).squeeze()
        probs = torch.softmax(scores, dim=0)
        return edge_list, probs

    def sample_action(self, G):
        edge_list, probs = self.forward(G)
        idx = torch.multinomial(probs, 2, replacement=False)
        e1, e2 = edge_list[idx[0]], edge_list[idx[1]]
        pattern = int(torch.randint(0, 2, (1,)))
        return (e1, e2, pattern), probs[idx].sum()
