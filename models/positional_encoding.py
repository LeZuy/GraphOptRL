import torch
import numpy as np
import networkx as nx

def laplacian_pe(G, k=8):
    L = nx.normalized_laplacian_matrix(G).astype(float).todense()
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)

    # take smallest k eigenvectors (excluding the trivial eigenvalue=0)
    vecs = vecs[:, idx[1:k+1]]
    return torch.tensor(vecs, dtype=torch.float32)

def fiedler_vector(G):
    L = nx.laplacian_matrix(G).astype(float).todense()
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)
    # second smallest eigenvector
    v2 = vecs[:, idx[1]]
    return torch.tensor(v2, dtype=torch.float32).unsqueeze(-1)
