import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

def compute_lambda2(G):
    A = nx.to_numpy_array(G)
    vals, _ = eigsh(A, k=2, which='LA')
    vals = np.sort(vals)
    lambda2 = vals[-2]
    return float(lambda2)
