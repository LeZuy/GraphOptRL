import torch
import networkx as nx
from envs.spectral import compute_lambda2
from utils.graph_utils import rewire_edges

class GraphEnv:
    def __init__(self, n, k, T):
        self.n = n
        self.k = k
        self.T = T  # max steps per episode
        self.reset()

    def reset(self):
        # Generate random k-regular graph
        self.G = nx.random_regular_graph(self.k, self.n)
        self.t = 0
        return self.get_state()

    def get_state(self):
        A = torch.tensor(nx.to_numpy_array(self.G), dtype=torch.float32)
        return A

    def step(self, action):
        # action = (e1, e2, pattern)
        self.G = rewire_edges(self.G, action)

        self.t += 1
        done = (self.t >= self.T)

        if done:
            lambda2 = compute_lambda2(self.G)
            reward = -lambda2
        else:
            reward = 0.0

        return self.get_state(), reward, done
