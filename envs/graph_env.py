import networkx as nx
from envs.spectral import compute_lambda2
from utils.graph_utils import rewire_edges

class GraphEnv:
    def __init__(self, n, k, T):
        self.n = n
        self.k = k
        self.T = T

    def reset(self):
        self.G = nx.random_regular_graph(self.k, self.n)
        self.t = 0
        return self.G

    def step(self, action):
        # Rewire in-place
        self.G = rewire_edges(self.G, action)

        self.t += 1
        done = (self.t >= self.T)

        if done:
            reward = -compute_lambda2(self.G)
        else:
            reward = 0

        return self.G, reward, done
