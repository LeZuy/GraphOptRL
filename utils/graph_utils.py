import random
import networkx as nx

def get_edge_list(G):
    return list(G.edges())

def sample_negative_edges(G, num_samples):
    all_nodes = list(G.nodes())
    negative_edges = set()
    while len(negative_edges) < num_samples:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u != v and not G.has_edge(u, v):
            negative_edges.add((u, v))
    return list(negative_edges)

def rewire_edges(G, action):
    (u, v), (x, y), pattern = action
    if G.has_edge(u, v) and G.has_edge(x, y):
        G.remove_edge(u, v)
        G.remove_edge(x, y)

        if pattern == 0:
            a, b = u, x
            c, d = v, y
        else:
            a, b = u, y
            c, d = v, x

        if not G.has_edge(a, b):
            G.add_edge(a, b)
        if not G.has_edge(c, d):
            G.add_edge(c, d)
    return G