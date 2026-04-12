from __future__ import annotations

import networkx as nx


def make_erdos_renyi(n: int = 149, avg_degree: int = 6, seed: int = 42) -> nx.Graph:
    """
    Create an Erdos-Renyi random graph with average degree about avg_degree.
    """
    p = avg_degree / (n - 1)
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)


def make_watts_strogatz(
    n: int = 149,
    k: int = 6,
    beta: float = 0.1,
    seed: int = 42,
) -> nx.Graph:
    """
    Create a Watts-Strogatz small-world graph.
    k should be even.
    """
    return nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=seed)


def make_barabasi_albert(n: int = 149, m: int = 3, seed: int = 42) -> nx.Graph:
    """
    Create a Barabasi-Albert scale-free graph.
    Average degree is roughly 2m, so m=3 gives avg degree about 6.
    """
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)