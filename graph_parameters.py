"""Structural parameters of a graph used in the experiments."""
import networkx as nx


def max_clique(G: nx.Graph):
    """A maximum clique (list of vertices), by enumeration of maximal cliques."""
    best = []
    for c in nx.find_cliques(G):
        if len(c) > len(best):
            best = c
    return best


def degeneracy(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return max(nx.core_number(G).values())


def neighbourhood_density(G: nx.Graph) -> float:
    """D_N(G): mean over all vertices of |E(G[N(v)])| / C(d(v), 2).

    Vertices of degree < 2 contribute 0 (the convention of
    ``networkx.average_clustering``; D_N coincides with the average
    clustering coefficient and is reported under the experiment's name).
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    total = 0.0
    for v in G:
        d = G.degree(v)
        if d >= 2:
            total += 2.0 * G.subgraph(G[v]).number_of_edges() / (d * (d - 1))
    return total / n


def structural_parameters(G: nx.Graph) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = [d for _, d in G.degree()]
    return {
        "n": n,
        "m": m,
        "density": 2.0 * m / (n * (n - 1)) if n > 1 else 0.0,
        "min_degree": min(degs) if degs else 0,
        "max_degree": max(degs) if degs else 0,
        "average_degree": 2.0 * m / n if n else 0.0,
        "degeneracy": degeneracy(G),
        "triangles": sum(nx.triangles(G).values()) // 3,
        "neighborhood_density": neighbourhood_density(G),
    }
