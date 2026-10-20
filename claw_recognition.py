"""Recognition of claw-free graphs.

A claw is an induced K_{1,3}.  A graph G is claw-free iff for every vertex v
the neighbourhood N(v) contains no independent set of size 3, i.e.
alpha(G[N(v)]) <= 2.
"""
from itertools import combinations

import networkx as nx


def find_claw(G: nx.Graph):
    """Return a claw as (center, a, b, c) or None if G is claw-free."""
    adj = {v: set(G[v]) for v in G}
    for v in G:
        nb = sorted(adj[v], key=repr)
        if len(nb) < 3:
            continue
        for a, b in combinations(nb, 2):
            if b in adj[a]:
                continue
            # a, b non-adjacent: look for c non-adjacent to both
            for c in nb:
                if c != a and c != b and c not in adj[a] and c not in adj[b]:
                    return (v, a, b, c)
    return None


def is_claw_free(G: nx.Graph) -> bool:
    """True iff G contains no induced K_{1,3}."""
    return find_claw(G) is None
