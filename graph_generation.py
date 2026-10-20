"""Generators of the test sets.

Set A: line graphs L(H) of random root graphs H (Erdos-Renyi, regular,
       bipartite, trees).
Set B: claw-free graphs that are not line graphs, obtained as complements of
       triangle-free graphs (bipartite graphs and the triangle-free process).
Set C: all connected claw-free graphs of small order, up to isomorphism.
"""
import random
from itertools import combinations

import networkx as nx

from claw_recognition import is_claw_free


# ------------------------------------------------------------------ set A
def _drop_isolated(H):
    H = H.copy()
    H.remove_nodes_from([v for v in list(H) if H.degree(v) == 0])
    return nx.convert_node_labels_to_integers(H)


def root_erdos_renyi(n, avg_deg, seed):
    return _drop_isolated(nx.gnp_random_graph(n, min(1.0, avg_deg / (n - 1)), seed=seed))


def root_regular(n, d, seed):
    return nx.random_regular_graph(d, n, seed=seed)


def root_bipartite(n, avg_deg, seed):
    a, b = n // 2, n - n // 2
    p = min(1.0, avg_deg * n / (2.0 * a * b))
    return _drop_isolated(nx.bipartite.random_graph(a, b, p, seed=seed))


def root_tree(n, seed):
    return nx.random_labeled_tree(n, seed=seed)


def line_graph_instance(H):
    """L(H) with integer vertices in random-free canonical order of E(H)."""
    return nx.convert_node_labels_to_integers(nx.line_graph(H))


def root_subdivided_regular(n0, d, seed):
    """d-regular graph of odd order n0 with one edge subdivided.

    The graph has even order and is not overfull, but it is of class II:
    in a d-edge-colouring every colour class would have to cover all n0
    vertices of degree d and hence (by parity) also the new vertex of degree 2,
    which is impossible for d > 2.
    """
    H = nx.random_regular_graph(d, n0, seed=seed)
    u, v = sorted(H.edges())[0]
    H.remove_edge(u, v)
    H.add_edge(u, n0)
    H.add_edge(n0, v)
    return H


def generate_set_A(reps=20, base_seed=1000):
    """Yield (graph_id, source, H) for the root graphs of set A."""
    seed = base_seed
    for n in (10, 20, 30, 40, 50):
        for d in (2, 4, 8, 16):
            if d > n - 2:
                continue
            for r in range(reps):
                seed += 1
                H = root_erdos_renyi(n, d, seed)
                yield f"A-er-n{n}-d{d}-{r}", "er", H
        for d in (2, 4, 8, 16):
            if d >= n // 2:
                continue
            for r in range(reps):
                seed += 1
                H = root_bipartite(n, d, seed)
                yield f"A-bip-n{n}-d{d}-{r}", "bipartite", H
        for r in range(reps):
            seed += 1
            yield f"A-tree-n{n}-{r}", "tree", root_tree(n, seed)
    for n in (10, 15, 20, 25, 30, 40, 50):
        for d in (3, 4, 6, 8):
            if (n * d) % 2:
                continue
            for r in range(reps):
                seed += 1
                yield f"A-reg-n{n}-d{d}-{r}", "regular", root_regular(n, d, seed)
    for n0 in (15, 25, 35):
        for d in (4, 6, 8):
            for r in range(reps):
                seed += 1
                yield (f"A-sub-n{n0}-d{d}-{r}", "subdivided",
                       root_subdivided_regular(n0, d, seed))


# ------------------------------------------------------------------ set B
def triangle_free_process(n, fraction, seed):
    """Random triangle-free graph: pairs are scanned in random order and added
    when they close no triangle.  ``fraction`` in (0, 1] keeps that share of
    the edges of the resulting maximal triangle-free graph (a prefix of the
    process)."""
    rng = random.Random(seed)
    pairs = list(combinations(range(n), 2))
    rng.shuffle(pairs)
    T = nx.empty_graph(n)
    order = []
    for u, v in pairs:
        if not (set(T[u]) & set(T[v])):
            T.add_edge(u, v)
            order.append((u, v))
    keep = order[:max(1, int(round(fraction * len(order))))]
    T2 = nx.empty_graph(n)
    T2.add_edges_from(keep)
    return T2


def generate_set_B(reps=20, sizes=(20, 30, 40, 50), base_seed=5000):
    """Yield (graph_id, source, T) where the instance is the complement of the
    triangle-free graph T."""
    seed = base_seed
    for n in sizes:
        for p in (0.3, 0.6):
            for r in range(reps):
                seed += 1
                T = nx.bipartite.random_graph(n // 2, n - n // 2, p, seed=seed)
                yield f"B-cobip-n{n}-p{p}-{r}", "co-bipartite", nx.Graph(T)
        for f in (0.5, 1.0):
            for r in range(reps):
                seed += 1
                yield f"B-cotf-n{n}-f{f}-{r}", "co-triangle-free", triangle_free_process(n, f, seed)


# ------------------------------------------------------------------ set C
def _invariant(G):
    deg = dict(G.degree())
    return (G.number_of_edges(),
            tuple(sorted((deg[v], tuple(sorted(deg[u] for u in G[v]))) for v in G)),
            sum(nx.triangles(G).values()))


def connected_claw_free_graphs(max_n=8):
    """All connected claw-free graphs on 1..max_n vertices up to isomorphism.

    Orderly-free generation by vertex extension: every connected graph on
    n+1 vertices has a vertex whose removal leaves a connected graph, and
    claw-freeness is hereditary, so each graph of order n+1 arises from a graph
    of order n by adding one vertex with a non-empty neighbourhood.
    Returns {n: [graphs]}.
    """
    levels = {1: [nx.empty_graph(1)]}
    for n in range(1, max_n):
        buckets = {}
        for G in levels[n]:
            for mask in range(1, 2 ** n):
                nb = [i for i in range(n) if mask >> i & 1]
                G2 = G.copy()
                G2.add_node(n)
                G2.add_edges_from((n, i) for i in nb)
                if not is_claw_free(G2):
                    continue
                key = _invariant(G2)
                bucket = buckets.setdefault(key, [])
                if not any(nx.is_isomorphic(G2, F) for F in bucket):
                    bucket.append(G2)
        levels[n + 1] = [G for b in buckets.values() for G in b]
    return levels
