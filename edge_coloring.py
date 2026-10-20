"""Exact edge colouring of a root graph H (star formulation).

Every vertex v of H yields one "all different" constraint on the edges
incident with v; these stars are exactly the cliques of the Krausz partition
of L(H).  By Vizing's theorem chi'(H) is Delta(H) or Delta(H)+1, hence a
single k-colourability test with k = Delta(H) decides the problem.
"""
import time

import networkx as nx

from coloring_backends import k_coloring


def is_overfull(H: nx.Graph) -> bool:
    """|E(H)| > Delta(H) * floor(|V(H)|/2): a sufficient condition for class II."""
    if H.number_of_edges() == 0:
        return False
    delta = max(d for _, d in H.degree())
    return H.number_of_edges() > delta * (H.number_of_nodes() // 2)


def chromatic_index_model(H: nx.Graph, time_limit=60.0, backend="highs"):
    """chi'(H) for a simple graph H.

    Returns dict(value, status, runtime, nodes, calls, coloring, root_class,
    prep_time); coloring maps frozenset({u, v}) -> colour when the Delta-test
    is feasible.
    """
    t0 = time.perf_counter()
    edges = [frozenset(e) for e in H.edges()]
    eidx = {e: i for i, e in enumerate(edges)}
    delta = max((d for _, d in H.degree()), default=0)
    if delta == 0:
        return dict(value=0, status="optimal", runtime=0.0, nodes=0, calls=0,
                    coloring={}, root_class="I", prep_time=0.0)
    groups, vmax = [], None
    for v in H:
        star = tuple(eidx[frozenset((v, u))] for u in H[v])
        if len(star) >= 2:
            groups.append(star)
        if vmax is None or H.degree(v) > H.degree(vmax):
            vmax = v
    fixed = {eidx[frozenset((vmax, u))]: c for c, u in enumerate(H[vmax])}
    prep = time.perf_counter() - t0
    status, col, dt, nd = k_coloring(len(edges), groups, delta, fixed,
                                     time_limit, backend)
    out = dict(runtime=dt, nodes=nd, calls=1, prep_time=prep, coloring=None,
               n_constraints=len(groups))
    if status == "feasible":
        out.update(value=delta, status="optimal", root_class="I",
                   coloring={edges[i]: c for i, c in enumerate(col)})
    elif status == "infeasible":          # Vizing: chi' = Delta + 1
        out.update(value=delta + 1, status="optimal", root_class="II")
    else:
        out.update(value=None, status="timeout", root_class="")
    return out
