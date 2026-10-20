"""Exact vertex colouring of G by a sequence of k-colourability models.

Two formulations are available:

* ``edge``   - one constraint per edge of G (no structural knowledge);
* ``clique`` - one constraint per maximal clique of G (requires enumeration
               of the maximal cliques; the enumeration time is reported).

Colours of a maximum clique are fixed to break the colour symmetry.
Back-ends: HiGHS (ILP) or OR-Tools CP-SAT, see ``coloring_backends``.
"""
import time

import networkx as nx

from coloring_backends import ascending_search


def chromatic_number_model(G: nx.Graph, lb: int, clique, model="edge",
                           time_limit=60.0, backend="highs", ub=None):
    """chi(G) given a lower bound lb (the clique number) and a clique whose
    colours are fixed; k = lb, lb+1, ... is tested until a colouring exists.  Returns the dict of ascending_search plus
    'prep_time' (time needed to build the constraint groups)."""
    t0 = time.perf_counter()
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    if model == "edge":
        groups = [(idx[u], idx[v]) for u, v in G.edges()]
    elif model == "clique":
        groups = [tuple(idx[v] for v in c) for c in nx.find_cliques(G) if len(c) >= 2]
    else:
        raise ValueError(model)
    fixed = {idx[v]: c for c, v in enumerate(clique)}
    prep = time.perf_counter() - t0
    out = ascending_search(len(nodes), groups, lb, fixed, time_limit, backend, ub)
    out["prep_time"] = prep
    out["n_constraints"] = len(groups)
    if out["coloring"] is not None:
        out["coloring"] = {nodes[i]: c for i, c in enumerate(out["coloring"])}
    return out
