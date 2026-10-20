"""DSATUR heuristic and exact DSATUR-based branch and bound (Brelaz, 1979)."""
import sys
import time

import networkx as nx


def _prepare(G: nx.Graph):
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    adj = [[idx[u] for u in G[v]] for v in nodes]
    return nodes, adj


def dsatur_heuristic(G: nx.Graph):
    """Greedy DSATUR.  Returns (number of colours, colouring dict)."""
    nodes, adj = _prepare(G)
    n = len(nodes)
    color = [-1] * n
    nb_colors = [set() for _ in range(n)]
    deg = [len(a) for a in adj]
    used = 0
    for _ in range(n):
        v = max((u for u in range(n) if color[u] < 0),
                key=lambda u: (len(nb_colors[u]), deg[u]))
        c = 0
        while c in nb_colors[v]:
            c += 1
        color[v] = c
        used = max(used, c + 1)
        for u in adj[v]:
            nb_colors[u].add(c)
    return used, {nodes[i]: color[i] for i in range(n)}


def dsatur_exact(G: nx.Graph, lower_bound: int = 1, time_limit: float = 60.0):
    """Exact chromatic number by DSATUR branch and bound.

    Returns a dict with keys chi (best number of colours found), status
    ('optimal' or 'timeout'), runtime, search_nodes, backtracks,
    initial_lower_bound, initial_upper_bound, coloring.
    """
    t0 = time.perf_counter()
    nodes, adj = _prepare(G)
    n = len(nodes)
    ub, heur = dsatur_heuristic(G)
    res = {
        "initial_lower_bound": lower_bound,
        "initial_upper_bound": ub,
        "search_nodes": 0,
        "backtracks": 0,
        "status": "optimal",
    }
    best = {"k": ub, "col": [heur[v] for v in nodes]}
    if n == 0 or ub <= lower_bound:
        res.update(chi=ub, runtime=time.perf_counter() - t0,
                   coloring=dict(zip(nodes, best["col"])))
        return res

    color = [-1] * n
    cnt = [dict() for _ in range(n)]  # cnt[v][c] = #neighbours of v coloured c
    deg = [len(a) for a in adj]
    deadline = t0 + time_limit
    stats = {"nodes": 0, "back": 0, "stop": False, "timeout": False}
    sys.setrecursionlimit(max(10000, 4 * n + 1000))

    def branch(n_colored: int, used: int):
        stats["nodes"] += 1
        if (stats["nodes"] & 255) == 0 and time.perf_counter() > deadline:
            stats["stop"] = stats["timeout"] = True
            return
        if n_colored == n:
            best["k"] = used
            best["col"] = color[:]
            if used <= lower_bound:
                stats["stop"] = True
            return
        # vertex of maximum saturation, ties broken by degree
        v, key = -1, (-1, -1)
        for u in range(n):
            if color[u] < 0:
                ku = (len(cnt[u]), deg[u])
                if ku > key:
                    v, key = u, ku
        c = 0
        # colours 0..used-1 and at most one new colour; must stay below best-1
        while c <= used and c < best["k"] - 1:
            if c not in cnt[v]:
                color[v] = c
                for u in adj[v]:
                    cu = cnt[u]
                    cu[c] = cu.get(c, 0) + 1
                before = best["k"]
                branch(n_colored + 1, max(used, c + 1))
                color[v] = -1
                for u in adj[v]:
                    cu = cnt[u]
                    if cu[c] == 1:
                        del cu[c]
                    else:
                        cu[c] -= 1
                if stats["stop"]:
                    return
                if best["k"] == before:
                    stats["back"] += 1
            c += 1

    branch(0, 0)
    res.update(
        chi=best["k"],
        status="timeout" if stats["timeout"] else "optimal",
        runtime=time.perf_counter() - t0,
        search_nodes=stats["nodes"],
        backtracks=stats["back"],
        coloring=dict(zip(nodes, best["col"])),
    )
    return res


def is_proper_coloring(G: nx.Graph, coloring: dict) -> bool:
    return all(coloring[u] != coloring[v] for u, v in G.edges())
