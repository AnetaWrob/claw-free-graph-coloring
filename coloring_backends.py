"""Feasibility of k-colouring under clique ("all different") constraints.

A colouring instance is given by a number of items 0..n-1 and a list of
groups (tuples of items) that must receive pairwise different colours.  A
group of size two is an ordinary edge constraint; a larger group is a clique.

Two interchangeable exact back-ends are provided:

* ``highs``  - 0/1 integer linear programme solved by HiGHS through
               ``scipy.optimize.milp`` (assignment formulation);
* ``cpsat``  - constraint programme solved by OR-Tools CP-SAT with integer
               colour variables, ``!=`` and ``AllDifferent`` constraints
               (requires the optional ``ortools`` package).

Both return (status, colouring, runtime, nodes) with status in
{'feasible', 'infeasible', 'timeout'}.
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def _highs(n, groups, k, fixed, time_limit):
    t0 = time.perf_counter()
    nvar = n * k
    rows, cols = [], []
    r = 0
    for i in range(n):                       # sum_c x[i,c] = 1
        for c in range(k):
            rows.append(r)
            cols.append(i * k + c)
        r += 1
    n_assign = r
    for g in groups:                         # sum_{i in g} x[i,c] <= 1
        for c in range(k):
            for i in g:
                rows.append(r)
                cols.append(i * k + c)
            r += 1
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, nvar)).tocsr()
    lo = np.concatenate([np.ones(n_assign), np.zeros(r - n_assign)])
    hi = np.ones(r)
    xl, xu = np.zeros(nvar), np.ones(nvar)
    for i, c in fixed.items():
        xu[i * k:(i + 1) * k] = 0.0
        xl[i * k + c] = xu[i * k + c] = 1.0
    res = milp(c=np.zeros(nvar), constraints=[LinearConstraint(A, lo, hi)],
               integrality=np.ones(nvar), bounds=Bounds(xl, xu),
               options={"time_limit": float(time_limit), "disp": False})
    nodes = int(getattr(res, "mip_node_count", 0) or 0)
    dt = time.perf_counter() - t0
    if res.status == 0 and res.x is not None:
        x = np.asarray(res.x).reshape(n, k)
        return "feasible", [int(np.argmax(x[i])) for i in range(n)], dt, nodes
    if res.status == 2:
        return "infeasible", None, dt, nodes
    return "timeout", None, dt, nodes


def _cpsat(n, groups, k, fixed, time_limit):
    from ortools.sat.python import cp_model  # optional dependency
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    col = [model.NewIntVar(0, k - 1, f"c{i}") for i in range(n)]
    for g in groups:
        if len(g) == 2:
            model.Add(col[g[0]] != col[g[1]])
        elif len(g) > 2:
            model.AddAllDifferent([col[i] for i in g])
    for i, c in fixed.items():
        model.Add(col[i] == c)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    dt = time.perf_counter() - t0
    nodes = int(solver.NumBranches())
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "feasible", [int(solver.Value(v)) for v in col], dt, nodes
    if status == cp_model.INFEASIBLE:
        return "infeasible", None, dt, nodes
    return "timeout", None, dt, nodes


def k_coloring(n, groups, k, fixed=None, time_limit=60.0, backend="highs"):
    """Decide whether the instance admits a colouring with k colours."""
    fixed = fixed or {}
    if any(c >= k for c in fixed.values()):
        return "infeasible", None, 0.0, 0
    if any(len(g) > k for g in groups):
        return "infeasible", None, 0.0, 0
    solve = {"highs": _highs, "cpsat": _cpsat}[backend]
    status, col, dt, nodes = solve(n, [tuple(g) for g in groups], k, fixed,
                                   time_limit)
    if status == "feasible":
        for g in groups:                      # independent verification
            assert len({col[i] for i in g}) == len(g), "solver returned an improper colouring"
    return status, col, dt, nodes


def ascending_search(n, groups, lb, fixed, time_limit, backend, ub=None):
    """Smallest feasible k >= lb, found by testing k = lb, lb+1, ...

    If ``ub`` is given, k = ub is assumed feasible and is not tested (unless
    ub == lb is the very first value, which is always tested so that every
    run contains at least one model call).  The time limit applies to the
    whole sequence.  Returns a dict with value, status, runtime, nodes, calls
    and coloring.
    """
    total, nodes, calls = 0.0, 0, 0
    k = lb
    while True:
        if ub is not None and k >= ub and calls > 0:
            return dict(value=ub, status="optimal", runtime=total, nodes=nodes,
                        calls=calls, coloring=None)
        remaining = time_limit - total
        if remaining <= 0:
            return dict(value=None, status="timeout", runtime=total,
                        nodes=nodes, calls=calls, coloring=None)
        status, col, dt, nd = k_coloring(n, groups, k, fixed, remaining, backend)
        total += dt
        nodes += nd
        calls += 1
        if status == "feasible":
            return dict(value=k, status="optimal", runtime=total, nodes=nodes,
                        calls=calls, coloring=col)
        if status == "timeout":
            return dict(value=None, status="timeout", runtime=total,
                        nodes=nodes, calls=calls, coloring=None)
        k += 1
