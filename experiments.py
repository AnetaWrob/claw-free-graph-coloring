"""Experiment driver.

Usage examples
--------------
    python experiments.py --set C --max-n 9
    python experiments.py --set A --reps 20 --time-limit 20
    python experiments.py --set B --reps 20 --time-limit 20
    python experiments.py --set A --backend cpsat      # needs ortools

One CSV row is written per graph to data/results/set_<X>_<backend>.csv.
"""
import argparse
import csv
import os
import platform
import random
import sys
import time
from multiprocessing import Pool

import networkx as nx

from claw_recognition import is_claw_free
from dsatur_exact import dsatur_exact, is_proper_coloring
from edge_coloring import chromatic_index_model, is_overfull
from graph_generation import (connected_claw_free_graphs, generate_set_A,
                              generate_set_B)
from graph_parameters import max_clique, structural_parameters
from line_graph_recognition import NotALineGraph, reconstruct_root_graph
from vertex_coloring import chromatic_number_model

FIELDS = [
    "graph_id", "set", "source", "family", "n", "m", "density", "min_degree",
    "max_degree", "average_degree", "degeneracy", "omega", "chi",
    "chi_minus_omega", "chi_over_omega", "triangles", "neighborhood_density",
    "is_line_graph", "root_n", "root_m", "root_Delta", "root_ratio",
    "root_bipartite", "root_overfull", "edge_chromatic_number", "root_class",
    "omega_time", "root_reconstruction_time",
    "dsatur_status", "dsatur_chi", "dsatur_runtime", "dsatur_search_nodes",
    "dsatur_backtracks", "dsatur_initial_ub",
    "vedge_status", "vedge_chi", "vedge_runtime", "vedge_nodes", "vedge_calls",
    "vedge_constraints",
    "vclique_status", "vclique_chi", "vclique_runtime", "vclique_prep_time",
    "vclique_nodes", "vclique_constraints",
    "estar_status", "estar_chi", "estar_runtime", "estar_nodes",
    "estar_constraints", "speedup",
    "chi_reference", "consistent",
]


def shuffled_copy(G, seed):
    """Relabel vertices by a random permutation (hides the generation order)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    perm = nodes[:]
    rng.shuffle(perm)
    mapping = dict(zip(nodes, perm))
    G2 = nx.Graph()
    G2.add_nodes_from(sorted(perm))
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    rng.shuffle(edges)
    G2.add_edges_from(edges)
    return G2


def process(task):
    gid, set_name, source, G, chi_ref, tl, backend = task
    row = {f: "" for f in FIELDS}
    row.update(graph_id=gid, set=set_name, source=source)
    assert is_claw_free(G), f"{gid}: instance is not claw-free"
    row.update(structural_parameters(G))

    t0 = time.perf_counter()
    clique = max_clique(G)
    row["omega_time"] = time.perf_counter() - t0
    omega = len(clique)
    row["omega"] = omega

    # --- line graph recognition and root reconstruction (from G only)
    t0 = time.perf_counter()
    try:
        H, edge_of = reconstruct_root_graph(G)
    except NotALineGraph:
        H = None
    row["root_reconstruction_time"] = time.perf_counter() - t0
    row["is_line_graph"] = int(H is not None)
    row["family"] = "line" if H is not None else "non-line claw-free"

    values = {}
    # --- M1: exact DSATUR
    ds = dsatur_exact(G, lower_bound=omega, time_limit=tl)
    assert is_proper_coloring(G, ds["coloring"])
    row.update(dsatur_status=ds["status"], dsatur_chi=ds["chi"],
               dsatur_runtime=ds["runtime"], dsatur_search_nodes=ds["search_nodes"],
               dsatur_backtracks=ds["backtracks"],
               dsatur_initial_ub=ds["initial_upper_bound"])
    if ds["status"] == "optimal":
        values["dsatur"] = ds["chi"]

    # --- M2: vertex model with edge constraints
    ve = chromatic_number_model(G, omega, clique, "edge", tl, backend)
    row.update(vedge_status=ve["status"], vedge_chi=ve["value"] or "",
               vedge_runtime=ve["runtime"], vedge_nodes=ve["nodes"],
               vedge_calls=ve["calls"], vedge_constraints=ve["n_constraints"])
    if ve["status"] == "optimal":
        values["vedge"] = ve["value"]
        if ve["coloring"] is not None:
            assert is_proper_coloring(G, ve["coloring"])

    if H is not None:
        delta = max(d for _, d in H.degree())
        row.update(root_n=H.number_of_nodes(), root_m=H.number_of_edges(),
                   root_Delta=delta,
                   root_ratio=H.number_of_edges() / H.number_of_nodes(),
                   root_bipartite=int(nx.is_bipartite(H)),
                   root_overfull=int(is_overfull(H)))
        # --- M2c: vertex model with maximal-clique constraints
        vc = chromatic_number_model(G, omega, clique, "clique", tl, backend)
        row.update(vclique_status=vc["status"], vclique_chi=vc["value"] or "",
                   vclique_runtime=vc["runtime"], vclique_prep_time=vc["prep_time"],
                   vclique_nodes=vc["nodes"], vclique_constraints=vc["n_constraints"])
        if vc["status"] == "optimal":
            values["vclique"] = vc["value"]
        # --- M3: edge model on the reconstructed root graph
        es = chromatic_index_model(H, tl, backend)
        row.update(estar_status=es["status"], estar_chi=es["value"] or "",
                   estar_runtime=es["runtime"], estar_nodes=es["nodes"],
                   estar_constraints=es["n_constraints"],
                   root_class=es["root_class"])
        if es["status"] == "optimal":
            values["estar"] = es["value"]
            row["edge_chromatic_number"] = es["value"]
            if es["coloring"] is not None:   # transfer the colouring back to G
                col = {v: es["coloring"][frozenset(e)] for v, e in edge_of.items()}
                assert is_proper_coloring(G, col)
        if ve["status"] == "optimal" and es["status"] == "optimal":
            row["speedup"] = ve["runtime"] / max(es["runtime"], 1e-9)

    if chi_ref is not None:
        values["reference"] = chi_ref
        row["chi_reference"] = chi_ref
    distinct = set(values.values())
    row["consistent"] = int(len(distinct) <= 1)
    if len(distinct) == 1:
        chi = distinct.pop()
        row.update(chi=chi, chi_minus_omega=chi - omega, chi_over_omega=chi / omega)
    return row


def build_tasks(args):
    tl, be = args.time_limit, args.backend
    tasks = []
    if args.set == "A":
        for i, (gid, src, H) in enumerate(generate_set_A(args.reps)):
            if H.number_of_edges() < 2:
                continue
            G = shuffled_copy(nx.convert_node_labels_to_integers(nx.line_graph(H)), 77 + i)
            tasks.append((gid, "A", src, G, None, tl, be))
    elif args.set == "B":
        for gid, src, T in generate_set_B(args.reps):
            G = nx.complement(T)
            # complement of a triangle-free graph: chi(G) = n - nu(T)
            nu = len(nx.max_weight_matching(T, maxcardinality=True))
            tasks.append((gid, "B", src, G, G.number_of_nodes() - nu, tl, be))
    elif args.set == "C":
        levels = connected_claw_free_graphs(args.max_n)
        for n, graphs in levels.items():
            for j, G in enumerate(graphs):
                tasks.append((f"C-n{n}-{j}", "C", "enumeration", G, None, tl, be))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices="ABC", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--max-n", type=int, default=9)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--backend", choices=["highs", "cpsat"], default="highs")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"data/results/set_{args.set}_{args.backend}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tasks = build_tasks(args)
    import scipy
    print(f"# {len(tasks)} graphs, set {args.set}, backend {args.backend}, "
          f"time limit {args.time_limit}s | Python {platform.python_version()}, "
          f"networkx {nx.__version__}, scipy {scipy.__version__}", flush=True)
    t0 = time.time()
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap(process, tasks, chunksize=1), 1):
                w.writerow(row)
                fh.flush()
                if i % 50 == 0:
                    print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    print(f"# done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
