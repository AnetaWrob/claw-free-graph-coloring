"""Unit tests (run with:  python -m pytest tests  from the project root)."""
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from claw_recognition import find_claw, is_claw_free
from dsatur_exact import dsatur_exact, is_proper_coloring
from edge_coloring import chromatic_index_model, is_overfull
from graph_generation import (connected_claw_free_graphs,
                              root_subdivided_regular, triangle_free_process)
from graph_parameters import max_clique
from line_graph_recognition import recognize_line_graph, reconstruct_root_graph, verify_root
from vertex_coloring import chromatic_number_model


def chi_models(G):
    c = max_clique(G)
    return {m: chromatic_number_model(G, len(c), c, m, 30)["value"] for m in ("edge", "clique")}


def test_claw():
    assert find_claw(nx.star_graph(3)) is not None
    assert not is_claw_free(nx.complete_bipartite_graph(2, 3))
    assert is_claw_free(nx.cycle_graph(7))
    assert is_claw_free(nx.complement(nx.cycle_graph(7)))
    assert is_claw_free(nx.line_graph(nx.petersen_graph()))


def test_line_graph_recognition():
    for H in (nx.petersen_graph(), nx.complete_graph(5), nx.path_graph(6),
              nx.complete_bipartite_graph(3, 4)):
        G = nx.convert_node_labels_to_integers(nx.line_graph(H))
        R, edge_of = reconstruct_root_graph(G)
        assert verify_root(G, edge_of)
        assert R.number_of_edges() == G.number_of_nodes()
    # Beineke: K_5 minus an edge and the 5-wheel are claw-free but not line graphs
    K5e = nx.complete_graph(5)
    K5e.remove_edge(0, 1)
    assert is_claw_free(K5e) and not recognize_line_graph(K5e)
    assert not recognize_line_graph(nx.wheel_graph(6))
    assert not recognize_line_graph(nx.star_graph(3))


def test_counts_small_claw_free():
    levels = connected_claw_free_graphs(7)
    assert [len(levels[n]) for n in range(1, 8)] == [1, 1, 2, 5, 14, 50, 191]
    assert [sum(recognize_line_graph(G) for G in levels[n]) for n in range(1, 8)] \
        == [1, 1, 2, 5, 12, 30, 79]


def test_chromatic_numbers():
    cases = [(nx.cycle_graph(5), 3), (nx.complete_graph(6), 6),
             (nx.complement(nx.cycle_graph(7)), 4), (nx.petersen_graph(), 3),
             (nx.line_graph(nx.petersen_graph()), 4)]
    for G, chi in cases:
        G = nx.convert_node_labels_to_integers(G)
        ds = dsatur_exact(G, lower_bound=len(max_clique(G)))
        assert ds["chi"] == chi and is_proper_coloring(G, ds["coloring"])
        assert set(chi_models(G).values()) == {chi}


def test_chromatic_index():
    assert chromatic_index_model(nx.petersen_graph())["value"] == 4      # class II
    assert chromatic_index_model(nx.complete_graph(5))["value"] == 5     # overfull
    assert is_overfull(nx.complete_graph(5))
    assert chromatic_index_model(nx.complete_graph(6))["value"] == 5
    assert chromatic_index_model(nx.complete_bipartite_graph(3, 5))["value"] == 5
    H = root_subdivided_regular(15, 4, 1)
    assert not is_overfull(H) and chromatic_index_model(H)["value"] == 5


def test_complement_of_triangle_free():
    T = triangle_free_process(14, 1.0, 3)
    assert sum(nx.triangles(T).values()) == 0
    G = nx.complement(T)
    assert is_claw_free(G)
    nu = len(nx.max_weight_matching(T, maxcardinality=True))
    assert dsatur_exact(G, len(max_clique(G)))["chi"] == 14 - nu
