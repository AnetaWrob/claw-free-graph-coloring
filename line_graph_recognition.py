"""Recognition of line graphs and reconstruction of a root graph.

The reconstruction relies on the Roussopoulos-type procedure shipped with
NetworkX (``networkx.inverse_line_graph``), applied to each connected component
separately, and is followed by an explicit verification: the returned
correspondence  vertex of G -> edge of H  must be a bijection under which two
vertices of G are adjacent iff the corresponding edges of H share an endpoint.
"""
import networkx as nx


class NotALineGraph(Exception):
    pass


def _root_of_component(C: nx.Graph):
    """Root graph of a connected graph C.

    Returns (H, edge_of) where edge_of maps each vertex of C to a pair of
    vertices of H.  Raises NotALineGraph.
    """
    nodes = list(C.nodes())
    if len(nodes) == 1:  # K_1 = L(K_2)
        v = nodes[0]
        H = nx.Graph()
        H.add_edge(0, 1)
        return H, {v: (0, 1)}
    # relabel to integers: inverse_line_graph is sensitive to node types
    idx = {v: i for i, v in enumerate(nodes)}
    Ci = nx.relabel_nodes(C, idx, copy=True)
    try:
        R = nx.inverse_line_graph(Ci)
    except nx.NetworkXError as exc:
        raise NotALineGraph(str(exc)) from None
    except Exception as exc:  # defensive: malformed input for the library
        raise NotALineGraph(f"reconstruction failed: {exc!r}") from None
    cells = list(R.nodes())
    cell_id = {c: i for i, c in enumerate(cells)}
    containing = {i: [] for i in range(len(nodes))}
    for c in cells:
        for member in c:
            if member in containing:
                containing[member].append(cell_id[c])
    H = nx.Graph()
    H.add_nodes_from(range(len(cells)))
    edge_of = {}
    for i, v in enumerate(nodes):
        if len(containing[i]) != 2:
            raise NotALineGraph("vertex does not lie in exactly two cells")
        a, b = containing[i]
        if H.has_edge(a, b):
            raise NotALineGraph("parallel edges in the root graph")
        H.add_edge(a, b)
        edge_of[v] = (a, b)
    return H, edge_of


def verify_root(G: nx.Graph, edge_of: dict) -> bool:
    """Check that v -> edge_of[v] realises G as a line graph."""
    if len({frozenset(e) for e in edge_of.values()}) != G.number_of_nodes():
        return False
    incident = {}
    for v, (a, b) in edge_of.items():
        incident.setdefault(a, []).append(v)
        incident.setdefault(b, []).append(v)
    pairs = set()
    for vs in incident.values():
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                pairs.add(frozenset((vs[i], vs[j])))
    return pairs == {frozenset(e) for e in G.edges()}


def reconstruct_root_graph(G: nx.Graph):
    """Return (H, edge_of) with L(H) isomorphic to G, or raise NotALineGraph.

    H has integer vertices and no isolated vertices; edge_of maps every vertex
    of G to the corresponding edge (pair of vertices) of H.
    """
    H = nx.Graph()
    edge_of = {}
    offset = 0
    for comp in nx.connected_components(G):
        Hc, ec = _root_of_component(G.subgraph(comp))
        H.add_edges_from((a + offset, b + offset) for a, b in Hc.edges())
        for v, (a, b) in ec.items():
            edge_of[v] = (a + offset, b + offset)
        offset += Hc.number_of_nodes()
    if not verify_root(G, edge_of):
        raise NotALineGraph("verification of the reconstructed root failed")
    return H, edge_of


def recognize_line_graph(G: nx.Graph) -> bool:
    try:
        reconstruct_root_graph(G)
        return True
    except NotALineGraph:
        return False
