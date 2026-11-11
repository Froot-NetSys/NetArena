import networkx as nx
import pytest

from solid_step_helper import check_graph_equal, solid_step_add_node_to_graph


def small_graph():
    # Same conventions as getGraphData: node id is the name, 'type' is a list, edge 'type' is a string
    graph = nx.DiGraph()
    graph.add_node('ju1.a1.m1', name='ju1.a1.m1', type=['EK_AGG_BLOCK'])
    graph.add_node('ju1.a1.m1.s2c1', name='ju1.a1.m1.s2c1', type=['EK_PACKET_SWITCH'])
    graph.add_edge('ju1.a1.m1', 'ju1.a1.m1.s2c1', type='RK_CONTAINS')
    return graph


def nodes_and_edges_by_name(graph):
    nodes = {attrs['name']: attrs for _, attrs in graph.nodes(data=True)}
    edges = {(graph.nodes[u]['name'], graph.nodes[v]['name'], attrs['type']) for u, v, attrs in graph.edges(data=True)}
    return nodes, edges


def test_add_port_follows_graph_conventions():
    graph = solid_step_add_node_to_graph(small_graph(), {'name': 'new_EK_PORT_5', 'type': 'EK_PORT'}, 'ju1.a1.m1.s2c1')
    nodes, edges = nodes_and_edges_by_name(graph)
    assert nodes['new_EK_PORT_5'] == {'name': 'new_EK_PORT_5', 'type': ['EK_PORT'], 'physical_capacity_bps': 1000}
    assert ('ju1.a1.m1.s2c1', 'new_EK_PORT_5', 'RK_CONTAINS') in edges


def test_add_packet_switch_gets_named_port():
    graph = solid_step_add_node_to_graph(small_graph(), {'name': 'new_EK_PACKET_SWITCH_7', 'type': 'EK_PACKET_SWITCH'}, 'ju1.a1.m1')
    nodes, edges = nodes_and_edges_by_name(graph)
    assert nodes['new_EK_PACKET_SWITCH_7'] == {'name': 'new_EK_PACKET_SWITCH_7', 'type': ['EK_PACKET_SWITCH']}
    assert nodes['new_EK_PACKET_SWITCH_7.p1'] == {'name': 'new_EK_PACKET_SWITCH_7.p1', 'type': ['EK_PORT'], 'physical_capacity_bps': 1000}
    assert ('ju1.a1.m1', 'new_EK_PACKET_SWITCH_7', 'RK_CONTAINS') in edges
    assert ('new_EK_PACKET_SWITCH_7', 'new_EK_PACKET_SWITCH_7.p1', 'RK_CONTAINS') in edges
    assert len(nodes) == 4


def add_port(graph):
    return solid_step_add_node_to_graph(graph, {'name': 'new_EK_PORT_5', 'type': 'EK_PORT'}, 'ju1.a1.m1.s2c1')


def reverse_edge(graph):
    graph.remove_edge('ju1.a1.m1', 'ju1.a1.m1.s2c1')
    graph.add_edge('ju1.a1.m1.s2c1', 'ju1.a1.m1', type='RK_CONTAINS')
    return graph


def set_edge_type(edge_type):
    def variant(graph):
        graph.edges['ju1.a1.m1', 'ju1.a1.m1.s2c1']['type'] = edge_type
        return graph
    return variant


@pytest.mark.parametrize('variant, equal', [
    pytest.param(lambda graph: graph.copy(), True, id='copy'),
    pytest.param(add_port, False, id='extra-node'),
    pytest.param(reverse_edge, False, id='reversed-edge'),
    pytest.param(set_edge_type('RK_CONTROLS'), False, id='wrong-edge-type'),
    pytest.param(set_edge_type(['RK_CONTAINS']), True, id='list-edge-type'),
])
def test_check_graph_equal(variant, equal):
    assert check_graph_equal(small_graph(), variant(small_graph())) == equal
