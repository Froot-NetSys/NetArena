import json
import networkx as nx
from networkx.readwrite import json_graph
from prototxt_parser.prototxt import parse
from collections import Counter
import re
import os

_file_dir = os.path.dirname(os.path.abspath(__file__))
GRAPH_TOPOLOGY_DIR = os.path.join(_file_dir, 'data')


def getGraphData():
    data_path = os.path.join(GRAPH_TOPOLOGY_DIR, 'malt-graph.textproto.txt')
    input_string = open(data_path).read()
    parsed_dict = parse(input_string)

    # Load MALT data
    G = nx.DiGraph()

    # Insert all the entities as nodes
    for entity in parsed_dict['entity']:
        # Check if the node exists
        if entity['id']['name'] not in G.nodes:
            G.add_node(entity['id']['name'], type=[entity['id']['kind']], name=entity['id']['name'])
        else:
            G.nodes[entity['id']['name']]['type'].append(entity['id']['kind'])
        # Add all the attributes
        for key, value in entity.items():
            if key == 'id':
                continue
            for k, v in value.items():
                G.nodes[entity['id']['name']][k] = v

    # Insert all the relations as edges
    for relation in parsed_dict['relationship']:
        G.add_edge(relation['a']['name'], relation['z']['name'], type=relation['kind'])

    rawData = json_graph.node_link_data(G)

    return rawData, G


def get_node_value_ranges(graph, saved_path):
    """For malt_real_graph, save node value range of each node type"""
    node_value_ranges = {}
    for node in graph.nodes:
        node_type = graph.nodes[node]['type'][0]
        node_value = graph.nodes[node]['name']
        if node_type not in node_value_ranges:
            node_value_ranges[node_type] = []
        node_value_ranges[node_type].append(node_value)
    # save the node_value_ranges to a json file
    with open(saved_path, 'w') as f:
        json.dump(node_value_ranges, f)

    return node_value_ranges


def solid_step_add_node_to_graph(graph_data, new_node, parent_node_name=None):
    """
    Adds a new node to the graph. Optionally adds an edge to a parent node with a specified relationship type.

    :param graph_data: The existing graph (a NetworkX graph or similar).
    :param new_node: A dictionary containing the new node's attributes (e.g., name, type).
    :param parent_node_name: Name of the parent node (optional). If provided, a relationship edge will be added.
    :return: updated graph data.
    """
    # Create a new unique node ID
    new_node_id = len(graph_data.nodes) + 1

    # if new_node['type'] is EK_PORT, add a new attribute 'physical_capacity_bps' to the new node
    if 'EK_PORT' in new_node['type']:
        new_node['physical_capacity_bps'] = 1000

    # if new_node['type'] is EK_PACKET_SWITCH, add a EK_PORT node as its child node with a new attribute 'physical_capacity_bps'
    if new_node['type'] == 'EK_PACKET_SWITCH':
        new_port_node = {'name': f'{new_node}.p1', 'type': 'EK_PORT'}
        new_port_node_id = len(graph_data.nodes) + 2  # Ensure unique ID
        new_port_node['physical_capacity_bps'] = 1000
        # Add the new node to the graph
        node_attrs = {'name': new_node['name'], 'type': new_node['type']}
        if 'physical_capacity_bps' in new_node:
            node_attrs['physical_capacity_bps'] = new_node['physical_capacity_bps']
        graph_data.add_node(new_node_id, **node_attrs)
        # Add the new port node to the graph
        graph_data.add_node(new_port_node_id, name=new_port_node['name'], type=new_port_node['type'], physical_capacity_bps=new_port_node['physical_capacity_bps'])
        # Add the edge between the new node and the new port node
        graph_data.add_edge(new_node_id, new_port_node_id, type='RK_CONTAINS')
    
    # Add the new node to the graph
    node_attrs = {'name': new_node['name'], 'type': new_node['type']}
    if 'physical_capacity_bps' in new_node:
        node_attrs['physical_capacity_bps'] = new_node['physical_capacity_bps']
    graph_data.add_node(new_node_id, **node_attrs)

    # If a parent node is specified, add an edge between parent and the new node
    if parent_node_name:
        parent_node_id = None
        for node in graph_data.nodes:
            if graph_data.nodes[node].get('name') == parent_node_name:
                parent_node_id = node
                break
        
        # Only add the edge if parent_node_id was found
        if parent_node_id is not None:
            graph_data.add_edge(parent_node_id, new_node_id, type='RK_CONTAINS')
            
    # For testing
    # parent_node_name = 'ju1.a1.m1'
    # new_node = {'name': 'new_port', 'type': 'EK_PORT'}
    # malt_graph = solid_step_add_node_to_graph(malt_real_graph, new_node, parent_node_name)
    return graph_data


def solid_step_remove_node_from_graph(graph_data, node_name):
    """
    Removes a node from the graph. Also removes any edges connected to the node.

    :param graph_data: The existing graph (a NetworkX graph or similar).
    :param node_name: The name of the node to be removed.
    :return: updated graph data.
    """
    # Find the node ID by name
    node_id = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == node_name:
            node_id = node
            break

    if node_id is None:
        print(f"Node with name '{node_name}' not found.")
        return graph_data

    # Remove the node and its edges from the graph
    graph_data.remove_node(node_id)

    return graph_data



# create a function for calculating the counting queries
def solid_step_counting_query(graph_data, node1, node2=None):
    """
    Count the number of node2 contained within node1 in the graph.
    or
    Count the total number of node1 contained in the graph.
    """
    if node2 is None:
        # directly count the total number of node1 in the graph
        total_count_node1_type = 0
        node1_type = node1['type']
        for node in graph_data.nodes(data=True):
            if node1_type in node[1]['type']:
                total_count_node1_type += 1
        return total_count_node1_type
    
    if node2:
        # Find the target node1
        target_node1 = None
        for node in graph_data.nodes:
            if graph_data.nodes[node].get('name') == node1['name']:
                target_node1 = node
                break

        if target_node1 is None:
            print(f"Node1 {node1['name']} not found")
            return {node1['name']: 'not found'}

        # Use BFS to count all node2 contained within node1
        node2_count = 0
        queue = [target_node1]
        visited = set()

        while queue:
            current_node = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)
            for edge in graph_data.out_edges(current_node, data=True):
                if edge[2]['type'] == 'RK_CONTAINS':
                    destination_node = edge[1]
                    if node2['type'] in graph_data.nodes[destination_node]['type']:
                        node2_count += 1
                    queue.append(destination_node)

        return node2_count

def solid_step_list_child_nodes(graph_data, parent_node):
    """
    list all nodes that are directly contained within the parent node
    """
    child_nodes = []
    parent_node_id = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == parent_node['name']:
            parent_node_id = node
            break

    if parent_node_id is None:
        print(f"Parent node with name '{parent_node['name']}' not found.")
        return child_nodes

    for edge in graph_data.out_edges(parent_node_id, data=True):
        if edge[2]['type'] == 'RK_CONTAINS':
            child_nodes.append(graph_data.nodes[edge[1]])
    
    # only return the name of the child nodes
    child_nodes_name = [node['name'] for node in child_nodes]

    return child_nodes_name
    
def solid_step_update_node_value(graph_data, child_node_name, new_value):
    """
    Update the physical_capacity_bps attribute of a child node in the graph if it is of type EK_PORT.
    """
    # Find the node ID by name
    child_node_id = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == child_node_name:
            child_node_id = node
            break

    if child_node_id is None:
        print(f"Node with name '{child_node_name}' not found.")
        return graph_data, child_node_name, new_value

    # Check if the node is of type EK_PORT and update its physical_capacity_bps attribute
    if 'EK_PORT' in graph_data.nodes[child_node_id]['type']:
        graph_data.nodes[child_node_id]['physical_capacity_bps'] = new_value
    else:
        print(f"Node with name '{child_node_name}' is not of type EK_PORT.")

    return graph_data

def solid_step_rank_child_nodes(graph_data, parent_node_name):
    """
    Rank the child nodes of a parent node by their total physical capacity in descending order.
    """
    # Find the parent node ID by name
    parent_node_id = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == parent_node_name:
            parent_node_id = node
            break
    if parent_node_id is None:
        print(f"Parent node with name '{parent_node_name}' not found.")
        return []

    # Initialize a list to store child nodes and their total physical capacity
    child_nodes_capacity = []

    # Find all child nodes and calculate their total physical capacity
    for edge in graph_data.out_edges(parent_node_id, data=True):
        if edge[2]['type'] == 'RK_CONTAINS':
            child_node = edge[1]
            total_physical_capacity_bps = 0
            if 'EK_PORT' in graph_data.nodes[child_node]['type']:
                total_physical_capacity_bps += graph_data.nodes[child_node].get('physical_capacity_bps', 0)
            for child_edge in graph_data.out_edges(child_node, data=True):
                if child_edge[2]['type'] == 'RK_CONTAINS':
                    grandchild_node = child_edge[1]
                    if 'EK_PORT' in graph_data.nodes[grandchild_node]['type']:
                        total_physical_capacity_bps += graph_data.nodes[grandchild_node].get('physical_capacity_bps', 0)
            child_nodes_capacity.append((graph_data.nodes[child_node], total_physical_capacity_bps))

    # Sort the child nodes by their total physical capacity in descending order
    child_nodes_capacity.sort(key=lambda x: x[1], reverse=True)

    # return only the sorted child nodes name and each of total physical capacity
    sorted_child_nodes_names = [(node['name'], capacity) for node, capacity in child_nodes_capacity]

    return sorted_child_nodes_names

def clean_up_llm_output_func(answer):
    '''
    Extract only the def process_graph() funtion from the output of LLM
    :param answer: output of LLM
    :return: cleaned function
    '''
    # If the function has "process_graph" in it, assume that this is the answer code.
    regex = re.compile(r'def\s+([a-zA-Z_0-9]*process_graph[a-zA-Z_0-9]*)')
    answer = regex.sub('def process_graph', answer)
    start = answer.find("def process_graph")
    if start == -1:
        return ""  # Return empty string if process_graph function not found
        
    # Find the code block ending
    code_block_end = answer.find("```", answer.find("```", start))
    
    # If we found proper code block markers
    if code_block_end != -1:
        clean_code = answer[start:code_block_end].strip()
    else:
        # Fallback to extract until the end of the string
        clean_code = answer[start:].strip()
    
    # Remove the lines that have "import package" in the code
    clean_code = '\n'.join([line for line in clean_code.split('\n') if not line.strip().startswith("import")])

    return clean_code

def check_list_equal(lst1, lst2):
    # check list type. if list1 is a [['ju1.a1.m1.s2c2', 0], ['ju1.a1.m1.s2c3', 0]], then convert to [('ju1.a1.m1.s2c2', 0), ('ju1.a1.m1.s2c3', 0)]
    if not isinstance(lst1, list) or not isinstance(lst2, list):
        return False
    if lst1 and isinstance(lst1[0], list):
        lst1 = [tuple(i) for i in lst1]
    if lst2 and isinstance(lst2[0], list):
        lst2 = [tuple(i) for i in lst2]
    return Counter(lst1) == Counter(lst2)


def validate_llm_output(ret):
    """
    Verify the output of the LLM by checking if it is in the valid format.
    :param ret: The output of the LLM's answer code.
    :return: True if the output is valid, False otherwise.

    The criteria for a valid output is as follows:
        ret = {
            'data': answer,
            'type': str,
            'updated_graph': dict,
        }
    Note that the function treats updated_graph as optional. 
    """
    if not isinstance(ret, dict):
        return False
    
    if 'data' not in ret or 'type' not in ret:
        return False

    return True


def clean_up_output_graph_data(ret):
    if isinstance(ret['data'], nx.Graph):
        # Create a nx.graph copy, so I can compare two nx.graph later directly
        jsonGraph = nx.node_link_data(ret['data'])
        ret_graph_copy = json_graph.node_link_graph(jsonGraph)

    else:  # Convert the jsonGraph back to nx.graph, to check if they are identical later
        ret_graph_copy = json_graph.node_link_graph(ret['data'])
        ret['data'] = json_graph.node_link_graph(ret['data'])

    return ret_graph_copy

def clean_up_updated_graph_data(ret):
    """
    Makes a copy of the updated_graph from JSON format to nx.Graph and stores the JSON graph in the 'updated_graph' key.
    If the 'data' key doesn't contain a nx.Graph (when applicable), it will use the updated_graph instead.
    """
    if 'updated_graph' not in ret:
        raise ValueError("updated_graph not found in ret")

    if isinstance(ret['updated_graph'], nx.Graph):
        # Create a nx.graph copy, so I can compare two nx.graph later directly
        ret_graph_copy = ret['updated_graph']
        jsonGraph = nx.node_link_data(ret['updated_graph'])
        ret['updated_graph'] = jsonGraph
    else:  # Convert the jsonGraph back to nx.graph, to check if they are identical later
        jsonGraph = ret['updated_graph']
        ret_graph_copy = json_graph.node_link_graph(jsonGraph)

    # If the data does not contain a graph, fall back to the updated_graph.
    if ret['type'] == 'graph' and not isinstance(ret['data'], nx.Graph):
        ret['data'] = json_graph.node_link_graph(jsonGraph)

    return ret_graph_copy

def node_attributes_are_equal(node1_attrs, node2_attrs):
    # Check if both nodes have the exact same set of attributes
    if set(node1_attrs.keys()) != set(node2_attrs.keys()):
        return False

    # Check if all attribute values are equal
    for attr_name, attr_value in node1_attrs.items():
        if attr_value != node2_attrs[attr_name]:
            return False

    return True