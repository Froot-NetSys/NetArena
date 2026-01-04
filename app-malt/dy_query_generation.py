import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import jsonlines
from loguru import logger
import json
import os
from enum import Enum

from solid_step_helper import get_node_value_ranges, getGraphData, GRAPH_TOPOLOGY_DIR


class ComplexityLevel(Enum):
    LEVEL1 = 'level1'
    LEVEL2 = 'level2'
    LEVEL3 = 'level3'


def fetch_benchmark_queries(
    benchmark_path: str, 
    num_queries: int, 
    complexity_level: list[ComplexityLevel], 
    regenerate_query: bool = False,
    start_index: int = 0,
    end_index: int | None = None
) -> list[dict]:
    query_generator = QueryGenerator()

    if regenerate_query:
        logger.info("Generating new queries due to regenerate_query=True")
        query_generator.generate_queries(num_each_type=num_queries, complexity_level=complexity_level)
        query_generator.save_queries_to_file(benchmark_path)
    else:
        if not os.path.exists(benchmark_path):
            logger.info(f"Benchmark file {benchmark_path} does not exist. Generating new queries...")
            query_generator.generate_queries(num_each_type=num_queries, complexity_level=complexity_level)
            query_generator.save_queries_to_file(benchmark_path)
        else:
            logger.info(f"Loading existing benchmark from {benchmark_path}")
            query_generator.load_queries_from_file(benchmark_path)

    # the format is {"messages": [{"question": "XXX."}, {"answer": "YYY"}]}
    benchmark_data = []
    with jsonlines.open(benchmark_path) as reader:
        for obj in reader:
            benchmark_data.append(obj['messages'])
    
    # Skip to start_index if specified
    start_idx = max(start_index, 0)
    end_idx = len(benchmark_data) if not isinstance(end_index, int) else min(end_index, len(benchmark_data))
    if 0 < start_idx or end_idx < len(benchmark_data):
        logger.info(f"Starting from query index {start_idx} (skipping {start_idx} queries) and ending at {end_idx} (processing {end_idx - start_idx} queries).")
        if start_idx >= end_idx:
            logger.warning(f"Warning: start_index {start_idx} is greater than or equal to end index ({len(benchmark_data)})")
        benchmark_data = benchmark_data[start_idx:end_idx]

    return benchmark_data


class QueryGenerator:
    def __init__(self,):
        _, self.malt_real_graph = getGraphData()
        data_path = os.path.join(GRAPH_TOPOLOGY_DIR, 'node_value_ranges.json')
        node_value_ranges_path = data_path
        self.node_value_ranges = get_node_value_ranges(self.malt_real_graph, node_value_ranges_path)
        self.queries = []

    def generate_level_1_query_groundtruth(self, operation_type='add'):
        """
        Level-1 query: one operation.
        """
        if operation_type == 'add':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            if child_node == 'EK_PORT':
                parent_node = 'EK_PACKET_SWITCH'
            else:
                parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = f"new_{child_node}_{random.randint(1, 100)}"
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Add new node with name {child_node_name} type {child_node}, to {parent_node_name}. Return a graph."
            new_node = {'name': child_node_name, 'type': child_node}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                        new_node = {new_node}
                        parent_node_name = '{parent_node_name}'
                        graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                        return_object = {{'type': 'graph', 'data': graph_data}}
                        return return_object"""
            return template, ground_truth, new_node

        elif operation_type == 'remove':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            child_node_name = random.choice(self.node_value_ranges[child_node])

            template = f"Remove {child_node_name} from the graph. Return a graph."
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    child_node_name = '{child_node_name}'
                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)
                                    return_object = {{'type': 'graph', 'data': graph_data}}
                                    return return_object"""
            return template, ground_truth, child_node_name

        elif operation_type == 'count':
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_type = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Count the {child_node_type} in the {parent_node_name}. Return the count number as text."
            node1 = {'type': parent_node, 'name': parent_node_name}
            node2 = {'type': child_node_type, 'name': None}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    node1 = {node1}
                                    node2 = {node2}
                                    count = solid_step_counting_query(graph_data, node1, node2)
                                    return_object = {{'type': 'text', 'data': count}}
                                    return return_object"""
            return template, ground_truth, None

        elif operation_type == 'list':
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN', 'EK_RACK', 'EK_PACKET_SWITCH'])
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"List all the child nodes of {parent_node_name}. Return a list of child node names."
            node = {'type': parent_node, 'name': parent_node_name}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                        node = {node}
                        child_nodes = solid_step_list_child_nodes(graph_data, node)
                        return_object = {{'type': 'list', 'data': child_nodes}}
                        return return_object"""
            return template, ground_truth, None

        elif operation_type == 'update':
            child_node = random.choice(['EK_PORT'])
            child_node_name = random.choice(self.node_value_ranges[child_node])
            new_value = random.randint(1, 100)

            template = f"Update the physical capacity value of {child_node_name} to {new_value}. Return a graph."
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    child_node_name = '{child_node_name}'
                                    new_value = {new_value}
                                    graph_data = solid_step_update_node_value(graph_data, child_node_name, new_value)
                                    return_object = {{'type': 'graph', 'data': graph_data}}
                                    return return_object"""
            return template, ground_truth, child_node_name

        elif operation_type == 'rank':
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Rank all child nodes of {parent_node} type {parent_node_name} based on physical_capacity_bps attribute. Return a list of tuple, each tuple has child node name and its total physical capacity."
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                parent_node_name = '{parent_node_name}'
                                ranked_child_nodes = solid_step_rank_child_nodes(graph_data, parent_node_name)
                                return_object = {{'type': 'list', 'data': ranked_child_nodes}}
                                return return_object"""
            return template, ground_truth, None


    def create_level_one_dataset(self, num_each_type):
        # operations = ['update', 'add', 'count', 'remove', 'list', 'rank']
        operations = ['add', 'rank', 'remove', 'list']
        for operation in operations:
            for _ in range(num_each_type):
                query, ground_truth, new_node = self.generate_level_1_query_groundtruth(operation_type=operation)
                self.queries.append({
                    "messages": [
                    {"question": query},
                    {"answer": ground_truth},
                    {"task_label": f"capacity planning, level-1, {operation}"}
                    ]
                })
    
    def generate_level_2_query_sequential(self, operation_type_1='add', operation_type_2='count'):
        """
        Level-2 query: two operations, control sequence is sequential.
        """
        if operation_type_1 == 'add' and operation_type_2 == 'count':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            if child_node == 'EK_PORT':
                parent_node = 'EK_PACKET_SWITCH'
            else:
                parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = f"new_{child_node}_{random.randint(1, 100)}"
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Add {child_node_name} to {parent_node_name}. Count the {child_node} in {parent_node_name} in the updated graph. Return the count number as text."

            new_node = {'name': child_node_name, 'type': child_node}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    new_node = {new_node}
                                    parent_node_name = '{parent_node_name}'
                                    graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                                    node1 = {{"type": "{parent_node}", "name": "{parent_node_name}"}}
                                    node2 = {{"type": "{child_node}", "name": None}}
                                    count = solid_step_counting_query(graph_data, node1, node2)
                                    return_object = {{'type': 'text', 'data': count}}
                                    return return_object"""
            return template, ground_truth, new_node

        elif operation_type_1 == 'remove' and operation_type_2 == 'count':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = random.choice(self.node_value_ranges[child_node])
            parent_node_substring = '.'.join(child_node_name.split('.')[:-1])

            template = f"Remove {child_node_name} from the graph. Count the {child_node} in {parent_node_substring} in the updated graph. Return the count number as text."

            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    child_node_name = '{child_node_name}'
                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)
                                    node1 = {{"type": "{parent_node}", "name": "{parent_node_substring}"}}
                                    node2 = {{"type": "{child_node}", "name": None}}
                                    count = solid_step_counting_query(graph_data, node1, node2)
                                    return_object = {{'type': 'text', 'data': count}}
                                    return return_object"""
            return template, ground_truth, child_node_name
        
        elif operation_type_1 == 'add' and operation_type_2 == 'list':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = f"new_{child_node}_{random.randint(1, 100)}"
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Add {child_node_name} to {parent_node_name}. List direct child nodes of {parent_node_name} in the updated graph. Return a list of child nodes name."

            new_node = {'name': child_node_name, 'type': child_node}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    new_node = {new_node}
                                    parent_node_name = '{parent_node_name}'
                                    graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                                    node = {{"type": "{parent_node}", "name": "{parent_node_name}"}}
                                    child_nodes = solid_step_list_child_nodes(graph_data, node)
                                    return_object = {{'type': 'list', 'data': child_nodes}}
                                    return return_object"""
            return template, ground_truth, new_node
        
        elif operation_type_1 == 'add' and operation_type_2 == 'rank':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = f"new_{child_node}_{random.randint(1, 100)}"
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Add node with name '{child_node_name}' to {parent_node_name}. Rank direct child nodes of {parent_node_name} in the updated graph based on physical_capacity_bps attribute. Return a list of tuple, each tuple has node name and its total physical capacity."

            new_node = {'name': child_node_name, 'type': child_node}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    new_node = {new_node}
                                    parent_node_name = '{parent_node_name}'
                                    graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                                    ranked_child_nodes = solid_step_rank_child_nodes(graph_data, parent_node_name)
                                    return_object = {{'type': 'list', 'data': ranked_child_nodes}}
                                    return return_object"""
            return template, ground_truth, new_node
        
        elif operation_type_1 == 'remove' and operation_type_2 == 'list':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = random.choice(self.node_value_ranges[child_node])
            parent_node_substring = '.'.join(child_node_name.split('.')[:-1])

            template = f"Remove {child_node_name} from the graph. List direct child nodes of {parent_node_substring} in the updated graph. Return a list of child nodes name."

            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    child_node_name = '{child_node_name}'
                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)
                                    node = {{"type": "{parent_node}", "name": '{parent_node_substring}'}}
                                    child_nodes = solid_step_list_child_nodes(graph_data, node)
                                    return_object = {{'type': 'list', 'data': child_nodes}}
                                    return return_object"""
            return template, ground_truth, child_node_name
        
        elif operation_type_1 == 'remove' and operation_type_2 == 'rank':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = random.choice(self.node_value_ranges[child_node])
            parent_node_substring = '.'.join(child_node_name.split('.')[:-1])

            template = f"Remove {child_node_name} from the graph. Rank direct child nodes of {parent_node_substring} in the updated graph based on physical_capacity_bps attribute. Return a list of tuple, each tuple has node name and its total physical capacity."

            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    child_node_name = '{child_node_name}'
                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)
                                    parent_node_name = '{parent_node_substring}'
                                    ranked_child_nodes = solid_step_rank_child_nodes(graph_data, parent_node_name)
                                    return_object = {{'type': 'list', 'data': ranked_child_nodes}}
                                    return return_object"""
            return template, ground_truth, child_node_name
        
        
    def create_level_two_dataset(self, num_each_type):
        # operations = [('add', 'count'), ('remove', 'count'), ('add', 'list'), ('add', 'rank'), ('remove', 'list'), ('remove', 'rank')]
        operations = [('remove', 'list'), ('remove', 'rank'), ('remove', 'count')]
        for operation1, operation2 in operations:
            for _ in range(num_each_type):
                query, ground_truth, new_node = self.generate_level_2_query_sequential(operation_type_1=operation1, operation_type_2=operation2)
                self.queries.append({
                    "messages": [
                    {"question": query},
                    {"answer": ground_truth},
                    {"task_label": f"capacity planning, level-2, {operation1}-{operation2}"}
                    ]
                })

    def create_level_three_dataset(self, num_each_type):
        # operations = [('add', 'count'), ('remove', 'count'), ('add', 'list'), ('add', 'rank'), ('remove', 'list'), ('remove', 'rank')]
        operations = [('add', 'list'), ('add', 'rank'), ('add', 'count')]
        for operation1, operation2 in operations:
            for _ in range(num_each_type):
                query, ground_truth, new_node = self.generate_level_2_query_sequential(operation_type_1=operation1, operation_type_2=operation2)
                self.queries.append({
                    "messages": [
                    {"question": query},
                    {"answer": ground_truth},
                    {"task_label": f"capacity planning, level-3, {operation1}-{operation2}"}
                    ]
                })


    def genarate_level_3_query_for_loop(self, operation_type_1='add', operation_type_2='count'):
        """
        Level-2 query: two operations, control sequence is for-loop.
        For each parent node in the graph, add a new child node to it. Count the total number of child nodes in the updated graph. Return the counts.
        """
        if operation_type_1 == 'add' and operation_type_2 == 'count':
            parent_node_type = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_type = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node_names = self.node_value_ranges[parent_node_type]

            template = f"For each {parent_node_type}, add a new {child_node_type} to it. Count the total number of {child_node_type} in the updated graph. Return only the counts."
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    for parent_node_name in {parent_node_names}:
                                        new_node = {{"name": f"new_{child_node_type}_{{random.randint(1, 100)}}", "type": "{child_node_type}"}}
                                        node2 = {{"type": "{child_node_type}", "name": None}}
                                        graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                                    count = solid_step_counting_query(graph_data, node2)
                                    return_object = {{'type': 'text', 'data': count}}
                                    return return_object"""
            return template, ground_truth, None
        
        
    # def create_level_three_dataset(self, num_each_type):
    #     # TODO: level-3 query creation has bugs
    #     operations = [('add', 'rank')]
    #     for operation1, operation2 in operations:
    #         for _ in range(num_each_type):
    #             query, ground_truth, new_node = self.genarate_level_3_query_for_loop(operation_type_1=operation1, operation_type_2=operation2)
    #             self.queries.append({
    #                 "messages": [
    #                 {"question": query},
    #                 {"answer": ground_truth},
    #                 {"task_label": f"capacity planning, level-3, {operation1}-{operation2}"}
    #                 ]
    #             })
    
    def generate_queries(self, num_each_type=3, complexity_level=[ComplexityLevel.LEVEL1, ComplexityLevel.LEVEL2]):
        if ComplexityLevel.LEVEL1 in complexity_level:
            self.create_level_one_dataset(num_each_type)
        if ComplexityLevel.LEVEL2 in complexity_level:
            self.create_level_two_dataset(num_each_type)
        if ComplexityLevel.LEVEL3 in complexity_level:
            self.create_level_three_dataset(num_each_type)

    def save_queries_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for item in self.queries:
                f.write(json.dumps(item) + "\n")
    
    def load_queries_from_file(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                self.queries.append(json.loads(line))

# Usage
# query_generator = QueryGenerator()
# query_generator.generate_queries()
# query_generator.save_queries_to_file('data/benchmark_level_1.jsonl')
