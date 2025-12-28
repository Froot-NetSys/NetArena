import json
import traceback
from dotenv import load_dotenv
import openai
import copy
import pandas as pd
from prototxt_parser.prototxt import parse
from collections import Counter
import os
from solid_step_helper import getGraphData, clean_up_llm_output_func, check_list_equal, \
    node_attributes_are_equal, clean_up_output_graph_data, clean_up_updated_graph_data, \
    solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, \
    solid_step_list_child_nodes, solid_step_update_node_value, solid_step_rank_child_nodes, validate_llm_output
import networkx as nx
import json
import re
import time
import sys
import numpy as np
from error_check import SafetyChecker


# output the evaluation results to a jsonl file
OUTPUT_JSONL_DIR = 'logs/llm_agents'
OUTPUT_JSONL_FILE = 'gpt4.jsonl'


class BenchmarkEvaluator:
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def run_agent_output(self, current_query, golden_answer, llm_answer=None):
        """Evaluate a single query against the graph.

        If `llm_answer` is provided, it will be used directly (for A2A workflows).
        Otherwise, a locally instantiated agent (if available) will be called.
        """
        print("Query: ", current_query)

        G = self.graph_data
        
        # Shared namespace for exec/eval with all required dependencies
        exec_namespace = {
            'G': G, 'copy': copy, 'nx': nx, 'json': json,
            # Include all solid_step helper functions
            'solid_step_add_node_to_graph': solid_step_add_node_to_graph,
            'solid_step_counting_query': solid_step_counting_query,
            'solid_step_remove_node_from_graph': solid_step_remove_node_from_graph,
            'solid_step_list_child_nodes': solid_step_list_child_nodes,
            'solid_step_update_node_value': solid_step_update_node_value,
            'solid_step_rank_child_nodes': solid_step_rank_child_nodes,
        }
        
        if llm_answer is None:
            # Provide a useful error structure that's consistent with existing handling
            ret = {'type': 'error', 'data': 'No LLM response provided and no local LLM agent available.'}
        else:
            try:
                exec(llm_answer, exec_namespace)
                ret = eval("process_graph(copy.deepcopy(G))", exec_namespace)
            except Exception:
                ret = {'type': "error", 'data': traceback.format_exc()}
        
        # if the type of ret is string, turn it into a json object
        if isinstance(ret, str):
            try:
                ret = json.loads(ret)
            except:
                ret = {'type': "error", 'data': 'LLM output is not a valid JSON string.'}
        
        # Ensure LLM output is formatted correctly, and produces a valid graph.
        ret_graph_copy = None
        if validate_llm_output(ret) and ret['type'] != 'error':
            # Even if we cannot recover the graph for safety verification, still pass the output to the ground truth check.
            try:
                ret_graph_copy = clean_up_updated_graph_data(ret)
            except:
                ret = {'type': ret['type'], 'data': ret['data']} 
        else:
            ret = {'type': 'error', 'data': ret}
     
        # Clean up the updated graph (if it exists).
        if ret_graph_copy is not None:
            verifier = SafetyChecker(ret_graph=ret_graph_copy, ret_list=None)
            verifier_results, verifier_error = verifier.evaluate_all()
        else:
            verifier_results = False
            verifier_error = "The LLM code is not correct, so the safety checker is not applied."
        print("Verifier results: ", verifier_results, verifier_error)

        # Where we get the golden answer (ground truth) code for each query
        goldenAnswerCode = golden_answer

        # ground truth answer should already be checked to ensure it can run successfully
        # Use a fresh copy of the namespace to avoid pollution from LLM code
        gt_namespace = dict(exec_namespace)
        exec(goldenAnswerCode, gt_namespace)
        ground_truth_ret = eval("ground_truth_process_graph(copy.deepcopy(G))", gt_namespace)
        # if the type of ground_truth_ret is string, turn it into a json object
        if isinstance(ground_truth_ret, str):
            ground_truth_ret = json.loads(ground_truth_ret)

        # Add the verifier error to the ground truth result
        if ground_truth_ret['type'] == 'graph':
            ground_truth_ret_graph_copy = ground_truth_ret['data']
            gt_verifier = SafetyChecker(ret_graph=ground_truth_ret_graph_copy, ret_list=None)
            gt_verifier_results, gt_verifier_error = gt_verifier.evaluate_all()
        else:
            gt_verifier_results = True
            gt_verifier_error = ""
        print("Ground truth verifier results: ", gt_verifier_results, gt_verifier_error)
        
        if ret['type'] != 'graph':
            print("LLM code result: ", ret['data'])
            print("Ground truth result: ", ground_truth_ret['data'])

        ground_truth_ret['reply'] = goldenAnswerCode
        ret['reply'] = llm_answer

        print("=========Current query process is done!=========")

        return ret, ground_truth_ret, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, ret_graph_copy

    def ground_truth_check(self, requestData, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency):
        # Helper function to log results and avoid code duplication
        def log_result(is_correct):
            log_func = self.result_log_correct if is_correct else self.result_log_wrong
            res = log_func(requestData, task_label, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, 
                    query_run_latency, ground_truth_ret, ret)
            return res

        # Convert numeric data to strings for text type
        if ground_truth_ret['type'] == 'text':
            for r in (ret, ground_truth_ret):
                if isinstance(r['data'], int):
                    r['data'] = str(r['data'])
        
        # Define comparison strategies for different types
        comparison_strategies = {
            'text': lambda gt, llm: gt == llm,
            'list': lambda gt, llm: check_list_equal(gt, llm),
            'table': lambda gt, llm: gt == llm,
            'graph': lambda gt, llm: nx.is_isomorphic(
                nx.Graph(gt), 
                nx.Graph(llm), 
                node_match=node_attributes_are_equal
            )
        }

        # Get the appropriate comparison strategy and execute it
        # Sometimes LLM output doesn't fully match the expected data type, so we need this.
        try:
            compare_func = comparison_strategies[ground_truth_ret['type']]
            is_correct = compare_func(ground_truth_ret['data'], ret['data'])
            res = log_result(is_correct)
        except:
            print("Error during comparison: ", traceback.format_exc())
            res = log_result(False)
        return res

    def result_log_wrong(self, current_query, task_label, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, ground_truth_ret, ret):
        result_object = {
            "Query": current_query,
            "Label": task_label,
            "Result-Correctness": "Fail",
            "Result-Safety": "Pass" if verifier_results else "Fail",
            "GT-Result-Safety": "Pass" if gt_verifier_results else "Fail",
            "Result-Latency": query_run_latency,
            "Ground truth code": ground_truth_ret['reply'],
            "LLM code": ret['reply']
        }

        # Model output not always JSON serializable (nx.Graph, generator objects, etc), so have to check.
        model_output = ret['data']
        try:
            json.dumps(model_output)
        except (TypeError, OverflowError):
            model_output = str(ret['data'])

        if ret['type'] == 'error':
            result_object["Error"] = model_output  # Execution error details
        elif ground_truth_ret['type'] == 'graph':
            result_object["Error"] = "Two graphs are not identical."
        else:
            result_object["Ground truth exec"] = ground_truth_ret['data']
            result_object["LLM code exec"] = model_output
            result_object["Error"] = {
                "Ground truth": ground_truth_ret['data'],
                "Model output": model_output
            }

        # Add verifier error details if verification failed
        if not verifier_results:
            result_object["Verifier-Error"] = verifier_error
        if not gt_verifier_results:
            result_object["GT-Verifier-Error"] = gt_verifier_error
        
        return result_object

    def result_log_correct(self, current_query, task_label, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, ground_truth_ret, ret):
        result_object = {
            "Query": current_query,
            "Label": task_label,
            "Result-Correctness": "Pass",
            "Result-Safety": "Pass" if verifier_results else "Fail",
            "GT-Result-Safety": "Pass" if gt_verifier_results else "Fail",
            "Result-Latency": query_run_latency,
            "Ground truth code": ground_truth_ret['reply'],
            "LLM code": ret['reply']
        }
        if ground_truth_ret['type'] != 'graph':
            result_object["Ground truth exec"] = ground_truth_ret['data']
            result_object["LLM code exec"] = ret['data']
        
        # Add verifier error details if verification failed
        if not verifier_results:
            result_object["Verifier-Error"] = verifier_error
        if not gt_verifier_results:
            result_object["GT-Verifier-Error"] = gt_verifier_error
        
        return result_object


