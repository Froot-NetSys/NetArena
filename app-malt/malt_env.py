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
    check_graph_equal, clean_up_output_graph_data, clean_up_updated_graph_data, \
    solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, \
    solid_step_list_child_nodes, solid_step_update_node_value, solid_step_rank_child_nodes, validate_llm_output
import networkx as nx
import jsonlines
import json
import re
import time
import sys
import numpy as np
from llm_model import AzureGPT4Agent, GoogleGeminiAgent, QwenModel, QwenModel_finetuned, ReAct_Agent
from error_check import SafetyChecker


# output the evaluation results to a jsonl file
OUTPUT_JSONL_DIR = 'logs/llm_agents'
OUTPUT_JSONL_FILE = 'gpt4.jsonl'


class BenchmarkEvaluator:
    def __init__(self, graph_data, llm_model_type, prompt_type, model_path=None):
        self.graph_data = graph_data
        # Call the output code from LLM agents file
        if llm_model_type == "AzureGPT4Agent":
            self.llm_agent = AzureGPT4Agent(prompt_type)
        elif llm_model_type == "GoogleGeminiAgent":
            self.llm_agent = GoogleGeminiAgent(prompt_type)
        elif llm_model_type == "Qwen2.5-72B-Instruct":
            self.llm_agent = QwenModel(prompt_type)
        elif llm_model_type == "QwenModel_finetuned":
            self.llm_agent = QwenModel_finetuned(prompt_type, model_path=model_path)
        elif llm_model_type == "ReAct_Agent":
            self.llm_agent = ReAct_Agent(prompt_type='base')

    def userQuery(self, current_query, golden_answer):
        # for each prompt in the prompt_list, append it as the value of {'query': prompt}
        print("Query: ", current_query)

        G = self.graph_data
        
        start_time = time.time()

        llm_answer = self.llm_agent.call_agent(current_query)
        print("LLM answer: ", llm_answer)

        try:
            exec(llm_answer)
            ret = eval("process_graph(copy.deepcopy(G))")
        except Exception:
            ret = {'type': "error", 'data': traceback.format_exc()}
        
        query_run_latency = time.time() - start_time

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
        exec(goldenAnswerCode)
        ground_truth_ret = eval("ground_truth_process_graph(copy.deepcopy(G))")
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

        return ret, ground_truth_ret, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, ret_graph_copy

    def ground_truth_check(self, requestData, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, output_path):
        # Helper function to log results and avoid code duplication
        def log_result(is_correct):
            log_func = self.result_log_correct if is_correct else self.result_log_wrong
            log_func(requestData, task_label, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, 
                    query_run_latency, ground_truth_ret, ret, output_path)

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
            'graph': lambda gt, llm: check_graph_equal(gt, llm)
        }

        # Get the appropriate comparison strategy and execute it
        compare_func = comparison_strategies.get(ground_truth_ret['type'])
        if compare_func:
            # Sometimes LLM output doesn't fully match the expected data type, so we need this.
            try:
                is_correct = compare_func(ground_truth_ret['data'], ret['data'])
                log_result(is_correct)
            except:
                print("Error during comparison: ", traceback.format_exc())
                log_result(False)

    def result_log_wrong(self, current_query, task_label, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, ground_truth_ret, ret, output_path):
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

        # Save result_object into a JsonLine file
        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(result_object)
        
        return None

    def result_log_correct(self, current_query, task_label, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, ground_truth_ret, ret, output_path):
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
        
        # Save result_object into a JsonLine file
        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(result_object)
        
        return None


