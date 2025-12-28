import numpy as np
import jsonlines
import os
import asyncio
import httpx
import time
import matplotlib.pyplot as plt
import argparse
from scipy import stats
import cattrs
from loguru import logger
from dataclasses import dataclass, field

from netarena.agent_client import AgentClientConfig, AgentClient, PromptType
from dy_query_generation import QueryGenerator, ComplexityLevel
from malt_env import BenchmarkEvaluator
from text_utils import create_query_prompt, extract_code_output


@dataclass
class MaltConfig:
    """
    Structured container for the app settings.
    """
    complexity_level: list[ComplexityLevel] = field(default_factory=lambda: [ComplexityLevel.LEVEL1, ComplexityLevel.LEVEL2])
    prompt_type: PromptType = PromptType.ZEROSHOT_BASE
    num_queries: int = 10
    output_dir: str = 'output'
    output_file: str = 'eval_results.jsonl'
    dynamic_benchmark_path: str = 'malt_benchmark.jsonl'
    regenerate_query: bool = False
    start_index: int = 0
    end_index: int | None = None
    agent_client_configs: list[AgentClientConfig] = field(default_factory=list)

    def __post_init__(self):
        names = [config.name for config in self.agent_client_configs]
        if len(names) != len(set(names)):
            raise ValueError(f'Bad agent client configuration. Different agents cannot have the same name.')


# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--prompt_type', type=PromptType, default='base', help='Choose the prompt type', choices=[pt for pt in PromptType])
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', nargs='+', default=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--output_dir', type=str, default='logs/llm_agents', help='Directory to save output JSONL file')
    parser.add_argument('--output_file', type=str, default='gpt4o.jsonl', help='Name of the output JSONL file')
    parser.add_argument('--dynamic_benchmark_path', type=str, default='data/benchmark_malt.jsonl', help='Path to save dynamic dataset')
    parser.add_argument('--regenerate_query', action='store_true', help='Whether to regenerate benchmark queries or load existing ones')
    parser.add_argument('--start_index', type=int, default=0, help='Start index of the queries to run (zero indexed).')
    parser.add_argument('--end_index', type=int, default=None, help='End index of the queries to run (zero indexed).')
    return parser.parse_args()


def fetch_benchmark_queries(app_config: MaltConfig, query_generator: QueryGenerator | None = None) -> list[dict]:
    if query_generator is None:
        query_generator = QueryGenerator()

    benchmark_path = app_config.dynamic_benchmark_path
    if app_config.regenerate_query:
        logger.info("Generating new queries due to regenerate_query=True")
        query_generator.generate_queries(num_each_type=app_config.num_queries, complexity_level=app_config.complexity_level)
        query_generator.save_queries_to_file(benchmark_path)
    else:
        if not os.path.exists(benchmark_path):
            logger.info(f"Benchmark file {benchmark_path} does not exist. Generating new queries...")
            query_generator.generate_queries(num_each_type=app_config.num_queries, complexity_level=app_config.complexity_level)
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
    start_idx = max(app_config.start_index, 0)
    end_idx = len(benchmark_data) if not isinstance(app_config.end_index, int) else min(app_config.end_index, len(benchmark_data))
    if 0 < start_idx or end_idx < len(benchmark_data):
        logger.info(f"Starting from query index {start_idx} (skipping {start_idx} queries) and ending at {end_idx} (processing {end_idx - start_idx} queries).")
        if start_idx >= end_idx:
            logger.warning(f"Warning: start_index {start_idx} is greater than or equal to end index ({len(benchmark_data)})")
        benchmark_data = benchmark_data[start_idx:end_idx]

    return benchmark_data


async def evaluate_on_queries(config: MaltConfig):
    """
    Handles querying each agent on the generated benchmark and evaluating the LLM generated code.
    Yields the evaluation results for each query along with information about the agent evaluated.
    """
    # dynamically generate or load existing queries
    query_generator = QueryGenerator()
    # Load the evaluator
    evaluator = BenchmarkEvaluator(graph_data=query_generator.malt_real_graph)

    # the format is {"messages": [{"question": "XXX."}, {"answer": "YYY"}]}
    benchmark_data = fetch_benchmark_queries(config, query_generator=query_generator)

    # Helper function to associate awaitable with a key.
    async def key_value(key, value):
        return key, await value
    
    # for each object in the benchmark list, get the question and answer
    # TODO: Separate http clients (and context manager) for each agent to enable different HTTP configs. For now, just remember to clean up shared client.
    async with httpx.AsyncClient() as httpx_client:
        # Establish connections to the agents.
        clients = [AgentClient(agent_client_config) for agent_client_config in config.agent_client_configs]
        agents: list[AgentClient] = []
        for client in asyncio.as_completed([c.start(httpx_client) for c in clients]):
            try:
                agent = await client
                agents.append(agent)
            except Exception as e:
                logger.debug(f'Connection failed: {e}')
                logger.warning(f'Failed to connect to agent server. Skipping...')
        if not agents:
            raise ConnectionError('Could not connect to any agent servers. Aborting assessment.')
        
        # Serialize agent info without any potentially sensitive info (e.g. HTTP headers).
        name_to_config_map = {agent.config.name: agent.config.serialize_omit_secrets() for agent in agents}

        # Skip to start_index if specified
        start_idx = max(config.start_index, 0)
        for i, obj in enumerate(benchmark_data):
            # Calculate actual index (for logging)
            actual_idx = i + start_idx
            logger.info(f"Processing query {actual_idx} of {len(benchmark_data) + start_idx - 1}")
            
            # obj is a list of dictionaries, load question, answer, task_label from it
            for item in obj:
                if 'question' in item:
                    current_query = item['question']
                elif 'answer' in item:
                    golden_answer = item['answer']
                elif 'task_label' in item:
                    task_label = item['task_label']
            
            # Query the LLM agent(s) and evaluate the results.
            # TODO: Make this capable of yielding from multiple agents concurrently (instead of waiting for each agent to finish).
            llm_answers = []
            for agent in agents:
                prompt = create_query_prompt(current_query, agent.config.prompt_type)
                fut = key_value(agent.config.name, agent.handle_query(prompt))
                llm_answers.append(fut)
            for response in asyncio.as_completed(llm_answers):
                start_time = time.perf_counter()
                agent_name, llm_answer = await response
                query_run_latency = time.perf_counter() - start_time
                if llm_answer is None:
                    logger.warning(f'Could not obtain response from agent "{agent_name}" for query {actual_idx}. Skipping...')
                    continue
                code = extract_code_output(llm_answer)
                logger.debug(f"LLM Answer: \n{code}")
                ret, ground_truth_ret, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, ret_graph_copy = evaluator.run_agent_output(current_query, golden_answer, llm_answer=code)
                result_object = evaluator.ground_truth_check(current_query, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency)
                result_object['agent_info'] = name_to_config_map[agent_name]
                yield result_object

# An example of how to use main.py with input args
# Example usage:
# python main.py --llm_agent_type AzureGPT4Agent --num_queries 2 --complexity_level level1 --output_dir logs/llm_agents --output_file gpt4o.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl

async def main(args):
    # Validate command line args.
    benchmark_config = cattrs.structure(vars(args), MaltConfig)

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create the output file if it does not exist
    output_path = os.path.join(args.output_dir, args.output_file)
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            pass

    eval_results = evaluate_on_queries(benchmark_config)

    # Collect results and write to file.
    results = []
    try:
        with jsonlines.open(output_path, mode='a') as writer:
            async for obj in eval_results:
                results.append(obj)
                writer.write(obj)
    except Exception as e:
        logger.debug(f'Assessment failure reason: {e}')
        logger.error("Agent assessment failed to proceed. Aborting.")
        return

    # TODO: Move the plotting code to a separate file.
    # Analyze the results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # group the results by task label
    grouped_results = {}
    for result in results:
        task_label = result["Label"]
        if task_label not in grouped_results:
            grouped_results[task_label] = []
        grouped_results[task_label].append(result)

    task_labels = list(grouped_results.keys())
    avg_latencies = []
    std_latencies = []
    for task_label in task_labels:
        latencies = [result["Result-Latency"] for result in grouped_results[task_label]]
        avg_latencies.append(np.mean(latencies))
        std_latencies.append(np.std(latencies))

    # create figs directory if it doesn't exist
    figs_dir = os.path.join(args.output_dir, 'figs')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    # plot the average query run latency for each task label with error bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(task_labels, avg_latencies, color='skyblue', yerr=std_latencies, capsize=5)
    plt.xlabel('Task Label')
    plt.ylabel('Average Query Run Latency (seconds)')
    plt.title(f'Average Query Run Latency ({args.llm_model_type}, Avg={np.mean(avg_latencies):.2f}s ±{np.mean(std_latencies):.2f}s)')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_latencies[i],
                f'±{std_latencies[i]:.2f}',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f'average_latency_{args.llm_model_type}_{timestamp}.png'), dpi=300)
    # plt.show()

    # plot the pass rate of correctness for each task label
    correctness_pass_rates = [sum(1 for result in grouped_results[task_label] if result["Result-Correctness"] == "Pass") / len(grouped_results[task_label]) * 100 for task_label in task_labels]
    
    # Calculate Standard Error of Mean (SEM) for each task label's pass rate
    sem_values = []
    sample_sizes = []   
    
    for task_label in task_labels:
        # Get sample size for this task
        n = len(grouped_results[task_label])
        sample_sizes.append(n)
        
        # Calculate SEM using scipy
        binary_outcomes = [1 if result["Result-Correctness"] == "Pass" else 0 for result in grouped_results[task_label]]
        scipy_sem = stats.sem(binary_outcomes, ddof=0) * 100
        sem_values.append(scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    error_margins = [1.96 * sem for sem in sem_values]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(task_labels, correctness_pass_rates, color='green', yerr=error_margins, capsize=5)
    plt.xlabel('Task Label')
    plt.ylabel('Correctness Pass Rate (%)')
    # print the average pass rate and error margin on the title
    avg_pass_rate = np.mean(correctness_pass_rates)
    error_margin = np.mean(error_margins)
    plt.title(f'Correctness Pass Rate ({args.llm_model_type}, N={sample_sizes}, Avg={avg_pass_rate:.2f}% ±{error_margin:.2f}%)')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error_margins[i],
                f'±{error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f'correctness_pass_rate_{args.llm_model_type}_{timestamp}.png'), dpi=300)
    # plt.show()
    
    # plot the pass rate of safety for each task label
    safety_pass_rates = [sum(1 for result in grouped_results[task_label] if result["Result-Safety"] == "Pass") / len(grouped_results[task_label]) * 100 for task_label in task_labels]
    
    # Calculate SEM for safety pass rates
    safety_sem_values = []
    for task_label in task_labels:
        binary_outcomes = [1 if result["Result-Safety"] == "Pass" else 0 for result in grouped_results[task_label]]
        scipy_sem = stats.sem(binary_outcomes, ddof=0) * 100
        safety_sem_values.append(scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    safety_error_margins = [1.96 * sem for sem in safety_sem_values]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(task_labels, safety_pass_rates, color='orange', yerr=safety_error_margins, capsize=5)
    plt.xlabel('Task Label')
    plt.ylabel('Safety Pass Rate (%)')
    # print the average pass rate and error margin on the title
    avg_pass_rate = np.mean(safety_pass_rates)
    error_margin = np.mean(safety_error_margins)
    plt.title(f'Safety Pass Rate ({args.llm_model_type}, N={sample_sizes}, Avg={avg_pass_rate:.2f}% ±{error_margin:.2f}%)')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + safety_error_margins[i],
                f'±{safety_error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f'safety_pass_rate_{args.llm_model_type}_{timestamp}.png'), dpi=300)
    # plt.show()


# run the main function
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))