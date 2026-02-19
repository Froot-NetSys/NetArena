import argparse
from test_function import static_benchmark_run_modify, run_benchmark_parallel
from datetime import datetime
from multiprocessing import Process



# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="Qwen/Qwen2.5-72B-Instruct", help='Choose the LLM agent')# choices=["GPT-Agent","Qwen/Qwen2.5-72B-Instruct","ReAct_Agent"]
    parser.add_argument('--num_queries', type=int, default=1, help='Number of queries to generate for each type')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/NetArena_benchmark/app-route/results", 
                        help='The directory where all output files will be saved. The directory will be created if it does not exist.')
    parser.add_argument('--max_iteration', type=int, default=10, help='Choose maximum trials for a query')
    parser.add_argument('--vllm', type=int, default=1, choices=[0, 1], help='Enable vllm if set to 1')
    parser.add_argument('--num_gpus', type=int, default=1, help='The number of GPUs to use for tensor parallelism (VLLM).')
    parser.add_argument('--benchmark_path', type=str, default='error_config.json', help='Where to save the generated benchmark relative to root_dir (or where to look for it).')
    parser.add_argument('--static_benchmark_generation', type=int, default=0, choices=[0, 1], help='If set to 1, generate static benchmark')
    parser.add_argument('--prompt_type', type=str, default="base", choices=["base", "cot", "few_shot_basic"], help='Choose the prompt type')
    parser.add_argument('--parallel', type=int, default=1, choices=[0, 1], help='If set to 1, run in parallel')
    return parser.parse_args()

# Call the appropriate function based on the full_test argument
if __name__ == "__main__":
    args = parse_args()
    if args.parallel == 1:
        start_time = datetime.now()
        run_benchmark_parallel(args)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Benchmark completed in {duration}")
    else:
        # Run the benchmark in a single process
        start_time = datetime.now()
        static_benchmark_run_modify(args)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Benchmark completed in {duration}")
