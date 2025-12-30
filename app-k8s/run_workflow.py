import os
import subprocess
import argparse
import json
from datetime import datetime
import shutil
import time
import asyncio
from dataclasses import dataclass, field
from cattrs import structure
from loguru import logger
import httpx

from file_util import file_write, summary_tests, plot_metrics, plot_summary_results
from inject_errors import inject_config_errors_into_policies, generate_config
from correctness_check import correctness_check, create_debug_container, EXPECTED_RESULTS
from correct_policy import copy_yaml_to_new_folder
from llm_agent import LLMAgent
from deploy_policies import deploy_policies
from netarena.agent_client import AgentClient, AgentClientConfig, PromptType
from text_utils import create_query_prompt, get_context_from_file, get_context_from_file, extract_command


@dataclass
class K8sConfig:
    """
    Structured container for the app settings.
    """
    prompt_type: PromptType = PromptType.ZEROSHOT_BASE
    num_queries: int = 10
    output_dir: str = 'output'
    microservice_dir: str = 'microservices-demo'
    output_file: str = 'eval_results.jsonl'
    benchmark_path: str = 'error_config.jsonl'
    config_gen: bool = False
    max_iterations: int = 10
    agent_client_configs: list[AgentClientConfig] = field(default_factory=list)

    def __post_init__(self):
        # TODO: Limit to only one agent for now (maybe use K8s namespaces to allow parallel assessments on a single cluster).
        if len(self.agent_client_configs) != 1:
            raise ValueError(f'Must have exactly one agent client config, got {len(self.agent_client_configs)}')


# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="Qwen/Qwen2.5-72B-Instruct", help='Choose the LLM agent')#choices=["Qwen/Qwen2.5-72B-Instruct", "GPT-4o", "ReAct_Agent"]
    parser.add_argument('--num_queries', type=int, default=1, help='Number of queries to generate for each type')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/NetPress_benchmark/app-k8s/results", help='Directory to save output files.')
    parser.add_argument('--microservice_dir', type=str, default="/home/ubuntu/microservices-demo", help='Directory to google microservice demo')
    parser.add_argument('--max_iteration', type=int, default=10, help='Choose maximum trials for a query')
    parser.add_argument('--config_gen', type=int, default=1, help='Choose whether to generate new config')
    parser.add_argument('--benchmark_path', type=str, default="/home/ubuntu/NetPress_benchmark/app-k8s/results/error_config.json",
                         help='Where to save the generated benchmark (config), or where to find it if config_gen is 0.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for tensor parallelism (VLLM). Only applies to locally run models.')
    parser.add_argument('--prompt_type', type=str, default="base", choices=["few_shot_basic", "base", "cot"], help='Choose the prompt type')
    parser.add_argument('--agent_test', type=int, default=0, choices=[0, 1], help='Choose whether to run the agent test')
    return parser.parse_args()

# Deploy a Kubernetes cluster using Skaffold
def deploy_k8s_cluster(skaffold_config_path: str):
    """
    Deletes the existing kind cluster, creates a new one, and deploys an application using skaffold.
    :param skaffold_config_path: Path to the skaffold configuration directory.
    """
    try:
        print("Deleting existing kind cluster...")
        subprocess.run(["kind", "delete", "cluster"], check=True)
        
        print("Creating a new kind cluster...")
        subprocess.run(["kind", "create", "cluster"], check=True)
        
        print("Deploying application using skaffold...")
        subprocess.run(["skaffold", "run"], cwd=skaffold_config_path, check=True)
        
        print("Deployment completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

# Run the configuration error test
async def run_config_error(args):
    args = structure(vars(args), K8sConfig)

    starttime = datetime.now()

    policy_names = [
        "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice",
        "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend",
        "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice",
        "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice"
    ]
    pod_names = [
        "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend",
        "loadgenerator", "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart", "shippingservice"
    ]

    # Create the result directory. Timestamp directory already included when running the agent test.
    llm_config = args.agent_client_configs[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(args.output_dir, f'{llm_config.name}_{args.prompt_type}', timestamp)

    os.makedirs(result_dir, exist_ok=True)
    if args.config_gen:
        error_config = generate_config(args.benchmark_path, policy_names, args.num_queries)

    # Read the error configuration
    error_config_path = args.benchmark_path

    with open(error_config_path, 'r') as error_config_file:
        error_config = json.load(error_config_file)

    total_error_num = len(error_config["details"])
    logger.info(f"Total number of errors: {total_error_num}")

    # Create debug containers for all pods
    debug_container_mapping = {}
    for pod_name in pod_names:
        debug_container_name = await create_debug_container(pod_name)
        if debug_container_name:
            debug_container_mapping[pod_name] = debug_container_name
    logger.info(f"Debug container mapping: {debug_container_mapping}")
    endtime = datetime.now()
    logger.info(f"Startup time: {endtime - starttime}")

    # Iterate through the error configurations and run tests
    async with httpx.AsyncClient() as httpx_client:
        # Establish connections to the agents.
        clients = [AgentClient(agent_client_config, http_client=httpx_client) for agent_client_config in args.agent_client_configs]
        agents: list[AgentClient] = []
        for client in asyncio.as_completed([c.start() for c in clients]):
            try:
                agent = await client
                agents.append(agent)
            except Exception as e:
                logger.debug(f'Connection failed: {e}')
                logger.warning(f'Failed to connect to agent server. Skipping...')
        if not agents:
            raise ConnectionError('Could not connect to any agent servers. Aborting assessment.')
        
        llm = agents[0] # Only one agent supported for now.
        for i, error in enumerate(error_config["details"]):

            starttime = datetime.now()
            policies_to_inject = error.get("policies_to_inject", [])
            inject_error_num = error.get("inject_error_num", [])
            error_detail = error.get("error_detail", [])

            logger.info(f"Error {i+1}:")
            logger.info(f"  Policies to inject: {policies_to_inject}")
            logger.info(f"  Inject error numbers: {inject_error_num}")
            logger.info(f"  Error details: {error_detail}")

            copy_yaml_to_new_folder(args.microservice_dir, args.output_dir)
            error_detail_str = "+".join([detail["type"] for detail in error_detail])

            json_file_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.json")
            with open(json_file_path, 'w') as json_file:
                print(f"Created JSON file: {json_file_path}")

            log_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.log")
            with open(log_path, 'w'):
                logger.info(f"Created log file: {log_path}")

            inject_config_errors_into_policies(policy_names, args.output_dir, inject_error_num, policies_to_inject, error_detail)
            deploy_policies(policy_names, args.output_dir)

            output = "None"
            llm_command = "None"
            mismatch_summary = {}
            endtime = datetime.now()
            logger.info(f"Deployment time: {endtime - starttime}")

            for k in range(args.max_iterations):
                starttime = datetime.now()
                if k == 0:
                    pass
                else:
                    file_write(llm_command, output, mismatch_summary, json_file_path, log_path)
                logger.info(f"Running LLM iteration {k+1}...")

                # Use a while True loop to continuously attempt to get the LLM command
                attempt = 0
                while True:
                    attempt += 1
                    logger.info(f"Attempt {attempt}: Calling LLM...")
                    # Read the connectivity status from the debug container logs.
                    connectivity_status = get_context_from_file(log_path)
                    prompt = create_query_prompt(connectivity_status, args.prompt_type)
                    llm_output = await llm.handle_query(prompt)
                    if llm_output is None:
                        logger.error(f"Error while generating LLM command. Retrying...")
                        await asyncio.sleep(3)
                        continue
                    else:
                        llm_command = extract_command(llm_output)
                        logger.info(f"Generated LLM command: {llm_command}")
                        break

                endtime = datetime.now()
                logger.info(f"LLM generation time: {endtime - starttime}")
                if llm_command is None:
                    logger.error("Error: llm_command is None")
                    continue
                if "sudo" in llm_command:
                    logger.error("Error: LLM command contains 'sudo'")
                    continue
                if "kubectl apply" in llm_command:
                    logger.error("Error: LLM command contains 'kubectl apply -f'")
                    continue
                if "kubectl create" in llm_command:
                    logger.error("Error: LLM command contains 'kubectl create -f'")
                    continue                
                starttime = datetime.now()
                try:
                    output = subprocess.run(llm_command, shell=True, executable='/bin/bash', check=True, text=True, capture_output=True, timeout=10).stdout
                except subprocess.TimeoutExpired:
                    logger.error(f"Command timed out after 60 seconds")
                    output = "Command timed out"
                except subprocess.CalledProcessError as e:
                    print(f"Command failed:\n{e.stderr}")
                    output = e.stderr
                endtime = datetime.now()
                logger.info(f"LLM command execution time: {endtime - starttime}")

                starttime = datetime.now()
                all_match, mismatch_summary = await correctness_check(EXPECTED_RESULTS, debug_container_mapping)
                endtime = datetime.now()
                logger.info(f"Correctness check time: {endtime - starttime}")

                if all_match:
                    logger.info(f"Success in iteration {k+1}")
                    file_write(llm_command, output, mismatch_summary, json_file_path, log_path)
                    break

    summary_tests(result_dir)
    plot_metrics(result_dir)

# Run the agent test
async def run_agent_test(args):
    args.root_dir = os.path.join(args.root_dir, "result", args.llm_agent_type, "agent_test", datetime.now().strftime("%Y%m%d_%H%M%S"))
    for i in range(5):
        if i == 0:
            deploy_k8s_cluster(args.microservice_dir)
            args.prompt_type = "cot"
            args.config_gen = 1
            await run_config_error(args)
        elif i == 1:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "few_shot_basic"
            await run_config_error(args)
            end_time = datetime.now()
            print(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")
        elif i == 2:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "few_shot_basic"
            args.llm_agent_type = "GPT-4o"
            await run_config_error(args)
            end_time = datetime.now()
            print(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")
        elif i == 3:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "few_shot_basic"
            await run_config_error(args)
            end_time = datetime.now()
            print(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")
        elif i == 4:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "base"
            args.llm_agent_type = "ReAct_Agent"
            await run_config_error(args)
            end_time = datetime.now()
            print(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")

    policies_dir = os.path.join(args.root_dir, "policies")
    if os.path.exists(policies_dir):
        shutil.rmtree(policies_dir)


    plot_summary_results(args.root_dir, 10)
    plot_summary_results(args.root_dir, 50)
    plot_summary_results(args.root_dir, 150)


# Main entry point
if __name__ == "__main__":
    args = parse_args()
    if args.agent_test == 1:
        asyncio.run(run_agent_test(args))
    else:
        asyncio.run(run_config_error(args))
