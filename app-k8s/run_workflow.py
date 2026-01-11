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
from inject_errors import inject_config_errors_into_policies, fetch_error_config
from correctness_check import correctness_check, create_debug_container, EXPECTED_RESULTS
from correct_policy import copy_yaml_to_new_folder
from deploy_policies import deploy_policies, POLICY_NAMES, POD_NAMES
from netarena.agent_client import AgentClient, AgentClientConfig, PromptType
from text_utils import create_query_prompt, get_context_from_file, get_context_from_file, extract_command, check_disallowed_commands


MAX_RETRIES = 5


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
    regenerate_config: bool = False
    max_iterations: int = 10
    agent_client_configs: list[AgentClientConfig] = field(default_factory=list)

    def __post_init__(self):
        # TODO: Limit to only one agent for now (maybe use K8s namespaces to allow parallel assessments on a single cluster).
        if len(self.agent_client_configs) > 1:
            raise ValueError(f'Must have exactly one agent client config, got {len(self.agent_client_configs)}')


# Deploy a Kubernetes cluster using Skaffold
def deploy_k8s_cluster(microservice_dir: str):
    """
    Deletes the existing K8s deployment and creates a new one.
    :param microservice_dir: Path to the microservice directory.
    """
    try:
        # TODO: Namespace support for multiple parallel assessments?
        try:
            logger.info("Tearing down previous deployments...")
            subprocess.run(["kubectl", "delete", "-f", "./release/kubernetes-manifests.yaml"], cwd=microservice_dir, check=True)
        except subprocess.CalledProcessError:
            logger.info("No previous deployments found or error during deletion. Continuing with deployment...")

        logger.info("Deploying application...")
        subprocess.run(["kubectl", "apply", "-f", "./release/kubernetes-manifests.yaml"], cwd=microservice_dir, check=True)
        subprocess.run(["kubectl", "wait", "--for=condition=ready", "deployment", "--all", "--timeout=120s"], cwd=microservice_dir, check=True)
        
        logger.info("Deployment completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred: {e}")

# Run the configuration error test
async def run_error_config(args: K8sConfig, result_dir: str | None = None):
    starttime = datetime.now()

    # Create the result directory. Timestamp directory already included when running the agent test.
    llm_config_dict = args.agent_client_configs[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if result_dir is None:
        result_dir = os.path.join(args.output_dir, f'{llm_config_dict.name}_{args.prompt_type}', timestamp)
        os.makedirs(result_dir, exist_ok=True)

    # Generate the error configuration if needed.
    error_config = fetch_error_config(args.benchmark_path, args.num_queries, args.regenerate_config)

    total_error_num = len(error_config)
    logger.info(f"Total number of errors: {total_error_num}")

    # Create debug containers for all pods
    debug_container_mapping = {}
    for pod_name in POD_NAMES:
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
        llm_config_dict = llm.config.serialize_omit_secrets()
        for i, error in enumerate(error_config):

            starttime = datetime.now()
            policies_to_inject = error.get("policies_to_inject", [])
            inject_error_num = error.get("inject_error_num", [])
            error_detail = error.get("error_detail", [])

            logger.info(f"Error {i + 1}:")
            logger.info(f"  Policies to inject: {policies_to_inject}")
            logger.info(f"  Inject error numbers: {inject_error_num}")
            logger.info(f"  Error details: {error_detail}")

            copy_yaml_to_new_folder(args.microservice_dir, args.output_dir)
            error_detail_str = "+".join([detail["type"] for detail in error_detail])

            json_file_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.json")
            with open(json_file_path, 'w'):
                logger.info(f"Created JSON file: {json_file_path}")

            log_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.log")
            with open(log_path, 'w'):
                logger.info(f"Created log file: {log_path}")

            inject_config_errors_into_policies(POLICY_NAMES, args.output_dir, inject_error_num, policies_to_inject, error_detail)
            deploy_policies(POLICY_NAMES, args.output_dir)

            output = "None"
            llm_command = "None"
            mismatch_summary = {}
            endtime = datetime.now()
            logger.info(f"Deployment time: {endtime - starttime}")

            k = 0
            results = []
            all_match = False
            is_safe = True
            prev_mismatch_count = float('inf')
            for k in range(args.max_iterations):
                starttime = datetime.now()
                logger.info(f"Running LLM iteration {k + 1}...")

                # Use a while True loop to continuously attempt to get the LLM command
                attempt = 0
                while attempt < MAX_RETRIES:
                    attempt += 1
                    logger.info(f"Attempt {attempt}: Calling LLM...")
                    # Read the connectivity status from the debug container logs.
                    connectivity_status = get_context_from_file(log_path)
                    prompt = create_query_prompt(connectivity_status, args.prompt_type)
                    llm_output = await llm.handle_query(prompt)
                    logger.debug(f"Raw LLM output: {llm_output}")
                    if llm_output is None:
                        logger.warning(f"Error while generating LLM command. Retrying...")
                        await asyncio.sleep(3)
                        continue
                    else:
                        llm_command = extract_command(llm_output)
                        logger.info(f"Generated LLM command: {llm_command}")
                        break
                else:
                    logger.error(f"Failed to get a valid LLM command after {MAX_RETRIES} attempts (this may indicate an issue with the LLM endpoint). Skipping this iteration.")
                    llm_command = ""
                    output = "LLM command generation failed."
                    await asyncio.sleep(MAX_RETRIES * 3)  # Backoff before next iteration
                    continue 

                endtime = datetime.now()
                logger.info(f"LLM generation time: {endtime - starttime}")
                if check_disallowed_commands(llm_command):
                    logger.error(f"Disallowed keyword detected in LLM output. Skipping execution.")
                    output = "Disallowed command detected."
                    continue               
                starttime = datetime.now()
                try:
                    output = subprocess.run(llm_command, shell=True, executable='/bin/bash', check=True, text=True, capture_output=True, timeout=10).stdout
                except subprocess.TimeoutExpired:
                    logger.error(f"Command timed out after 60 seconds")
                    output = "Command timed out"
                except subprocess.CalledProcessError as e:
                    logger.error(f"Command failed:\n{e.stderr}")
                    output = e.stderr
                endtime = datetime.now()
                logger.info(f"LLM command execution time: {endtime - starttime}")

                starttime = datetime.now()
                all_match, mismatch_summary = await correctness_check(debug_container_mapping, EXPECTED_RESULTS)
                endtime = datetime.now()
                logger.info(f"Correctness check time: {endtime - starttime}")

                # If the number of mismatches has increased, mark as unsafe.
                mismatch_count = mismatch_summary.count("Mismatch")
                if mismatch_count > prev_mismatch_count:
                    is_safe = False
                prev_mismatch_count = mismatch_count

                data = file_write(llm_command, output, mismatch_summary, json_file_path, log_path)
                results.append(data)
                if all_match:
                    logger.info(f"Success in iteration {k + 1}")
                    break

            # Format the result for this error configuration.
            res = {
                "agent_info": llm_config_dict,
                "error_detail": error_detail,
                "iterations": k + 1,
                "success": all_match,
                "safe": is_safe,
                "results": results
            }
            yield res


async def run_evaluation(args: K8sConfig):
    llm_config_dict = args.agent_client_configs[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(args.output_dir, f'{llm_config_dict.name}_{args.prompt_type}', timestamp)
    os.makedirs(result_dir, exist_ok=True)

    deploy_k8s_cluster(args.microservice_dir)

    # TODO: Ideally, this should only produce result artifacts (and not write anything to disk). Ignore the results, just run the evaluation.
    async for _ in run_error_config(args, result_dir=result_dir):
        pass

    summary_tests(result_dir)
    plot_metrics(result_dir)


