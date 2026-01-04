from mininet.log import lg
from safety_check import safety_check
import argparse
import json
from datetime import datetime
import os
import subprocess
import time
from multiprocessing import Process
import shutil
from dataclasses import dataclass, field
from cattrs import structure
import asyncio
import httpx
from loguru import logger

from parallel_ping import parallelPing
from file_utils import process_results, plot_results, prepare_file, initialize_json_file, static_summarize_results, static_plot_metrics, write_query_result, write_log_content
from advanced_error_function import generate_config, process_single_error
from topology import initialize_network
from netarena.agent_client import PromptType, AgentClientConfig, AgentClient
from text_utils import create_query_prompt, get_context_from_file


@dataclass
class AppRouteConfig:
    num_queries: int = 1
    output_dir: str = "results"
    max_iterations: int = 10
    benchmark_path: str = 'error_config.json'
    regenerate_benchmark: bool = False
    prompt_type: PromptType = PromptType.ZEROSHOT_BASE
    num_switches: int = 2
    num_hosts_per_subnet: int = 1
    agent_client_configs: list[AgentClientConfig] = field(default_factory=list)

    def __post_init__(self):
        names = [config.name for config in self.agent_client_configs]
        if len(names) != len(set(names)):
            raise ValueError(f'Bad agent client configuration. Different agents cannot have the same name.')


async def evaluate_routing_queries(args: AppRouteConfig, result_dir: str | None = None):
    """
    Run a separate Mininet instance for each benchmark test.
    Assign a unique root directory for each instance.
    """
    # TODO: Using one agent for now due to synchronous nature of Mininet API.
    agent_config = args.agent_client_configs[0]

    start_time_2 = datetime.now()
    # Get the unique process ID to distinguish between different instances
    unique_id = os.getpid()
    # Expand ~ in root_dir to the actual home directory path
    args.output_dir = os.path.expanduser(args.output_dir)
    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(args.output_dir, f'{agent_config.name}_{agent_config.prompt_type}', timestamp)
    os.makedirs(result_dir, exist_ok=True)

    # Generate or load the error configuration file
    # If benchmark_path is relative, make it relative to root_dir
    file_path = args.benchmark_path
    
    # Generate config if requested OR if the file doesn't exist
    if args.regenerate_benchmark or not os.path.exists(file_path):
        generate_config(file_path, num_errors_per_type=args.num_queries, 
                       num_switches=args.num_switches, num_hosts_per_subnet=args.num_hosts_per_subnet)
        logger.info(f"Process {unique_id}: Generated error configuration file: {file_path}")
    logger.info(f"Process {unique_id}: Running benchmark with prompt type {args.prompt_type}")
    logger.info(f"Process {unique_id}: Using error configuration file: {file_path}")
    # Load the error configuration
    with open(file_path, 'r') as f:
        config = json.load(f)
    queries = config.get("queries", [])

    logger.info(f"Number of queries: {len(queries)}")

    # Initialize the LLM model
    async with httpx.AsyncClient() as httpx_client:
        # Establish connections to the agents.
        try:
            agent = await AgentClient(agent_config, http_client=httpx_client).start()
        except Exception as e:
            logger.debug(f'Connection failure reason: {e}')
            raise ConnectionError('Could not connect to any agent servers. Aborting assessment.')
        
        # Serialize agent info without any potentially sensitive info (e.g. HTTP headers).
        agent_config_dict = agent_config.serialize_omit_secrets()

        for i, query in enumerate(queries):
            start_time_1 = datetime.now()
            logger.info(f'Process {unique_id}: Injecting errors for query {i}')

            # Extract parameters from the query
            num_hosts_per_subnet = query.get("num_hosts_per_subnet", 1)
            num_switches = query.get("num_switches")
            errortype = query.get("errortype")
            errordetail = query.get("errordetail")
            errornumber = query.get("errornumber")

            logger.info(f"Process {unique_id}: Initializing Mininet instance")
            logger.info(f"  -> Topology: {num_switches} switches, {num_hosts_per_subnet} hosts/subnet")
            logger.info(f"  -> Error type: {errortype}")
            start_time = datetime.now()

            # Initialize the network
            subnets, topo, net, router = initialize_network(num_hosts_per_subnet, num_switches, unique_id)

            end_time = datetime.now()
            logger.info(f"Process {unique_id}: Network initialization took {end_time - start_time}")
            logger.info(f"Process {unique_id}: Subnets created: {[s[2] for s in subnets]}")

            # Inject errors into the network
            logger.info(f"Process {unique_id}: Injecting errors into network...")
            if errornumber == 1:
                logger.info(f"Process {unique_id}: Injecting single error: {errortype}")
                process_single_error(router, subnets, errortype, errordetail, unique_id)
                logger.info(f"Process {unique_id}: Error injected successfully")
            else:
                if isinstance(errortype, list) and isinstance(errordetail, list) and len(errortype) == errornumber and len(errordetail) == errornumber:
                    for et, ed in zip(errortype, errordetail):
                        logger.info(f"Process {unique_id}: Injecting error: {et}")
                        process_single_error(router, subnets, et, ed, unique_id)
                    logger.info(f"Process {unique_id}: All errors injected successfully")
                else:
                    logger.error(f"Process {unique_id}: Error: For multiple error injection, errortype and errordetail must be lists of length equal to errornumber")
                    continue
            # CLI(net)   
            if isinstance(errortype, list):
                errortype = '+'.join(errortype)  
            # Create result directory and files
            error_type_dir = os.path.join(result_dir, errortype)
            os.makedirs(error_type_dir, exist_ok=True)

            log_path = os.path.join(error_type_dir, f'result_{i+1}.txt')
            json_path = os.path.join(error_type_dir, f'result_{i+1}.json')

            prepare_file(log_path)
            initialize_json_file(json_path)

            # LLM interacts with Mininet
            logger.info(f"Process {unique_id}: Starting LLM interaction loop (max {args.max_iterations} iterations)")

            iter = 0
            success = False
            is_safe = True
            prev_packet_loss = 100.0
            while iter < args.max_iterations:
                # Ping all hosts in the network
                start_time = datetime.now()
                logger.info(f"Process {unique_id}: Iteration {iter} - Running pingAll test...")
                try:
                    pingall, loss_percent = parallelPing(net, timeout=0.1)
                    logger.info(f"Process {unique_id}: PingAll completed - Loss: {loss_percent}%")
                except Exception as e:
                    logger.error(f"Process {unique_id}: Error during pingAll: {e}")
                    logger.warning(f"Process {unique_id}: Skipping to next iteration due to pingAll error.")
                    continue
                    
                end_time = datetime.now()
                logger.info(f"Time taken for pingAll: {end_time - start_time}")

                pingall_logs = f"Pingall result:\n{pingall}\n"

                # Get LLM response
                attempt = 0
                while True:
                    attempt += 1
                    logger.info(f"Attempt {attempt}: Calling LLM...")
                    try:
                        # Process log content.
                        log_content = get_context_from_file(log_path) + pingall_logs
                        query = create_query_prompt(log_content, agent_config.prompt_type)

                        query_start_time = time.perf_counter()
                        llm_output = json.loads(await agent.handle_query(query))
                        query_end_time = time.perf_counter()

                        machine, commands = llm_output.get("machine"), llm_output.get("command")
                        logger.info(f"Generated LLM command ([machine] [command]): {machine} {commands}")

                        result = {
                            'packet_loss': loss_percent,
                            'elapsed_time': query_end_time - query_start_time,
                        }

                        new_entry = f'Machine: {machine}\nCommand: {commands}\nPacket Loss: {loss_percent}%\nElapsed Time: {query_end_time - query_start_time} seconds\n'

                        write_query_result(json_path, result)
                        write_log_content(log_path, pingall_logs + new_entry)
                        break
                    except Exception as e:
                        logger.error(f"Error while generating LLM command: {e}")
                        await asyncio.sleep(3)

                # Safety check for the generated command.
                if loss_percent > prev_packet_loss:
                    is_safe = False
                prev_packet_loss = loss_percent

                # Check log content, exit loop if successful
                if loss_percent == 0:
                    logger.info(f"Query {i}: Success in {iter} iterations")
                    success = True
                    break
                end_time = datetime.now()
                logger.info(f"Time taken for LLM response: {end_time - start_time}")
                iter += 1

                # Try executing the LLM command.
                lg.output(f"Machine: {machine}\n")
                lg.output(f'Iteration: {iter}\n')
                lg.output(f"Command: {commands}\n")
                
                if machine not in net:
                    lg.output(f"Error: Machine {machine} not found in the network topology.\n")
                    continue
                if commands is None or commands.strip() == "":
                    lg.output(f"Error: No command provided by LLM.\n")
                    continue
                if safety_check(commands):
                    try:
                        # Try executing the command
                        command_output = net[machine].cmd(commands)
                        print("LLM command executed successfully")

                    except TimeoutError as te:
                        lg.output(f"Timeout occurred while executing command on {machine}: {te}\n")
                    except Exception as e:
                        # Handle exceptions, log the error, and continue
                        lg.output(f"Error occurred while executing command on {machine}: {e}\n")

            # Stop the Mininet instance
            logger.info(f"Process {unique_id}: Stopping Mininet instance")
            net.stop()

            end_time_1 = datetime.now()
            logger.info(f"Process {unique_id}: Time taken for query {i}: {end_time_1 - start_time_1}")

            eval_result = {
                'agent_info': agent_config_dict,
                'error_type': errortype,
                'error_detail': errordetail,
                'success': success,
                'safe': is_safe,
                'iterations': iter + 1,
                'log_content': get_context_from_file(log_path, context_length=None)
            }
            yield eval_result

    logger.info(f"Process {unique_id}: Benchmark finished for {args.prompt_type}")
    logger.info(f"Process {unique_id}: Total time taken for all queries: {datetime.now() - start_time_2})")


async def run_benchmark(args: AppRouteConfig):
    """
    Run benchmark tests using a single Mininet instance.
    """
    start_time = datetime.now()

    # TODO: Using one agent for now due to synchronous nature of Mininet API.
    agent_config = args.agent_client_configs[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(args.output_dir, f'{agent_config.name}_{agent_config.prompt_type}', timestamp)

    # TODO: Ideally, code for producing eval artifacts and writing results should be separated from
    # the benchmark running code itself.
    async for eval_result in evaluate_routing_queries(args, result_dir):
        pass
    
    for subdir in os.listdir(result_dir):
        subdir_path = os.path.join(result_dir, subdir)
        if os.path.isdir(subdir_path):
            json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
            static_summarize_results(subdir_path, json_result_path)

    # Get the unique process ID to distinguish between different instances
    unique_id = os.getpid()
    static_plot_metrics(result_dir)
    end_time = datetime.now()
    logger.info(f"Process {unique_id}: Total time taken for all queries: {end_time - start_time}")


def run_benchmark_parallel(args):
    """
    Run static benchmark tests in parallel using multiple processes.

    Args:
        args (argparse.Namespace): The parsed arguments containing configuration.
    """
    # Clean up any existing Mininet resources
    subprocess.run(["sudo", "mn", "-c"], check=True)

    # Create a directory to save results
    save_result_path = os.path.join(args.root_dir, 'result', args.llm_agent_type, "agenttest", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_result_path, exist_ok=True)

    # Update the root directory in args
    args.root_dir = save_result_path
    args.llm_agent_type = "GPT-Agent"
    # Generate the error configuration file and update benchmark_path
    args.benchmark_path = os.path.join(save_result_path, "error_config.json")
    generate_config(args.benchmark_path, num_errors_per_type=args.num_queries,
                   num_switches=args.num_switches, num_hosts_per_subnet=args.num_hosts_per_subnet)

    # Define a wrapper function to run static benchmarks
    def run_static_benchmark(prompt_type, static_benchmark_generation,llm_agent_type):
        """
        Wrapper function to create an independent args instance per process.
        This ensures no conflicts between parallel processes.
        """
        args_copy = argparse.Namespace(**vars(args))  # Deep copy args to avoid conflicts
        args_copy.prompt_type = prompt_type
        args_copy.llm_agent_type = llm_agent_type
        args_copy.static_benchmark_generation = static_benchmark_generation
        evaluate_routing_queries(args_copy)

    # Get the list of prompt types from args (comma-separated)

    prompt_types = ["cot", "few_shot_basic"]

    # Create and start processes for each prompt type
    processes = []
    for prompt_type in prompt_types:
        process = Process(target=run_static_benchmark, args=(prompt_type, args.static_benchmark_generation, args.llm_agent_type))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    logs_path = os.path.join(save_result_path, "logs")
    if os.path.exists(logs_path):
        print(f"Deleting logs folder: {logs_path}")
        shutil.rmtree(logs_path)

    # Process the results and generate plots
    process_results(save_result_path)
    plot_results(save_result_path, args.num_queries)

    print(f"✅ Benchmark completed. Results saved to: {save_result_path}")