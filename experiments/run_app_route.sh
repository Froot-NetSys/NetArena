#!/bin/bash 
cd "$(dirname "$0")/.."  # Navigate to the root directory
cd app-route  # Enter the application directory

# Define common parameters
NUM_QUERIES=1
ROOT_DIR="/NetArena/app-route/results"
BENCHMARK_PATH="${ROOT_DIR}/error_config.json"
MAX_ITERATION=10
FULL_TEST=1
STATICGEN=1
PROMPT_TYPE="base"
# Topology parameters
NUM_SWITCHES=2           # Number of switches (default: 2, range: 2-8)
NUM_HOSTS_PER_SUBNET=1   # Number of hosts per subnet (default: 1, range: 1-4)
# Define the model and parallel mode parameters
MODEL="GPT-Agent"  # Options: "Qwen/Qwen2.5-72B-Instruct", "GPT-Agent", "ReAct_Agent"
NUM_GPUS=1  # Number of GPUs to use for tensor parallelism. Only relevant for models running locally with VLLM.
PARALLEL=0  # Default to parallel execution. Set to 0 for single process.

# Create a log file with a timestamp to avoid overwriting
mkdir -p "$ROOT_DIR"
LOG_FILE="${ROOT_DIR}/experiment_$(date +'%Y-%m-%d_%H-%M-%S').log"

# Huggingface token (only needed for local LLMs like Qwen)
# export HUGGINGFACE_TOKEN="your_token_here"

# Azure OpenAI configuration is now loaded from config.json
# No need to export environment variables if using config.json

# Function to run command with or without sudo (skip sudo if already root)
run_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

# Function to clean up existing controller processes
cleanup_controllers() {
    echo "Cleaning up existing controller processes..." | tee -a "$LOG_FILE"
    run_cmd killall controller 2>/dev/null
    run_cmd mn -c >/dev/null 2>&1
    sleep 2  # Give some time for processes to fully terminate
}

# Function to run the benchmark
run_benchmark() {
    echo "Running experiment with model: $MODEL and parallel mode: $PARALLEL..." | tee -a "$LOG_FILE"
    
    cleanup_controllers
    
    # Run in foreground to see output (remove 'nohup' and '&' for interactive use)
    python main.py \
        --llm_agent_type "$MODEL" \
        --num_queries $NUM_QUERIES \
        --root_dir "$ROOT_DIR" \
        --max_iteration $MAX_ITERATION \
        --static_benchmark_generation $STATICGEN \
        --benchmark_path "$BENCHMARK_PATH" \
        --prompt_type "$PROMPT_TYPE" \
        --num_gpus $NUM_GPUS \
        --parallel "$PARALLEL" \
        --num_switches $NUM_SWITCHES \
        --num_hosts_per_subnet $NUM_HOSTS_PER_SUBNET 2>&1 | tee "$LOG_FILE"
}


# Run the benchmark based on the specified parallel mode
run_benchmark


