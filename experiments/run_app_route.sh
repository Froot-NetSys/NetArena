#!/bin/bash 
cd ..  # Navigate to the root directory
cd app-route  # Enter the application directory

# Define common parameters
NUM_QUERIES=1
ROOT_DIR="/home/ubuntu/NetArena_benchmark/app-route/results"
BENCHMARK_PATH="${ROOT_DIR}/error_config.json"
MAX_ITERATION=10
FULL_TEST=1
STATICGEN=1
PROMPT_TYPE="base"
# Define the model and parallel mode parameters
MODEL="Qwen/Qwen2.5-72B-Instruct"  # Replace with the model you want to use
NUM_GPUS=4  # Number of GPUs to use for tensor parallelism. Only relevant for models running locally with VLLM.
PARALLEL=0  # Default to parallel execution. Set to 0 for single process.

# Create a log file with a timestamp to avoid overwriting
mkdir -p "$ROOT_DIR"
LOG_FILE="${ROOT_DIR}/experiment_$(date +'%Y-%m-%d_%H-%M-%S').log"

export HUGGINGFACE_TOKEN="[YOUR_TOKEN_HERE]"  # Set your Hugging Face token here

# Azure OpenAI configuration
export AZURE_OPENAI_ENDPOINT="https://your-gpt-4o-deployment.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-gpt-4o-deployment"
export AZURE_OPENAI_API_VERSION="YYYY-MM-DD" # Optional. Defaults to "2024-10-01".
export AZURE_OPENAI_API_KEY="API_KEY" # Not needed if Entra ID is used.

# Function to clean up existing controller processes
cleanup_controllers() {
    echo "Cleaning up existing controller processes..." | tee -a "$LOG_FILE"
    sudo killall controller 2>/dev/null
    sudo mn -c >/dev/null 2>&1
    sleep 2  # Give some time for processes to fully terminate
}

# Function to run the benchmark
run_benchmark() {
    echo "Running experiment with model: $MODEL and parallel mode: $PARALLEL..." | tee -a "$LOG_FILE"
    
    cleanup_controllers
    
    nohup sudo -E $(which python) main.py \
        --llm_agent_type "$MODEL" \
        --num_queries $NUM_QUERIES \
        --root_dir "$ROOT_DIR" \
        --max_iteration $MAX_ITERATION \
        --static_benchmark_generation $STATICGEN \
        --benchmark_path "$BENCHMARK_PATH" \
        --prompt_type "$PROMPT_TYPE" \
        --num_gpus $NUM_GPUS \
        --parallel "$PARALLEL" >> "$LOG_FILE" 2>&1 &
}


# Run the benchmark based on the specified parallel mode
run_benchmark


