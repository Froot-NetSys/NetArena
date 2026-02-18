#!/bin/bash
cd ../app-k8s

# Parameter settings
LLM_AGENT_TYPE="GPT-4o"  # Type of LLM agent
NUM_QUERIES=1                              # Number of queries to generate
ROOT_DIR="/home/ubuntu/NetArena_benchmark/app-k8s/results"  # Root directory for output
BENCHMARK_PATH="${ROOT_DIR}/error_config.json"  # Path to save the benchmark (config) query data.
MICROSERVICE_DIR="/home/ubuntu/microservices-demo"  # Directory for microservice demo
MAX_ITERATION=10                           # Maximum number of iterations for a query
CONFIG_GEN=1                               # Whether to generate a new configuration (1 = yes, 0 = no)
PROMPT_TYPE="few_shot_basic"                         # Type of prompt to use
NUM_GPUS=4                                 # Number of GPUs to use for tensor parallel (VLLM). Should not be used with AGENT_TEST=1, which evaluates multiple agents asynchronously.
AGENT_TEST=0                               # Whether to test multiple agents (1 = yes, 0 = no)

# Replace special characters in LLM_AGENT_TYPE to make it a valid file name
SAFE_LLM_AGENT_TYPE=$(echo "$LLM_AGENT_TYPE" | tr '/' '_')

# Log file path
mkdir -p "$ROOT_DIR"
LOG_FILE="${ROOT_DIR}/${SAFE_LLM_AGENT_TYPE}_${PROMPT_TYPE}.log"

export HUGGINGFACE_TOKEN="[YOUR_TOKEN_HERE]"  # Set your Hugging Face token here

# Azure OpenAI configuration
export AZURE_OPENAI_ENDPOINT="https://your-gpt-4o-deployment.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-gpt-4o-deployment"
export AZURE_OPENAI_API_VERSION="YYYY-MM-DD" # Optional. Defaults to "2024-10-01".
export AZURE_OPENAI_API_KEY="API_KEY" # Not needed if Entra ID is used.

# Create the log file if it does not exist
touch "$LOG_FILE"

# Print the log file location
echo "Log saved to $LOG_FILE"

# Run the Python script and log the output
nohup $(which python) run_workflow.py \
    --llm_agent_type "$LLM_AGENT_TYPE" \
    --num_queries "$NUM_QUERIES" \
    --root_dir "$ROOT_DIR" \
    --benchmark_path "$BENCHMARK_PATH" \
    --microservice_dir "$MICROSERVICE_DIR" \
    --max_iteration "$MAX_ITERATION" \
    --config_gen "$CONFIG_GEN" \
    --prompt_type "$PROMPT_TYPE" \
    --num_gpus "$NUM_GPUS" \
    --agent_test "$AGENT_TEST" > "$LOG_FILE" 2>&1 &