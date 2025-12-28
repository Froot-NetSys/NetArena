#!/bin/bash

# Make sure your working directory is in the green_agent folder when running.

trap cleanup EXIT

cleanup() {
    echo "Cleaning up"
    kill $server_pid1 $server_pid2  # Terminates both server processes
    exit
}

# Choose model to serve along with exporting relevant secrets (see LiteLLM docs for more details on configuring env variables).
export AZURE_API_KEY="<API_KEY>"
export AZURE_API_BASE="<API_BASE_URL>"
export AZURE_API_VERSION="<API_VERSION>"

MODEL_NAME="azure/<DEPLOYMENT_NAME>"

uv run ./litellm_a2a_server.py \
    --model-name "${MODEL_NAME}" \
    --port 8000 &
server_pid1=$!  # Get the process ID of the last backgrounded command

sleep 3

# Serve the MALT evaluation agent.
uv run ./malt_agent.py &
server_pid2=$!  # Get the process ID of the last backgrounded command

sleep 3

# Run the client code to send a mock evaluation request. Modify scenario.toml to point to existing benchmark data (or regenerate new ones).
uv run ./client_cli.py ./scenario.toml ./output.json

cleanup()