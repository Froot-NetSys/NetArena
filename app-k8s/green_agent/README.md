# K8s Green Agent

An LLM-based agent that diagnoses and fixes Kubernetes network policy misconfigurations.

## Prerequisites

- Python 3.11+ with [uv](https://github.com/astral-sh/uv) installed
- `kubectl` installed and in PATH
- Access to an Azure OpenAI endpoint (or other LiteLLM-compatible LLM)
- A running Kubernetes cluster with the microservices-demo deployed

## Quick Start

1. **Clone the microservices-demo** (if not already present):
   ```bash
   git clone https://github.com/GoogleCloudPlatform/microservices-demo.git /path/to/microservices-demo
   ```

2. **Configure `scenario.toml`**:
   ```toml
   [config]
   microservice_dir = "/path/to/microservices-demo"  # Update this path
   ```

3. **Configure `green_agent_demo.sh`**:
   ```bash
   # Set your Azure OpenAI credentials
   export AZURE_API_KEY="your-api-key"
   export AZURE_API_BASE="https://your-endpoint.openai.azure.com/"
   export AZURE_API_VERSION="2024-12-01-preview"

   # Point to your kubeconfig
   export KUBECONFIG="/path/to/your/kubeconfig"

   # Set model name
   MODEL_NAME="azure/gpt-4.1"
   ```

4. **Run the agent**:
   ```bash
   cd app-k8s/green_agent
   ./green_agent_demo.sh
   ```

## What It Does

1. Starts a LiteLLM A2A server (port 8000) to proxy LLM requests
2. Starts the K8s evaluation agent (port 9999)
3. Runs evaluation scenarios from `test_data.json`
4. The agent iteratively:
   - Injects network policy errors into the cluster
   - Asks the LLM to diagnose connectivity issues
   - Executes kubectl commands suggested by the LLM
   - Validates if the fix restores correct connectivity

## Output

Results are saved to `output/` with:
- `*_result_N.json` - Structured command/output history
- `*_result_N.log` - Human-readable execution log

## Configuration Options

| File | Setting | Description |
|------|---------|-------------|
| `scenario.toml` | `num_queries` | Number of evaluation scenarios to run |
| `scenario.toml` | `max_iterations` | Max LLM attempts per scenario |
| `scenario.toml` | `prompt_type` | Prompt strategy (e.g., `zeroshot_base`) |
| `scenario.toml` | `benchmark_path` | Path to test scenarios JSON |

