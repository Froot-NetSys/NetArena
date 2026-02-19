# Routing Application Setup

This app serves to evaluate the capabilities of different LLM agents on dynamically generated routing configuration tasks in a simulated network. The following guide outlines necessary configuration steps to get started.

## Python Prerequisites

To set up the Python environment, we use `conda` to create a virtual environment. You can install the required dependencies by running the following commands:

```bash
conda env create -f environment_mininet.yml
conda activate mininet
```

## Install Mininet Emulator Environment
To install the Mininet emulator, run the following command (we tested on Ubuntu 22.04):

```
chmod +x install_mininet.sh
./install_mininet.sh
```

If you see `Enjoy Mininet`, you install mininet environment successfully!

## Authentication
To run open source LLMs like `Qwen2.5-72B-Instruct` locally you will need to authenticate to Huggingface with your Huggingface access token like so:
```bash
export HUGGINGFACE_TOKEN="your_huggingface_token"
```
Information on how to get an acess token can be found on [here](https://huggingface.co/docs/hub/en/security-tokens).

### Azure GPT Usage
If you want to use Azure GPT on a Azure VM, you will need to create a `GPT-4o` deployment on Azure AI. If you haven't done so already, you can follow the instructions at the Azure GPT [quickstart](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=keyless%2Ctypescript-keyless%2Cpython-new%2Cbash&pivots=programming-language-python). Once you have a working deployment, you can export the following information:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-gpt-4o-deployment.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-gpt-4o-deployment"
export AZURE_OPENAI_API_VERSION="YYYY-MM-DD" # Optional. Defaults to "2024-10-01".
export AZURE_OPENAI_API_KEY="API_KEY" # Not needed if Entra ID is used.
```
Instead of explicitly specifying an API key, you can authenticate with Entra ID, which is done via `DefaultAzureCredential`. Using the [Azure CLI](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity) with appropriate role assignment or creating a [managed identity](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/how-to-configure-managed-identities?pivots=qs-configure-portal-windows-vm) on the host VM (with similar role assignment) are among a few ways to achieve this. For more information on authentication methods/details refer to the Azure docs (see `DefaultAzureCredential`).

## Running Benchmark Tests

The quickest way to start is to use our already provided experiment script.

### 1. Navigate to experiments
To run the benchmark tests, you can use our `run_app_route.sh` in `experiments`:
```
cd experiments
```
### 2. Modify Parameters in `run_app_route.sh`

Before running the benchmarking script, you need to modify the parameters in `run_app_route.sh` to suit your setup.  Below is a list of configurable parameters and their explanations.

### `MODEL`:
- **Description**: Specifies the type of LLM agent to be used in the benchmark. The format typically includes the name and version of the agent, such as `Qwen/Qwen2.5-72B-Instruct`. This determines which LLM model will be evaluated during the benchmarking process.
- **Possible Values**: `GPT-Agent`, `Qwen/Qwen2.5-72B-Instruct`, `ReAct_Agent`, `YourModel`
- **Example**: `Qwen/Qwen2.5-72B-Instruct`

### `NUM_QUERIES`:
- **Description**: Defines the number of queries to generate during the benchmarking process. This determines how many individual queries for each error type will be tested.
- **Example**: `10` (Test with 10 queries per type)

### `ROOT_DIR`:
- **Description**: The root directory where output logs and results will be stored. This path should point to the location on your machine where the benchmark results will be saved. Ensure that the specified directory exists and is accessible.
- **Example**: `/home/ubuntu/NetArena_benchmark/app-route/results`

### `MAX_ITERATION`:
- **Description**: The maximum number of iterations to run for each query. This helps control the number of times the agent will execute the query in each benchmark run. 
- **Example**: `10` (Run each query up to 10 iterations)

### `STATIC_GEN`:
- **Description**: This parameter controls whether a new configuration should be generated for each benchmark. Set it to `1` to generate a new configuration, or `0` to skip this step and use the existing configuration.
- **Example**: `1` (Generate new configuration)

### `BENCHMARK_PATH`:
- **Description**: The path where the above config (benchmark) containing the generated queries will be saved (JSON). If `STATIC_GEN=0`, then this parameter indicates where the existing config file can be found.
- **Example**: `/home/ubuntu/NetArena_benchmark/app-route/results/error_config.json`

### `PROMPT_TYPE`:
- **Description**: Specifies the type of prompt to use when interacting with the LLM. The prompt type affects the nature of the queries sent to the LLM. You can choose between basic and more advanced prompts, depending on your test requirements.
- **Possible Values**: `base`, `cot`, `few_shot_basic`
- **Example**: `base` (Use the basic prompt type)

### `NUM_GPUS`:
- **Description**: The number of GPUs to use with VLLM for tensor parallelism. This parameter only applies to locally run models like `Qwen2.5-72B-Instruct`, and should not used with `AGENT_TEST=1`. Note that the `NUM_GPUS` should evenly divide the number of attention heads of the model to be run. By default, only the first GPU is used, with the rest spilling over to RAM/disk.
- **Example**: `4`

### `PARALLEL`:
- **Description**: If set to 1 (0 otherwise), runs evaluation on multiple prompt types in parallel. Should not be used with `NUM_GPUS` values greater than 1 when running local models. If you want to run parallel evaluation with multiple GPUs locally you are better off setting `CUDA_VISIBLE_DEVICES` and running each experiment manually to distribute GPUs properly. 
- **Example**: `0`

### 3. Run the Benchmark
After modifying the parameters, you can execute the benchmarking process by running the script with the following command:
```bash
bash run_app_route.sh
```
## Testing Your Own Model

To integrate and test your own model in this benchmark framework, you need to make changes in three places:

1. **Model Name and Initialization**
   - In the `llm_model.py` file, locate the `LLMModel` class.
   - Update the `_create_model` and `_initialize_YourModel` methods to use your model's name and initialization logic.
   - **TODO comments** are provided in the code to indicate where you should make these changes.

2. **Model Loading**
   - In the `llm_model.py` file, locate the `YourModel` class.
   - Update the `_load_model` method to correctly load your own model and tokenizer.
   - Look for the **TODO comments** in the code to guide your modifications.

3. **Model Prediction/Inference**
   - In the `llm_model.py` file, within the `YourModel` class, find the `predict` method.
   - Modify this method to generate results using your LLM based on the provided prompt.
   - Again, follow the **TODO comments** for where to insert your logic.

Once you have completed these three steps, your model will be integrated into the benchmark environment. You can then proceed with testing by following the instructions in the **Running Benchmark Tests** section above.

---

**Tip:**  
All locations that require your changes are clearly marked with `# ====== TODO:` comments in the code for your convenience.
