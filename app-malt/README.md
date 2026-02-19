# Capacity Planning in Datacenter

This guide outlines steps to evaluate LLM agent performance on data center planning tasks. Agents are placed in a mock datacenter and tasked with making arbitrary, nonbreaking modifications to the topology while obeying operational constraints.

## Python Prerequisites

To set up the Python environment, we use `conda` to create a virtual environment. You can install the required dependencies by running the following commands:

```bash
conda env create -f environment_ai_gym.yml
conda activate ai_gym
```

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

## Run the application

You can either run the application directly from the command line, or you can modify the parameters in `run_app_malt.sh`, our preprovided run script.

Example usage.
```bash
# Running directly.
python main.py 
    --llm_agent_type AzureGPT4Agent \
    --num_queries 3 \
    --complexity_level level1 level2 \
    --output_dir logs/llm_agents \
    --output_file gpt4o.jsonl \
    --dynamic_benchmark_path data/benchmark_malt.jsonl
```

```bash
# Using our run script (with appropriate modifications).
cd experiments
bash run_app_malt.sh
```

### Explanation of Command Line Arguments

Below is a list of configurable parameters and the relevant details.

### `--llm_model_type`:
- **Description**: Specifies the type of LLM agent to be used in the benchmark. The format typically includes the name and version of the agent, such as `Qwen/Qwen2.5-72B-Instruct`. This determines which LLM model will be evaluated during the benchmarking process.
- **Possible Values**: `AzureGPT4Agent`, `GoogleGeminiAgent`, `Qwen2.5-72B-Instruct`, `QwenModel_finetuned`, `ReAct_Agent`
- **Example**: `Qwen2.5-72B-Instruct`

### `--prompt_type`:
- **Description**: Specifies the type of prompt to use when interacting with the LLM. The prompt type affects the nature of the queries sent to the LLM. You can choose between basic and more advanced prompts, depending on your test requirements.
- **Possible Values**: `base`, `cot`, `few_shot_basic`, `few_shot_semantic`
- **Example**: `base` (Use the basic prompt type)

### `--complexity_level`:
- **Description**: A space separated list that determines what level queries will be evaluated with each level having one or more different types of queries. For example, `level1` has 4 different types of queries. Possible values include: `level1`, `level2`, and `level3`.
- **Accepted Values**: Any combination of `level1`, `level2`, and `level3`
- **Example**: `level1 level3` (Only test with level 1 and level 3 queries)

### `--num_queries`:
- **Description**: Defines the number of queries to generate during the benchmarking process. This determines how many individual queries for each error type will be tested.
- **Example**: `10` (Test with 10 queries per type, so 40 queries total with `level1`)

### `--output_dir`:
- **Description**: The output directory where output figures, logs, etc will be stored. This path should point to the location on your machine where the benchmark results will be saved. Ensure that the specified directory exists and is accessible.
- **Example**: `/home/ubuntu/NetArena_benchmark/app-malt/data/results`

### `--output_file`:
- **Description**: The output file name (JSONL) containing the LLM response, and the correctness/safety information versus the ground truth. The output file will be saved under `--output_dir`.
- **Example**: `malt_100q_level1_eval.jsonl` (so output would be saved to `"${OUTPUT_DIR}/malt_100q_level1_eval.jsonl"`)

### `--regenerate_query`:
- **Description**:If specified, generates a new configuration based on the `--complexity_level` and `--num_queries` specified. Saves to `--dynamic_benchmark_path`.

### `--dynamic_benchmark_path`:
- **Description**: The path where the above benchmark containing the generated queries will be saved (JSONL). If `--regenerate_query` is not specified, then this parameter indicates where the existing benchmark file can be found.
- **Example**: `/home/ubuntu/NetArena_benchmark/app-malt/data/malt_bench_100q_level1.jsonl`

