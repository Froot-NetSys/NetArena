# MALT: Datacenter Capacity Planning Benchmark

Evaluate LLM agents on datacenter topology management tasks. The benchmark tests if LLMs can correctly query, modify, and maintain datacenter graph structures while respecting operational constraints.

---

## Quick Start (Docker)

### Step 1: Start the Docker Container

```bash
cd /path/to/NetArena

# Start container with GPU support
sudo docker run -itd --name netarena_malt \
  --gpus all \
  -v $(pwd):/NetArena \
  netarena:latest

# Enter the container
sudo docker exec -it netarena_malt /bin/bash
```

### Step 2: Configure and Run

```bash
# Activate environment
conda activate ai_gym_env

# Setup config (first time only)
cd /NetArena/app-malt
cp config.template.toml config.toml
nano config.toml  # Add your Azure API key

# Run the benchmark
python run.py
```

That's it! Results will be saved to `logs/llm_agents/`.

---

## Configuration

Edit `config.toml` before running:

```toml
[model]
agent_type = "AzureGPT4Agent"    # Which LLM to use
prompt_type = "base"             # Prompt strategy

[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "your-deployment-name"
api_key = "your-api-key"         # Required

[benchmark]
num_queries = 10                 # Queries per error type
complexity_levels = ["level1", "level2"]  # Which levels to test
regenerate_queries = false       # Set true to generate new queries

[output]
output_dir = "logs/llm_agents"
output_file = "results.jsonl"
```

### Model Options

| Agent Type | Description | Requirements |
|------------|-------------|--------------|
| `AzureGPT4Agent` | Azure OpenAI GPT-4 | Azure API key |
| `GoogleGeminiAgent` | Google Gemini 1.5 Pro | `GOOGLE_API_KEY` env var |
| `Qwen2.5-72B-Instruct` | Local Qwen via vLLM | GPU, HuggingFace token |
| `QwenModel_finetuned` | Fine-tuned Qwen | Local model path |
| `ReAct_Agent` | ReAct reasoning agent | Azure API key |

### Prompt Types

| Type | Description |
|------|-------------|
| `base` | Basic zero-shot prompt |
| `cot` | Chain-of-thought reasoning |
| `few_shot_basic` | Few-shot with fixed examples |
| `few_shot_semantic` | Few-shot with semantic similarity |

### Complexity Levels

| Level | Description |
|-------|-------------|
| `level1` | Basic queries (4 types) - counting, listing |
| `level2` | Intermediate - adding/removing nodes |
| `level3` | Advanced - complex modifications |

---

## Troubleshooting

### "API key not found" or prompts for password
**Solution:** Make sure `api_key` is set in `config.toml` under `[model.azure]`

### "HUGGINGFACE_TOKEN not found"
**Solution:** For local models, set the token:
```bash
export HUGGINGFACE_TOKEN="your-token"
```
Or add it to `config.toml` under `[model.huggingface]`

### "config.toml not found"
**Solution:** Copy the template first:
```bash
cp config.template.toml config.toml
```

### CUDA out of memory
**Solution:** For local models, ensure you have enough GPU memory (72B model needs ~40GB). Use `AzureGPT4Agent` for smaller GPU setups.

---

## Complete Docker Commands Reference

### First Time Setup
```bash
# Build the Docker image (from NetArena root)
cd /path/to/NetArena
docker build -t netarena:latest .

# Start container with GPU
sudo docker run -itd --name netarena_malt \
  --gpus all \
  -v $(pwd):/NetArena \
  netarena:latest

# Enter container
sudo docker exec -it netarena_malt /bin/bash
```

### Inside Container - Full Setup
```bash
# 1. Activate conda environment
conda activate ai_gym_env

# 2. Setup config
cd /NetArena/app-malt
cp config.template.toml config.toml
nano config.toml   # Add your API key

# 3. Run benchmark
python run.py
```

### Re-running Later
```bash
# If container exists but stopped
sudo docker start netarena_malt
sudo docker exec -it netarena_malt /bin/bash

# Inside container
conda activate ai_gym_env
cd /NetArena/app-malt
python run.py
```

### Cleanup
```bash
# Stop container (from host)
sudo docker stop netarena_malt

# Remove container completely
sudo docker rm netarena_malt
```

---

## Understanding the Benchmark

### What It Tests
The MALT benchmark evaluates LLM ability to:
1. **Query** datacenter topology (count nodes, list children)
2. **Modify** topology (add/remove nodes, update values)
3. **Maintain safety** (respect operational constraints)

### Task Types
- **Counting queries** - "How many servers in rack X?"
- **Listing queries** - "List all child nodes of cluster Y"
- **Add operations** - "Add a new server to rack Z"
- **Remove operations** - "Remove server S from the topology"
- **Update operations** - "Change capacity of node N"
- **Ranking queries** - "Rank nodes by utilization"

### Output Files
Results are saved to `logs/llm_agents/`:
- `results.jsonl` - Detailed results per query
- `figs/` - Visualization plots:
  - `average_latency_*.png` - Response time by task
  - `correctness_pass_rate_*.png` - Accuracy by task
  - `safety_pass_rate_*.png` - Safety compliance by task

---

## Configuration Reference

| Section | Key | Description |
|---------|-----|-------------|
| `[model]` | `agent_type` | LLM to use |
| `[model]` | `prompt_type` | Prompting strategy |
| `[model]` | `model_path` | Path for fine-tuned models |
| `[model.azure]` | `endpoint` | Azure OpenAI endpoint |
| `[model.azure]` | `deployment_name` | Deployment name |
| `[model.azure]` | `api_key` | API key |
| `[model.azure]` | `api_version` | API version (default: 2024-12-01-preview) |
| `[model.google]` | `api_key` | Google API key for Gemini |
| `[model.huggingface]` | `token` | HuggingFace token for local models |
| `[benchmark]` | `num_queries` | Queries per type |
| `[benchmark]` | `complexity_levels` | Array of levels to test |
| `[benchmark]` | `benchmark_path` | Path to save/load queries |
| `[benchmark]` | `regenerate_queries` | Generate new queries (true/false) |
| `[benchmark]` | `start_index` | Start from query N (default: 0) |
| `[benchmark]` | `end_index` | End at query N (0 = all) |
| `[output]` | `output_dir` | Results directory |
| `[output]` | `output_file` | Results filename |

---

## Usage Examples

### Basic Run
```bash
python run.py
```

### Preview Configuration
```bash
python run.py --show-config
```

### Custom Config File
```bash
python run.py --config my_experiment.toml
```

### Legacy CLI (Alternative)
```bash
python main.py \
    --llm_agent_type AzureGPT4Agent \
    --num_queries 10 \
    --complexity_level level1 level2 \
    --output_dir logs/llm_agents \
    --output_file results.jsonl
```

---

## Authentication Options

### Azure OpenAI (Recommended)
```toml
[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "gpt-4"
api_key = "your-api-key"
```

### Google Gemini
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

### HuggingFace (Local Models)
```bash
export HUGGINGFACE_TOKEN="your-huggingface-token"
```
Or in config:
```toml
[model.huggingface]
token = "your-token"
```

### Azure Credential (No API Key)
Leave `api_key` empty and ensure you're authenticated via Azure CLI:
```bash
az login
```
