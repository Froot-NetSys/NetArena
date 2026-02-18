# Route: Network Routing Benchmark

Evaluate LLM agents on network routing configuration tasks. The benchmark uses Mininet to simulate network topologies and tests if LLMs can diagnose and fix routing issues.

---

## Quick Start (Docker)

### Step 1: Start the Docker Container

```bash
cd /path/to/NetArena

# Start container with privileged mode (required for Mininet)
sudo docker run -itd --name netarena_route \
  --privileged \
  --net=host \
  -v $(pwd):/NetArena \
  netarena:latest

# Enter the container
sudo docker exec -it netarena_route /bin/bash
```

### Step 2: Start Open vSwitch (Required for Mininet)

```bash
# Inside the container
service openvswitch-switch start
```

### Step 3: Test Mininet (Optional but Recommended)

```bash
mn --test pingall
# Should see "All pings successful" or similar
```

### Step 4: Configure and Run

```bash
# Activate environment
conda activate mininet

# Setup config (first time only)
cd /NetArena/app-route
cp config.template.toml config.toml
nano config.toml  # Add your Azure API key

# Run the benchmark
python run.py
```

That's it! Results will be saved to `results/`.

---

## Configuration

Edit `config.toml` before running:

```toml
[model]
agent_type = "GPT-Agent"         # Which LLM to use
prompt_type = "base"             # Prompt strategy

[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "your-deployment-name"
api_key = "your-api-key"         # Required

[benchmark]
num_queries = 10                 # Queries per error type
max_iteration = 10               # Max fix attempts per query
regenerate = true                # Set true for first run
parallel = false                 # Run prompt types in parallel

[topology]
num_switches = 2                 # Network switches
num_hosts_per_subnet = 1         # Hosts per subnet

[output]
output_dir = "results"
```

### Model Options

| Agent Type | Description | Requirements |
|------------|-------------|--------------|
| `GPT-Agent` | Azure OpenAI GPT-4 | Azure API key |
| `Qwen/Qwen2.5-72B-Instruct` | Local Qwen via vLLM | GPU, HuggingFace token |
| `ReAct_Agent` | ReAct reasoning agent | Azure API key |

### Prompt Types

| Type | Description |
|------|-------------|
| `base` | Basic zero-shot prompt |
| `cot` | Chain-of-thought reasoning |
| `few_shot_basic` | Few-shot with fixed examples |

---

## Troubleshooting

### "Cannot find openvswitch" or Mininet errors
**Solution:** Start Open vSwitch first:
```bash
service openvswitch-switch start
```

### "mn: command not found"
**Solution:** Mininet not installed. Use the Docker image which has it pre-installed.

### "RTNETLINK answers: Operation not permitted"
**Solution:** Container needs `--privileged` flag:
```bash
sudo docker run --privileged ...
```

### "API key not found"
**Solution:** Make sure `api_key` is set in `config.toml` under `[model.azure]`

### "config.toml not found"
**Solution:** Copy the template first:
```bash
cp config.template.toml config.toml
```

### "error_config.json not found"
**Solution:** Set `regenerate = true` in config for first run to generate the benchmark file.

---

## Complete Docker Commands Reference

### First Time Setup
```bash
# Build the Docker image (from NetArena root)
cd /path/to/NetArena
docker build -t netarena:latest .

# Start container with required flags
sudo docker run -itd --name netarena_route \
  --privileged \
  --net=host \
  -v $(pwd):/NetArena \
  netarena:latest

# Enter container
sudo docker exec -it netarena_route /bin/bash
```

### Inside Container - Full Setup
```bash
# 1. Start Open vSwitch (required for Mininet)
service openvswitch-switch start

# 2. Test Mininet works
mn --test pingall

# 3. Activate conda environment
conda activate mininet

# 4. Setup config
cd /NetArena/app-route
cp config.template.toml config.toml
nano config.toml   # Add your API key

# 5. Run benchmark
python run.py
```

### Re-running Later
```bash
# If container exists but stopped
sudo docker start netarena_route
sudo docker exec -it netarena_route /bin/bash

# Inside container
service openvswitch-switch start
conda activate mininet
cd /NetArena/app-route
python run.py
```

### Cleanup
```bash
# Stop container (from host)
sudo docker stop netarena_route

# Remove container completely
sudo docker rm netarena_route
```

---

## Understanding the Benchmark

### What It Tests
The Route benchmark evaluates LLM ability to:
1. **Diagnose** network connectivity issues
2. **Generate** correct routing commands
3. **Fix** misconfigurations in simulated networks

### How It Works
1. Creates a Mininet network topology
2. Injects routing errors (wrong routes, missing entries, etc.)
3. Shows the LLM the network state and ping failures
4. Tests if the LLM can generate the fix command
5. Verifies the fix restores connectivity

### Error Types
- Missing routes
- Wrong next-hop addresses
- Incorrect subnet masks
- Duplicate routes
- And more...

### Output Files
Results are saved to `results/`:
- `*.txt` - Human-readable logs with commands and responses
- `*.json` - Structured results with timing data

---

## Configuration Reference

| Section | Key | Description |
|---------|-----|-------------|
| `[model]` | `agent_type` | LLM to use: `GPT-Agent`, `Qwen/...`, `ReAct_Agent` |
| `[model]` | `prompt_type` | Prompting strategy |
| `[model]` | `vllm` | Enable vLLM for local models (0/1) |
| `[model]` | `num_gpus` | GPUs for tensor parallelism |
| `[model.azure]` | `endpoint` | Azure OpenAI endpoint |
| `[model.azure]` | `deployment_name` | Deployment name |
| `[model.azure]` | `api_key` | API key |
| `[model.huggingface]` | `token` | HuggingFace token |
| `[benchmark]` | `num_queries` | Queries per error type |
| `[benchmark]` | `max_iteration` | Max fix attempts |
| `[benchmark]` | `regenerate` | Generate new benchmark |
| `[benchmark]` | `benchmark_path` | Benchmark config file |
| `[benchmark]` | `parallel` | Run in parallel |
| `[topology]` | `num_switches` | Number of switches |
| `[topology]` | `num_hosts_per_subnet` | Hosts per subnet |
| `[output]` | `output_dir` | Results directory |

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
    --llm_agent_type GPT-Agent \
    --num_queries 10 \
    --root_dir results \
    --max_iteration 10 \
    --prompt_type base \
    --parallel 0
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
Leave `api_key` empty and authenticate via Azure CLI:
```bash
az login
```

---

## Testing Your Own Model

Edit `llm_model.py` and look for `# ====== TODO:` comments:

1. **Add model name** in `LLMModel._create_model()`
2. **Implement initialization** in `YourModel.__init__()`
3. **Implement prediction** in `YourModel.predict()`
