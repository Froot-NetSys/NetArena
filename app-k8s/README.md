# K8s: Kubernetes Network Policy Benchmark

Evaluate LLM agents on debugging Kubernetes network policy misconfigurations. The benchmark injects errors into network policies and tests if the LLM can identify and fix them.

---

## Quick Start (Docker)

### Step 1: Start the Docker Container

```bash
cd /path/to/NetArena

# Start container with required mounts
sudo docker run -itd --name netarena_k8s \
  --privileged \
  --network host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/NetArena \
  netarena:latest

# Enter the container
sudo docker exec -it netarena_k8s /bin/bash
```

### Step 2: Create Kubernetes Cluster (Inside Container)

```bash
# Create a KIND (Kubernetes in Docker) cluster
kind create cluster

# Verify the cluster is running
kubectl get nodes
# Wait until one node show with STATUS "Ready"
```

### Step 3: Deploy Microservices (Takes ~20 minutes)

```bash
# Deploy Google's Online Boutique demo
cd /microservices-demo
skaffold run

# Wait for all pods to be ready (this takes time)
kubectl get pods -w
# Wait until all pods show STATUS "Running" and READY "1/1"
# Press Ctrl+C to exit the watch
```

### Step 4: Configure and Run Benchmark

```bash
# Setup config (first time only)
conda activate mininet
cd /NetArena/app-k8s
cp config.template.toml config.toml
nano config.toml  # Add your Azure API key

# Run the benchmark
python run.py
```

---

## Configuration

Edit `config.toml` before running:

```toml
[model]
agent_type = "GPT-4o"            # Options: "GPT-4o", "Qwen/Qwen2.5-72B-Instruct", "ReAct_Agent"
prompt_type = "base"             # Options: "base", "cot", "few_shot_basic"

[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "your-deployment-name"
api_key = "your-api-key"         # Required for GPT-4o

[benchmark]
num_queries = 10                 # Queries per error type (15 types = 150 total)
max_iteration = 10               # Max fix attempts per query
regenerate = true                # Set true for first run, false to reuse

[paths]
output_dir = "results"
microservice_dir = "/microservices-demo"
```

---

## Troubleshooting

### "connection refused" errors
```
dial tcp 127.0.0.1:8080: connect: connection refused
```
**Solution:** KIND cluster not running. Run `kind create cluster`.

### "No such file: /microservices-demo"
**Solution:** The microservices-demo wasn't cloned. Run:
```bash
cd /
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
```

### Pods not starting / ImagePullBackOff
**Solution:** Wait longer, or check Docker has enough resources. The first `skaffold run` builds all images locally which takes time.

### "Cannot connect to Docker daemon"
**Solution:** Make sure you started the container with `-v /var/run/docker.sock:/var/run/docker.sock`

---

## Complete Docker Commands Reference

### First Time Setup
```bash
# Build the Docker image (from NetArena root)
cd /path/to/NetArena
docker build -t netarena:latest .

# Start container
sudo docker run -itd --name netarena_k8s \
  --privileged \
  --network host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/NetArena \
  netarena:latest

# Enter container
sudo docker exec -it netarena_k8s /bin/bash
```

### Inside Container - Full Setup
```bash
# 1. Activate conda environment
conda activate mininet

# 2. Create K8s cluster
kind create cluster
kubectl get nodes  # Verify

# 3. Deploy microservices (~20 min)
cd /microservices-demo
skaffold run
kubectl get pods   # Wait for all "Running"

# 4. Run benchmark
cd /NetArena/app-k8s
cp config.template.toml config.toml
nano config.toml   # Add your API key
python run.py
```

### Re-running Later
```bash
# If container exists but stopped
sudo docker start netarena_k8s
sudo docker exec -it netarena_k8s /bin/bash

# Inside container - cluster should still exist
conda activate mininet
kubectl get pods   # Check if pods still running
cd /NetArena/app-k8s
python run.py
```

### Cleanup
```bash
# Delete KIND cluster (inside container)
kind delete cluster

# Stop container (from host)
sudo docker stop netarena_k8s

# Remove container completely
sudo docker rm netarena_k8s
```

---

## Understanding the Benchmark

### What It Tests
The benchmark:
1. Deploys correct network policies to the microservices
2. Injects errors (e.g., blocks required traffic)
3. Gives the LLM the current state and error symptoms
4. Tests if the LLM can generate the correct fix command

### Error Types (15 total)
- `remove_ingress` - Removes allowed ingress rules
- `remove_egress` - Removes allowed egress rules
- `wrong_port` - Changes port numbers
- `wrong_protocol` - Changes TCP/UDP
- And more...

### Output
Results are saved to `results/` directory:
- `*_result_*.json` - Detailed results per query
- `*_result_*.txt` - Human-readable logs
- Summary plots showing pass rates

---

## Configuration Reference

| Section | Key | Description |
|---------|-----|-------------|
| `[model]` | `agent_type` | LLM to use: `GPT-4o`, `Qwen/Qwen2.5-72B-Instruct`, `ReAct_Agent` |
| `[model]` | `prompt_type` | Prompt strategy: `base`, `cot`, `few_shot_basic` |
| `[model]` | `num_gpus` | GPUs for local models (default: 1) |
| `[model.azure]` | `endpoint` | Azure OpenAI endpoint URL |
| `[model.azure]` | `deployment_name` | Your deployment name |
| `[model.azure]` | `api_key` | API key (leave empty for Azure credential) |
| `[benchmark]` | `num_queries` | Queries per error type |
| `[benchmark]` | `max_iteration` | Max fix attempts |
| `[benchmark]` | `regenerate` | Generate new benchmark (true/false) |
| `[paths]` | `output_dir` | Where to save results |
| `[paths]` | `microservice_dir` | Path to microservices-demo |

---

## Testing Your Own Model

Edit `llm_agent.py` and look for `# ====== TODO:` comments:

1. **Add your model name** in `LLMAgent.__init__`
2. **Implement loading** in `YourModel.__init__`
3. **Implement inference** in `YourModel.call_agent`

---

## Local Installation (Without Docker)

If you prefer running locally instead of Docker:

### Prerequisites
- Ubuntu 22.04
- Docker
- Go 1.19+
- NVIDIA GPU (for local LLMs)

### Install KIND
```bash
sudo apt install golang-go
go install sigs.k8s.io/kind@v0.26.0
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
echo 'export PATH=$PATH:$HOME/go/bin' >> ~/.bashrc
```

### Install kubectl
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

### Install Skaffold
```bash
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/v2.15.0/skaffold-linux-amd64
chmod +x skaffold
sudo mv skaffold /usr/local/bin/
```

### Clone Microservices Demo
```bash
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
# Fix platform issue
sed -i 's/$BUILDPLATFORM/linux\/amd64/g' microservices-demo/src/loadgenerator/Dockerfile
```

### Python Environment
```bash
conda env create -f environment_mininet.yml
conda activate mininet
```

Then follow Steps 2-4 from the Quick Start above.
