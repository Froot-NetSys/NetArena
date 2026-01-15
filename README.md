# NetArena: Dynamically Generated LLM Benchmarks for Network Applications

## Overview
NetArena is a dynamic benchmark generation framework for evaluating LLM agents in real-world network applications. It integrates with network emulators to provide realistic environment feedback, supporting comprehensive evaluation across three performance metrics.

## Paper
The research behind NetArena is detailed in our paper:  
Zhou, Y., Ruan, J., Wang, E. S., Fouladi, S., Yan, F. Y., Hsieh, K., & Liu, Z. (2025). 
NetPress: Dynamically Generated LLM Benchmarks for Network Applications. *arXiv preprint arXiv:2506.03231*. [[paper]](https://arxiv.org/abs/2506.03231)

```bibtex
@article{zhou2025netpress,
  title={NetPress: Dynamically Generated LLM Benchmarks for Network Applications},
  author={Zhou, Yajie and Ruan, Jiajun and Wang, Eric S and Fouladi, Sadjad and Yan, Francis Y and Hsieh, Kevin and Liu, Zaoxing},
  journal={arXiv preprint arXiv:2506.03231},
  year={2025}
}
```

# 🐳 Assessing with Docker

Each benchmark app ships as an A2A agent following the [Agentbeats green agent format](https://agentbeats.dev/).

---

### 📋 How It Works

1. **Benchmark apps** receive evaluation requests via the A2A protocol
2. **Requests describe** the A2A agents involved (endpoints, API keys, etc.) and benchmark configuration (query difficulty, number of queries, etc.)
3. **This initiates** a round of evaluation against your agent

---

### 🟣 Build Your Own Purple Agent

Want to build your own agent? Check out our example prompt-based agents for reference:

| App | Prompt Agent |
|-----|--------------|
| MALT | [`app-malt/old_code/prompt_agent.py`](./app-malt/old_code/prompt_agent.py) |
| Route | [`app-route/old_code/prompt_agent.py`](./app-route/old_code/prompt_agent.py) |
| K8s | [`app-k8s/old_code/prompt_agent.py`](./app-k8s/old_code/prompt_agent.py) |

---

### 🏆 Submit Your Results

Ready to see how your agent stacks up? Visit the **[NetArena Leaderboard](https://github.com/Froot-NetSys/netarena_leaderboard/tree/main)** for submission instructions and to compare against other agents.

---

The following section includes instructions on how to build the container for each application.

## Prerequisites

- Linux-based system (tested on Ubuntu 20.04)
- At least 4 CPU cores
- Docker installed and running

## Quick Start

### 1. Build the Container

From the NetPress root directory, build the container for your target app:

```bash
cd ~/NetPress

# For MALT app
docker build -t malt_agent:latest -f ./app-malt/green_agent/Dockerfile .

# For K8s app
docker build -t k8s_agent:latest -f ./app-k8s/green_agent/Dockerfile .

# For Route app
docker build -t route_agent:latest -f ./app-route/green_agent/Dockerfile .
```

**(Optional)** For testing purposes, you may also build the baseline purple agent we use in this demo as a Docker container:

```bash
cd ~/NetPress

# Test purple agent.
docker build -t litellm_agent:latest -f ./a2a_llm/Dockerfile .
```

### 2. Start the Container

See [How to Start the Container for Each App](#how-to-start-the-container-for-each-app) below for app-specific commands.

If you wish to also run the baseline purple agent, the process is similar to the ones above. Just remember to pass in relevant environment variables

```bash
docker run -itd --network=host --env-file "./env.list" --name purple_agent litellm_agent:latest --host "0.0.0.0" --port 8000
```

Example `./env.list` file:
```bash
AZURE_API_KEY="<YOUR_API_KEY>"
AZURE_API_BASE="<YOUR_API_ENDPOINT>"
AZURE_API_VERSION="<YOUR_API_VERSION>"
MODEL_NAME="azure/XXX"
```

   For details on how to configure environment variables for a certain model provider, see the [LiteLLM documenation](https://docs.litellm.ai/docs/#basic-usage).


---

## How to Start the Container for Each App

> **⚠️ Important:** When switching between apps, always remove the existing `green_agent` container first with `docker rm -f green_agent`. All apps share the same container name.

### MALT App

Standard setup - no additional flags required:

```bash
# Remove any existing green_agent container first
docker rm -f green_agent

docker run -itd --network=host --name green_agent malt_agent:latest --host "0.0.0.0" --port 9999
```

### Route App

Requires `--privileged` flag for Mininet to function:

```bash
# Remove any existing green_agent container first
docker rm -f green_agent

docker run -itd --network=host --privileged --name green_agent route_agent:latest --host "0.0.0.0" --port 9999
```

Sometimes, you may also have to include the following flag to mount kernel modules for Open vSwitch:

```bash
-v /lib/modules:/lib/modules
```

### K8s App

Requires access to a Kubernetes cluster.

```bash
docker rm -f green_agent

docker run -itd --network=host \
  -v <KUBECONFIG_PATH>:/root/.kube/config \
  -e KUBECONFIG=/root/.kube/config \
  -v <NETPRESS_ROOT>/app-k8s/microservices-demo:/data/microservices-demo \
  --name green_agent k8s_agent:latest --host "0.0.0.0" --port 9999
```

**Replace:**
- `<KUBECONFIG_PATH>` — Your kubeconfig file (default: `~/.kube/config`, or `app-k8s/config` if included)
- `<NETPRESS_ROOT>` — Absolute path to NetPress repo (e.g., `/home/user/NetPress`)

**No remote cluster?** Follow the [Kubernetes in Docker (KinD)](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) installation instructions to create a local cluster.

---

## Troubleshooting

### K8s connection refused

If kubectl commands fail with "connection refused":

1. Verify your kubeconfig is mounted correctly
2. Check the remote cluster is reachable: `nc -zv <CLUSTER-IP> 6443`
3. Ensure the `KUBECONFIG` env var is set in the container

### Agent not responding

Check if the container is healthy:

```bash
docker logs --tail 50 green_agent
curl http://127.0.0.1:9999/.well-known/agent-card.json
```
