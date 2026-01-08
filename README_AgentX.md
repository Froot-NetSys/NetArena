# Green Agent Container Testing

How to build and test that the green agent containers behave as expected.

## Prerequisites

- Linux-based system (tested on Ubuntu 20.04)
- At least 4 CPU cores
- Docker installed and running
- `uv` package manager installed

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

### 2. Modify the Demo Script

In the `green_agent_demo.sh` file for your app:

1. **Update the LLM credentials** - Replace the API key and endpoint with your own:

   ```bash
   export AZURE_API_KEY="<YOUR_API_KEY>"
   export AZURE_API_BASE="<YOUR_API_ENDPOINT>"
   export AZURE_API_VERSION="<YOUR_API_VERSION>"
   MODEL_NAME="azure/XXX"
   ```

2. **Comment out** the lines that launch the green agent locally (we'll run it in a container instead):

   ```bash
   # uv run ./<APP-NAME>_agent.py &
   # server_pid2=$!
   ```

### 3. Start the Container

See [How to Start the Container for Each App](#how-to-start-the-container-for-each-app) below for app-specific commands.

### 4. Run the Demo

```bash
cd ~/NetPress/app-<APP-NAME>/green_agent
./green_agent_demo.sh
```

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

**No remote cluster?** Follow the instructions in [`app-k8s/README.md`](./app-k8s/README.md) to create a local cluster using Kubernetes in Docker (KinD).

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

