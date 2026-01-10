import asyncio
import subprocess
import json
import time
from loguru import logger


# Expected connectivity results for correctness checking
EXPECTED_RESULTS = {
    "frontend": {
        "adservice:9555": True,
        "cartservice:7070": True,
        "checkoutservice:5050": True,
        "currencyservice:7000": True,
        "productcatalogservice:3550": True,
        "recommendationservice:8080": True,
        "shippingservice:50051": True,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "adservice": {
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "cartservice": {
        "adservice:9555": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": True
    },
    "checkoutservice": {
        "adservice:9555": False,
        "cartservice:7070": True,
        "currencyservice:7000": True,
        "productcatalogservice:3550": True,
        "recommendationservice:8080": False,
        "shippingservice:50051": True,
        "emailservice:5000": True,
        "paymentservice:50051": True,
        "redis-cart:6379": False
    },
    "productcatalogservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "recommendationservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": True,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "shippingservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "emailservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "redis-cart": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False
    },
    "loadgenerator": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    }
}


async def find_pod_by_prefix(prefix):
    """Find a pod whose name starts with the specified prefix."""
    try:
        result = await asyncio.create_subprocess_exec(
            "kubectl", "get", "pods", "--no-headers", "-o", "custom-columns=:metadata.name",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await result.communicate()
        pods = stdout.decode().strip().split("\n")
        for pod in pods:
            if pod.startswith(prefix):
                return pod
    except Exception as e:
        logger.error(f"Error listing pods: {e}")
    return None

async def wait_for_debug_container(pod_name, container_prefix="debugger-", timeout=60):
    """
    Poll pod information until the debug container (name starts with container_prefix) is in the running state.
    Note: The debug container will appear in the pod's ephemeralContainers.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pod", pod_name, "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            pod_json = json.loads(stdout.decode())
            ephemeral_containers = pod_json.get("spec", {}).get("ephemeralContainers", [])
            statuses = pod_json.get("status", {}).get("ephemeralContainerStatuses", [])
            for ec in ephemeral_containers:
                name = ec.get("name", "")
                if name.startswith(container_prefix):
                    # Check if the status is running
                    for status in statuses:
                        if status.get("name") == name:
                            if "running" in status.get("state", {}):
                                return name
        except Exception as e:
            logger.error(f"Error fetching pod info: {e}")
        await asyncio.sleep(1)
    return None

async def create_debug_container(pod_name_prefix, timeout=30):
    """Create a debug container in the specified pod.
    
    Args:
        pod_name: Name of the target pod
        timeout: Timeout for command execution (default: 3 seconds)
        
    Returns:
        Name of the created debug container or None if failed
    """
    # Determine target container based on pod name patterns
    if 'loadgenerator' in pod_name_prefix:
        target = "main"  # Verify actual container name for loadgenerator
    elif 'redis-cart' in pod_name_prefix:
        target = "redis"  # Container name from pod spec
    else:
        target = "server"  # Default assumption for other services
    pod_name = await find_pod_by_prefix(pod_name_prefix)
    if not pod_name:
        print(f"Pod {pod_name_prefix} not found")
        return None
    # Construct debug command with dynamic target container
    debug_command = [
        "kubectl", "debug", "-it", pod_name,
        "--image=busybox",
        f"--target={target}",  # Dynamically set target container
        "--quiet",  # Suppress verbose output
        "--attach=false",  # Run in detached mode
        "--", "sleep", "infinity"  # Keep container alive
    ]
    try:
        # Execute debug container creation
        logger.info(f"Creating debug container in pod {pod_name} targeting container {target}")
        logger.debug(f"Debug command: {' '.join(debug_command)}")
        process = await asyncio.create_subprocess_exec(
            *debug_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Timeout creating debug container in pod {pod_name}")
        return None
    except Exception as e:
        logger.error(f"Error creating debug container in pod {pod_name}: {e}")
        return None

    # Wait for the debug container to start
    debug_container_name = await wait_for_debug_container(pod_name)
    if not debug_container_name:
        logger.error(f"Failed to detect debug container in pod {pod_name}")
    return debug_container_name

async def check_connectivity_with_debug(pod_name, debug_container_name, host, port, timeout=1):
    """Check connectivity using kubectl exec."""
    nc_command = [
        "kubectl", "exec", pod_name,
        "-c", debug_container_name,
        "--", "nc", "-zv", "-w", str(timeout), host, str(port)
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *nc_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout + 1)
        output = (stdout + stderr).decode().strip()
        return "open" in output
    except asyncio.TimeoutError:
        return False
    except Exception as e:
        logger.error(f"Error executing nc command: {e}")
        return False

async def process_pod(pod_prefix, targets, debug_container_mapping):
    """Process a single pod and check connectivity for all targets."""
    pod_name = await find_pod_by_prefix(pod_prefix)  
    if not pod_name:
        return False, f"Pod {pod_prefix} not found"

    debug_container_name = debug_container_mapping.get(pod_prefix)
    if not debug_container_name:
        return False, f"Debug container for pod {pod_name} not found in mapping"

    pod_all_match = True
    pod_mismatch_messages = []

    tasks = []
    for target, expected in targets.items():
        host, port = target.split(":")
        tasks.append(check_connectivity_with_debug(pod_name, debug_container_name, host, port))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for (target, expected), actual in zip(targets.items(), results):
        if isinstance(actual, Exception):
            pod_mismatch_messages.append(f"Error checking connectivity for {pod_prefix} → {target}: {actual}")
            pod_all_match = False
        elif actual != expected:
            pod_mismatch_messages.append(f"Mismatch: {pod_prefix} → {target} (Expected: {expected}, Actual: {actual})")
            pod_all_match = False

    return pod_all_match, "\n".join(pod_mismatch_messages)

async def correctness_check(debug_container_mapping, expected_results=EXPECTED_RESULTS):
    """Check connectivity for all pods in parallel."""
    tasks = [
        process_pod(pod_prefix, targets, debug_container_mapping)
        for pod_prefix, targets in expected_results.items()
    ]
    results = await asyncio.gather(*tasks)

    all_match = True
    mismatch_messages = []
    for pod_all_match, pod_mismatch_summary in results:
        if not pod_all_match:
            all_match = False
            mismatch_messages.append(pod_mismatch_summary)

    mismatch_summary = "\n".join(mismatch_messages) if mismatch_messages else "No mismatches found."
    return all_match, mismatch_summary

async def main():
    expected_results = {
        "frontend": {
            "adservice:9555": True,
            "cartservice:7070": True,
            "checkoutservice:5050": True,
            "currencyservice:7000": True,
            "productcatalogservice:3550": True,
            "recommendationservice:8080": True,
            "shippingservice:50051": True,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "adservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "cartservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": True
        },
        "checkoutservice": {
            "adservice:9555": False,
            "cartservice:7070": True,
            "checkoutservice:5050": False,
            "currencyservice:7000": True,
            "productcatalogservice:3550": True,
            "recommendationservice:8080": False,
            "shippingservice:50051": True,
            "emailservice:5000": True,
            "paymentservice:50051": True,
            "redis-cart:6379": False
        },
        "currencyservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "productcatalogservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "recommendationservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": True,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "shippingservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "emailservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "paymentservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "redis-cart": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "loadgenerator": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        }
    }

    pod_names = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "loadgenerator", "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart", "shippingservice"]
    debug_container_mapping = {}
    for pod_name in pod_names:
        debug_container_name = await create_debug_container(pod_name)
        if debug_container_name:
            debug_container_mapping[pod_name] = debug_container_name
    start_time = time.time()
    all_match, mismatch_summary = await correctness_check(expected_results, debug_container_mapping)
    print(f"\nFinal result: All tests passed: {all_match}")
    print(f"Mismatch details: {mismatch_summary}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    exit(0 if all_match else 1)
