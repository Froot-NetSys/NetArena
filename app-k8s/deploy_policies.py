import os
import subprocess
from loguru import logger


def deploy_policies(policy_names: list[str], root_dir: str):
    """Deploy policies to the Kubernetes cluster."""
    logger.info(f"Deploying policies: {policy_names}")
    for name in policy_names:
        filename = os.path.join(root_dir, "policies", f"{name}.yaml")
        try:
            result = subprocess.run(["kubectl", "apply", "-f", filename], check=True, text=True, capture_output=True)
            logger.info(f"Deployed {filename}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy {filename}:\n{e.stderr}")