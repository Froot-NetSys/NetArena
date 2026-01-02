import os
import subprocess
from loguru import logger


POLICY_NAMES = [
    "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice",
    "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend",
    "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice",
    "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice"
]

POD_NAMES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend",
    "loadgenerator", "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart", "shippingservice"
]


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