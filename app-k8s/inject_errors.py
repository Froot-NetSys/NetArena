import os
import random
import yaml
import itertools
import json
from typing import List, Dict
from loguru import logger

from deploy_policies import POLICY_NAMES


def fetch_error_config(benchmark_path: str, num_queries: int, generate_config_flag: bool) -> List[Dict]:
    """Fetch error configuration from a JSON file."""
    if generate_config_flag:
        logger.info("Force generating new error configuration...")
        config_data = generate_config(benchmark_path, POLICY_NAMES, num_queries)
    else:
        if os.path.exists(benchmark_path):
            logger.info(f"Loading error configuration from {benchmark_path}...")
            error_config_path = benchmark_path
            with open(error_config_path, "r") as f:
                config_data = json.load(f)['details']
        else:
            logger.info("Config file not found, generating new error configuration...")
            config_data = generate_config(benchmark_path, POLICY_NAMES, num_queries)
    return config_data

def generate_config(config_path, policy_names, num_queries):
    # Define error types and combinations
    basic_errors = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]
    error_combinations = list(itertools.combinations(basic_errors, 2))
    error_config = []

    # Predefined detail ranges
    detail_range = {
        "add_ingress_rule": {
            "adservice": ["recommendationservice", "productcatalogservice", "cartservice", "checkoutservice",  "emailservice", "shippingservice"],
            "recommendationservice": ["adservice", "productcatalogservice", "cartservice", "checkoutservice", "emailservice", "shippingservice"],
            "productcatalogservice": ["adservice", "recommendationservice", "cartservice", "emailservice", "shippingservice"],
            "redis": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "checkoutservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "shippingservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice"],
            "currencyservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "paymentservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "emailservice": ["loadgenerator", "frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "shippingservice"],
            "cartservice": ["adservice", "recommendationservice", "productcatalogservice","emailservice", "shippingservice"],
            "loadgenerator": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "emailservice", "shippingservice"],
            "frontend": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "emailservice", "shippingservice"]
        },
        "add_egress_rule": {
            "adservice": ["recommendationservice", "productcatalogservice", "cartservice", "checkoutservice",  "emailservice", "shippingservice","redis-cart"],
            "productcatalogservice": ["adservice", "recommendationservice", "cartservice", "emailservice", "shippingservice","redis-cart"],
            "shippingservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice","redis-cart"],
            "paymentservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "currencyservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice", "redis-cart"],
            "emailservice": ["loadgenerator", "frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "shippingservice", "redis-cart"],
            "recommendationservice": ["adservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice","redis-cart"],
            "checkoutservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice", "redis-cart"],
            "cartservice": ["adservice", "recommendationservice", "productcatalogservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "frontend": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice", "resid-cart"],
            "redis":["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"]
        },
        "remove_ingress_rule": {
            "adservice": ["frontend"],
            "recommendationservice": ["frontend"],
            "productcatalogservice": ["frontend", "recommendationservice", "checkoutservice"],
            "cartservice": ["frontend", "checkoutservice"],
            "checkoutservice": ["frontend"],
            "shippingservice": ["frontend", "checkoutservice"],
            "currencyservice": ["frontend", "checkoutservice"],
            "paymentservice": ["checkoutservice"],
            "emailservice": ["checkoutservice"]
        },
    }

    # Process single error: remove_ingress (target: min(num_queries, 14))
    target_remove = num_queries if num_queries < 14 else 14
    remove_ingress_entries = []
    for pol, apps in detail_range["remove_ingress_rule"].items():
        for app in apps:
            remove_ingress_entries.append((pol, app))
    remove_ingress_entries = remove_ingress_entries[:target_remove]
    for pol, app in remove_ingress_entries:
        detail = {"type": "remove_ingress", "app": app}
        policies = [f"network-policy-{pol}"]
        error_config.append({
            "policies_to_inject": policies,
            "inject_error_num": [1],
            "error_detail": [detail]
        })

    # Process single error: change_protocol (target: min(num_queries, 18))
    count = 0
    for policy in policy_names:
        if not (policy == "network-policy-frontend" or policy == "netwwork-policy-loadgenerator" or policy == "network-policy-loadgenerator"):
            for key_value in ["UDP", "SCTP"]:
                detail = {"type": "change_protocol", "new_protocol": key_value}
                policies = [policy]
                count = count + 1
                if count <= num_queries:
                    error_config.append({
                        "policies_to_inject": policies,
                        "inject_error_num": [1],
                        "error_detail": [detail]
                    })

    # Process other single errors 
    for error in basic_errors:
        if error == "remove_ingress" or error == "change_protocol":
            continue  # already processed
        for _ in range(num_queries):
            detail = {"type": error}
            policy = random.choice(policy_names)
            policy_name = policy.replace("network-policy-", "")
            if error == "add_ingress":
                allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                policy_name = random.choice(allowed_policies)
                policy = f"network-policy-{policy_name}"  
                if policy_name in detail_range["add_ingress_rule"]:
                    detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
            elif error == "change_port":
                allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                policy_name = random.choice(allowed_policies) 
                policy = f"network-policy-{policy_name}" 
                detail["new_port"] = random.randint(1, 65535)
            elif error == "add_egress":
                allowed_policies = ["recommendationservice", "checkoutservice", "cartservice", "frontend"]
                policy_name = random.choice(allowed_policies)  
                policy = f"network-policy-{policy_name}"  
                detail["app"] = random.sample(detail_range["add_egress_rule"][policy_name], 2)
            error_config.append({
                "policies_to_inject": [policy],
                "inject_error_num": [1],
                "error_detail": [detail]
            })

    # Process combination errors: generate num_queries records for each combination
    for combo in error_combinations:
        for _ in range(num_queries):
            details = []
            policies = []
            
            for error in combo:    
                detail = {"type": error}
                if error == "add_ingress":
                    policy_name = random.choice(["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice", "frontend"])
                    policy = f"network-policy-{policy_name}"
                    policies.append(policy)
                    if policy_name in detail_range.get("add_ingress_rule", {}):
                        detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
                elif error == "change_port":
                    allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                    policy_name = random.choice(allowed_policies) 
                    policy = f"network-policy-{policy_name}" 
                    policies.append(policy)
                    detail["new_port"] = random.randint(1, 65535)
                elif error == "change_protocol":
                    allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                    policy_name = random.choice(allowed_policies) 
                    policy = f"network-policy-{policy_name}" 
                    policies.append(policy)
                    detail["new_protocol"] = random.choice(["UDP", "SCTP"])
                elif error == "add_egress":
                    allowed_policies = ["recommendationservice", "checkoutservice", "cartservice", "frontend"]
                    policy_name = random.choice(allowed_policies)  
                    policy = f"network-policy-{policy_name}"  
                    policies.append(policy)
                    if policy_name in detail_range.get("add_egress_rule", {}):
                        detail["app"] = random.sample(detail_range["add_egress_rule"][policy_name], 2)
                elif error == "remove_ingress":
                    allowed_policies = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice"]
                    policy_name = random.choice(allowed_policies)
                    policy = f"network-policy-{policy_name}"
                    if policy_name in detail_range.get("remove_ingress_rule", {}):
                        detail["app"] = random.choice(detail_range["remove_ingress_rule"][policy_name])
                    policies.append(policy)
                details.append(detail)
            
            error_config.append({
                "policies_to_inject": policies,
                "inject_error_num": [len(combo)],
                "error_detail": details
            })

    # Save result to file
    with open(config_path, "w") as f:
        json.dump({"details": error_config}, f, indent=2)

    return error_config

def inject_config_errors_into_policies(
    policy_names: List[str],
    root_dir: str,
    inject_error_num: List[int],  # Strictly validate this parameter
    policies_to_inject: List[str],
    error_detail: List[Dict]
):
    """
    Strict validation for precise error injection
    
    Parameter structure as required:
    policy_names, root_dir, inject_error_num, policies_to_inject, error_detail
    """
    # Strict parameter validation (three new validation layers)
    if not isinstance(inject_error_num, list):
        raise TypeError("inject_error_num must be a list")
    
    if len(inject_error_num) != 1:
        raise ValueError("inject_error_num must be a single-element list, e.g., [2]")
    
    if inject_error_num[0] != len(error_detail):
        raise ValueError(
            f"Error count mismatch! Config declares {inject_error_num[0]} errors, "
            f"but actually provided {len(error_detail)} error details"
        )
    print(f"policies_to_inject:", policies_to_inject)
    # Validate policy name validity
    invalid_policies = [name for name in policies_to_inject if name not in policy_names]
    if invalid_policies:
        raise ValueError(f"Invalid policy names: {invalid_policies}")

    # Iterate over each target policy to inject errors
    for i, policy_name in enumerate(policies_to_inject):
        policy_path = os.path.join(root_dir, 'policies', f"{policy_name}.yaml")
        
        # Read policy file
        try:
            with open(policy_path, "r") as f:
                original_policy = yaml.safe_load(f)
            print(f"[INFO] Original policy loaded: {policy_path}")
        except FileNotFoundError:
            print(f"[ERROR] Policy file not found: {policy_path}")
            continue

        # Perform injection and carry error count validation
        modified_policy = _inject_errors_with_detail(
            original_policy,
            error_detail,
            i + 1  # number_of_errors represents the current iteration
        )
        print(f"[INFO] Modified policy: {modified_policy},{policy_name}")
        # Write back to file
        with open(policy_path, "w") as f:
            yaml.dump(modified_policy, f, default_flow_style=False)
        
        print(f"Successfully injected {len(error_detail)} errors into {policy_name}")
        print(modified_policy)
        print(i)

    return modified_policy

def _inject_errors_with_detail(
    policy: Dict,
    error_details: List[Dict],
    number_of_error: int
) -> Dict:
    """Enhanced core logic for error injection with correct ingress/egress structure"""

    modified_policy = policy.copy()
    

    detail = error_details[number_of_error - 1]
    error_type = detail["type"]
    
    match error_type:
        case "remove_ingress":
            if modified_policy["spec"].get("ingress"):
                if modified_policy["spec"]["ingress"]:
                    modified_policy["spec"]["ingress"].pop(0)

        case "add_ingress":
            _validate_required_fields(detail, ["app"])
            if not isinstance(detail["app"], list) or not detail["app"]:
                raise ValueError(f"Invalid app list in add_ingress: {detail['app']}")

            new_rules = [
                {
                    "from": [{"podSelector": {"matchLabels": {"app": app}}}]
                }
                for app in detail["app"]
            ]

            modified_policy["spec"].setdefault("ingress", []).extend(new_rules)
        
        case "change_port":
            _validate_required_fields(detail, ["new_port"])
            if "ingress" in modified_policy["spec"]:
                for rule in modified_policy["spec"]["ingress"]:
                    for port in rule.get("ports", []):
                        port["port"] = detail["new_port"]
        
        case "change_protocol":
            _validate_required_fields(detail, ["new_protocol"])
            if "ingress" in modified_policy["spec"]:
                for rule in modified_policy["spec"].get("ingress", []):
                    for port in rule.get("ports", []):
                        port["protocol"] = detail["new_protocol"]
        
        case "add_egress":
            _validate_required_fields(detail, ["app"])
            if not isinstance(detail["app"], list) or not detail["app"]:
                raise ValueError(f"Invalid app list in add_egress: {detail['app']}")

            # Remove empty egress rules
            modified_policy["spec"]["egress"] = [
                rule for rule in modified_policy["spec"].get("egress", []) if rule
            ]

            new_rules = [
                {
                    "to": [{"podSelector": {"matchLabels": {"app": app}}}]
                }
                for app in detail["app"]
            ]

            modified_policy["spec"].setdefault("egress", []).extend(new_rules)
        
        case _:
            raise ValueError(f"Unknown error type: {error_type}")

    # Maintain field order and ensure correct format
    return {
        "apiVersion": modified_policy.get("apiVersion", "networking.k8s.io/v1"),
        "kind": "NetworkPolicy",
        "metadata": modified_policy.get("metadata", {}),
        "spec": {
            "podSelector": modified_policy["spec"].get("podSelector", {}),
            "policyTypes": modified_policy["spec"].get("policyTypes", []),
            "ingress": modified_policy["spec"].get("ingress", []),
            "egress": modified_policy["spec"].get("egress", [])
        }
    }

def _validate_required_fields(detail: Dict, required_fields: List[str]):
    """Validate required fields are present"""
    missing = [field for field in required_fields if field not in detail]
    if missing:
        raise ValueError(
            f"Error type {detail['type']} is missing required fields: {missing}"
        )
