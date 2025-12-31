#!/usr/bin/env python3
"""
K8s Benchmark Runner
====================
Simple entry point for running K8s network policy evaluation.

Usage:
    python run.py                    # Uses config.toml
    python run.py --config my.toml   # Uses custom config file
"""

import argparse
import asyncio
import os
import sys
from cattrs import structure
from loguru import logger
from run_workflow import run_evaluation, K8sConfig

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def load_config(config_path: str) -> dict:
    """Load and validate TOML configuration file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        template_path = os.path.join(os.path.dirname(config_path) or ".", "config.template.toml")
        if os.path.exists(template_path):
            print(f"\nTo get started, copy the template and add your credentials:")
            print(f"  cp {template_path} {config_path}")
        else:
            print("Please create a config.toml file or specify a valid config path.")
        sys.exit(1)
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    return config


def print_config_summary(config: dict) -> None:
    """Print a summary of the configuration being used."""
    model = config.get("model", {})
    benchmark = config.get("benchmark", {})
    paths = config.get("paths", {})
    
    print("\n" + "=" * 60)
    print("K8s Benchmark Configuration")
    print("=" * 60)
    print(f"  Agent:           {model.get('agent_type', 'GPT-4o')}")
    print(f"  Prompt Type:     {model.get('prompt_type', 'base')}")
    print(f"  Queries/Type:    {benchmark.get('num_queries', 10)}")
    print(f"  Max Iterations:  {benchmark.get('max_iteration', 10)}")
    print(f"  Output Dir:      {paths.get('output_dir', 'results')}/")
    print(f"  Microservices:   {paths.get('microservice_dir', '/microservices-demo')}")
    print(f"  Agent Test:      {'Yes' if benchmark.get('agent_test', False) else 'No'}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run K8s network policy benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    # Use default config.toml
    python run.py -c custom.toml     # Use custom config file
    python run.py --show-config      # Show current configuration
        """
    )
    parser.add_argument(
        "-c", "--config",
        default="config.toml",
        help="Path to TOML configuration file (default: config.toml)"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration and exit without running"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print summary
    print_config_summary(config)
    
    if args.show_config:
        print("Configuration loaded successfully. Use without --show-config to run.")
        return
    
    # Setup environment variables BEFORE importing modules that need them
    
    # Convert to args namespace for run_workflow.py compatibility
    run_args = structure(config, K8sConfig)
    
    # Create output directory
    os.makedirs(run_args.output_dir, exist_ok=True)
    
    # Run the benchmark
    print("Starting K8s benchmark evaluation...\n")
    
    asyncio.run(run_evaluation(run_args))
    
    print("\nBenchmark evaluation complete!")


if __name__ == "__main__":
    main()

