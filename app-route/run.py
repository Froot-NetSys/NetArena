#!/usr/bin/env python3
"""
Route Benchmark Runner
======================
Simple entry point for running route evaluation.

Usage:
    python run.py                    # Uses config.toml
    python run.py --config my.toml   # Uses custom config file
"""

import argparse
import os
import sys
from datetime import datetime
import asyncio
from loguru import logger
from cattrs import structure

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from test_function import run_benchmark, AppRouteConfig


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
    topology = config.get("topology", {})
    output = config.get("output", {})
    
    print("\n" + "=" * 60)
    print("Route Benchmark Configuration")
    print("=" * 60)
    print(f"  Agent:        {model.get('agent_type', 'GPT-Agent')}")
    print(f"  Prompt Type:  {model.get('prompt_type', 'base')}")
    print(f"  Queries:      {benchmark.get('num_queries', 10)}")
    print(f"  Max Iter:     {benchmark.get('max_iteration', 10)}")
    print(f"  Topology:     {topology.get('num_switches', 2)} switches, {topology.get('num_hosts_per_subnet', 1)} hosts/subnet")
    print(f"  Output:       {output.get('output_dir', 'results')}/")
    print(f"  Parallel:     {'Yes' if benchmark.get('parallel', False) else 'No'}")
    print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Run route benchmark evaluation",
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
    
    # Convert to args namespace for main.py compatibility
    run_args = structure(config, AppRouteConfig)
    
    # Run the benchmark
    print("Starting route benchmark evaluation...\n")
    start_time = datetime.now()
    
    await run_benchmark(run_args)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nBenchmark completed in {duration}")


if __name__ == "__main__":
    asyncio.run(main())
