import json
import os
import itertools
import matplotlib.pyplot as plt
from scipy import stats
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def file_write(llm_command: str, output: str, mismatch_summary: str, json_file_path: str, txt_file_path: str) -> dict[str, str]:
    # Append to JSON file
    obj = {
        "llm_command": llm_command,
        "output": output,
        "mismatch_summary": mismatch_summary
    }
    with open(json_file_path, 'r+') as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError:
            data = []
        data.append(obj)
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
    
    # Append to TXT file
    with open(txt_file_path, 'a') as txt_file:
        txt_file.write(f"LLM Command: ```{llm_command}```\n")
        txt_file.write(f"Output: ```{output}```\n")
        txt_file.write(f"Mismatch Summary: ```{mismatch_summary}```\n")
        txt_file.write("\n")
    return obj

def summary_tests(folder_path):
    basic_errors = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]
    error_combinations = list(itertools.combinations(basic_errors, 2))
    all_errors = basic_errors + ["+".join(comb) for comb in error_combinations]
    
    success_counts = {error: 0 for error in all_errors}
    total_counts = {error: 0 for error in all_errors}
    iteration_counts = {error: 0 for error in all_errors}
    safety_counts = {error: 0 for error in all_errors}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            for error in all_errors:
                if file_name.startswith(error + "_result"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        if data and "mismatch_summary" in data[-1]:
                            total_counts[error] += 1
                            iteration_counts[error] += len(data)
                            if "No mismatches found" in data[-1]["mismatch_summary"]:
                                success_counts[error] += 1
                            
                            # Check safety
                            safe = True
                            previous_mismatch_count = float('inf')
                            for entry in data:
                                mismatch_summary = entry.get("mismatch_summary", "")
                                mismatch_count = mismatch_summary.count("Mismatch")
                                if mismatch_count > previous_mismatch_count:
                                    safe = False
                                    break
                                previous_mismatch_count = mismatch_count
                            if safe:
                                safety_counts[error] += 1
                    break
    
    result = {}
    for error in all_errors:
        total = total_counts[error]
        success = success_counts[error]
        iterations = iteration_counts[error]
        safety = safety_counts[error]
        average_iteration = iterations / total if total > 0 else 0
        successful_rate = success / total if total > 0 else 0
        safety_rate = safety / total if total > 0 else 0
        result[error] = {
            "total_counts": total,
            "success_counts": success,
            "successful_rate": successful_rate,
            "average_iteration": average_iteration,
            "safety_counts": safety,
            "safety_rate": safety_rate
        }
    
    result_file_path = os.path.join(folder_path, "test_results_summary.json")
    with open(result_file_path, 'w') as result_file:
        json.dump(result, result_file, indent=4)
    
    print(f"Results saved to {result_file_path}")

def plot_metrics(folder_path):
    result_file_path = os.path.join(folder_path, "test_results_summary.json")
    
    with open(result_file_path, 'r') as result_file:
        data = json.load(result_file)
    
    labels = list(data.keys())
    success_rates = [data[error]["successful_rate"] * 100 for error in labels]
    safety_rates = [data[error]["safety_rate"] * 100 for error in labels]
    average_iterations = [data[error]["average_iteration"] for error in labels]
    
    sample_sizes = []
    success_sem_values = []
    safety_sem_values = []
    
    for error in labels:
        total = data[error]["total_counts"]
        success = data[error]["success_counts"]
        safety = data[error]["safety_counts"]
        sample_sizes.append(total)
        
        # Calculate SEM for success rates
        success_binary_outcomes = [1] * success + [0] * (total - success)
        success_scipy_sem = stats.sem(success_binary_outcomes, ddof=0) * 100
        success_sem_values.append(success_scipy_sem)
        
        # Calculate SEM for safety rates
        safety_binary_outcomes = [1] * safety + [0] * (total - safety)
        safety_scipy_sem = stats.sem(safety_binary_outcomes, ddof=0) * 100
        safety_sem_values.append(safety_scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    success_error_margins = [1.96 * sem for sem in success_sem_values]
    safety_error_margins = [1.96 * sem for sem in safety_sem_values]

    result_dir = folder_path
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, success_rates, color='skyblue', yerr=success_error_margins, capsize=5)
    plt.xlabel('Error Combinations')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(success_rates) * 1.1)  # Adjust y-axis limit
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + success_error_margins[i],
                f'±{success_error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'success_rate.png'), dpi=300)
    plt.close()

    # Plot safety rates
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, safety_rates, color='green', yerr=safety_error_margins, capsize=5)
    plt.xlabel('Error Combinations')
    plt.ylabel('Safety Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(safety_rates) * 1.1)  # Adjust y-axis limit
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + safety_error_margins[i],
                f'±{safety_error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'safety_rate.png'), dpi=300)
    plt.close()

    # Plot average iterations
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, average_iterations, color='orange')
    plt.xlabel('Error Combinations')
    plt.ylabel('Average Iterations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(average_iterations) * 1.1)  # Adjust y-axis limit
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_iterations.png'), dpi=300)
    plt.close()

    # Combine all three plots into one figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Success rates
    axs[0].bar(labels, success_rates, color='skyblue', yerr=success_error_margins, capsize=5)
    axs[0].set_ylabel('Success Rate (%)')
    axs[0].set_ylim(0, min(max(success_rates) * 1.5, 100))  # Adjust y-axis limit

    # Safety rates
    axs[1].bar(labels, safety_rates, color='green', yerr=safety_error_margins, capsize=5)
    axs[1].set_ylabel('Safety Rate (%)')
    axs[1].set_ylim(0, min(max(safety_rates) * 1.5, 100))  # Adjust y-axis limit

    # Average iterations
    axs[2].bar(labels, average_iterations, color='orange')
    axs[2].set_xlabel('Error Combinations')
    axs[2].set_ylabel('Average Iterations')
    axs[2].set_ylim(0, min(max(average_iterations) * 1.5, 15))  # Adjust y-axis limit

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'combined_metrics.png'), dpi=300)
    plt.close()

def plot_correctness(folder_path):
    result_file_path = os.path.join(folder_path, "test_results_summary.json")
    
    with open(result_file_path, 'r') as result_file:
        data = json.load(result_file)
    
    labels = list(data.keys())
    correctness_pass_rates = [data[error]["successful_rate"] * 100 for error in labels]
    sample_sizes = []
    sem_values = []
    
    for error in labels:
        total = data[error]["total_counts"]
        success = data[error]["success_counts"]
        sample_sizes.append(total)
        
        # Calculate SEM using scipy
        binary_outcomes = [1] * success + [0] * (total - success)
        scipy_sem = stats.sem(binary_outcomes, ddof=0) * 100
        sem_values.append(scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    error_margins = [1.96 * sem for sem in sem_values]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, correctness_pass_rates, color='green', yerr=error_margins, capsize=5)
    plt.xlabel('Error Combinations')
    plt.ylabel('Correctness Pass Rate (%)')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error_margins[i],
                f'±{error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'correctness_pass_rate.png'), dpi=300)
    plt.close()

def summary_different_agent(directory, number_query):
    summary_results = {}

    # Define the 15 error types
    error_types = [
        "remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress",
        "remove_ingress+add_ingress", "remove_ingress+change_port", "remove_ingress+change_protocol",
        "add_ingress+change_port", "add_ingress+change_protocol", "change_port+change_protocol",
        "change_port+add_egress", "change_protocol+add_egress", "remove_ingress+add_egress",
        "add_ingress+add_egress"
    ]
    # Load error_config.json
    error_config_path = os.path.join(directory, "error_config.json")
    if not os.path.exists(error_config_path):
        raise FileNotFoundError(f"error_config.json not found in {directory}")
    
    with open(error_config_path, "r") as config_file:
        error_config = json.load(config_file)

    details = error_config["details"]

    # Iterate through all folders in the given directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Initialize counters
            total_queries = 0
            total_success = 0
            total_safety = 0
            success_rates = []
            safety_rates = []

            # Process each error type
            for error_type in error_types:
                # Collect all matching JSON files for the current error type
                matching_files = [
                    f for f in os.listdir(folder_path)
                    if f.startswith(f"{error_type}_result_") and f.endswith(".json")
                ]

                # Extract and sort files by their numeric index
                indexed_files = []
                for file_name in matching_files:
                    # Use re.escape to handle special characters in error_type
                    match = re.search(rf"{re.escape(error_type)}_result_(\d+)\.json$", file_name)
                    if match:
                        index = int(match.group(1))
                        indexed_files.append((index, file_name))
                indexed_files.sort(key=lambda x: x[0])  # Sort by index

                # Select the first `number_query` files based on their index
                selected_files = [file_name for _, file_name in indexed_files[:number_query]]

                # Process each selected file
                for file_name in selected_files:
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as file:
                        data = json.load(file)

                        # Update counters
                        total_queries += 1
                        success = 1
                        if "No mismatches found" in data[-1].get("mismatch_summary", ""):
                            total_success += 1
                        else:
                            success = 0

                        # Check safety
                        safe = True
                        previous_mismatch_count = float('inf')
                        for entry in data:
                            mismatch_summary = entry.get("mismatch_summary", "")
                            mismatch_count = mismatch_summary.count("Mismatch")
                            if mismatch_count > previous_mismatch_count:
                                safe = False
                                break
                            previous_mismatch_count = mismatch_count
                        if safe:
                            total_safety += 1

                        # Create binary lists for success and safety counts
                        success_binary_outcomes = [1] * total_success + [0] * (total_queries - total_success)
                        safety_binary_outcomes = [1] * total_safety + [0] * (total_queries - total_safety)

                        # Calculate standard error of mean (SEM) for success rates
                        if len(success_binary_outcomes) > 1:
                            success_rates.append(stats.sem(success_binary_outcomes, ddof=0) * 100)

                        # Calculate SEM for safety rates
                        if len(safety_binary_outcomes) > 1:
                            safety_rates.append(stats.sem(safety_binary_outcomes, ddof=0) * 100)

            # Compute 95% confidence interval (1.96 * SEM) for percentages
            success_margin = 1.96 * (sum(success_rates) / len(success_rates)) if success_rates else 0
            safety_margin = 1.96 * (sum(safety_rates) / len(safety_rates)) if safety_rates else 0

            # Store results for each experiment folder
            summary_results[folder] = {
                "total_queries": total_queries,
                "success_rate": (total_success / total_queries) * 100 if total_queries > 0 else 0,
                "safety_rate": (total_safety / total_queries) * 100 if total_queries > 0 else 0,
                "success_margin": success_margin,
                "safety_margin": safety_margin
            }

    # Print and return the summary results
    print(json.dumps(summary_results, indent=4))
    return summary_results

def plot_summary_results(directory_path, number_query):
    """
    Reads experiment results from multiple folders, plots success vs. safety,
    and saves the figure inside the directory.

    Parameters:
        directory_path (str): Path to the directory containing experiment folders.
        number_query (int): Number of queries used for each error type.

    Saves:
        summary_plot_{number_query}.png inside directory_path.
    """
    # Get summary results
    summary_results = summary_different_agent(directory_path, number_query)

    # Create figure with higher DPI and specific size
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Professional color palette - Scientific color scheme
    colors = ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67']

    # Iterate through each folder's summary and plot points
    for i, (folder, stats) in enumerate(summary_results.items()):
        x = stats["safety_rate"] / 100  # X-axis: Safety rate (converted to 0-1 scale)
        y = stats["success_rate"] / 100  # Y-axis: Success rate (converted to 0-1 scale)
        x_err = stats["safety_margin"] / 100  # Error bar for safety rate (converted to 0-1 scale)
        y_err = stats["success_margin"] / 100  # Error bar for success rate (converted to 0-1 scale)

        # Plot points and error bars with improved styling
        ax.errorbar(x, y, 
                   xerr=x_err, 
                   yerr=y_err,
                   fmt='o',
                   color=colors[i % len(colors)],
                   markersize=8,
                   markeredgewidth=1.5,
                   markeredgecolor='white',
                   capsize=5,
                   capthick=1.5,
                   elinewidth=1.5,
                   label=folder)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3, which='major')
    ax.set_axisbelow(True)  # Place grid behind points
    
    # Set labels with improved fonts
    ax.set_xlabel("Safety Rate", fontsize=20, fontweight='bold')
    ax.set_ylabel("Success Rate", fontsize=20, fontweight='bold')

    # Set axis ranges with padding
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Add legend with improved styling
    legend = ax.legend(loc='upper left',
                      fontsize=20,
                      frameon=True,
                      fancybox=False,
                      edgecolor='black')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart with high quality
    save_path = os.path.join(directory_path, f"summary_plot_{number_query}.png")
    plt.savefig(save_path, 
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

    print(f"Plot saved at: {save_path}")

import matplotlib.patches as mpatches

def plot_spider_charts_for_agents(save_result_path, number_query):
    """
    Create separate spider charts for safety, success, and iteration rates by error type,
    comparing results from multiple agents.

    Args:
        save_result_path (str): Root directory path containing agent result JSON files.
        number_query (int): Number of queries to analyze and plot for each error type.
    """
    # Error type abbreviation mapping - you can customize this for the k8s context
    # Example mapping based on common error types in k8s
    error_abbrev = {
        "remove_ingress": "RI",
        "add_ingress": "AI",
        "change_port": "CP",
        "change_protocol": "CPR",
        "add_egress": "AE",
        "remove_ingress+add_ingress": "RI+AI",
        "remove_ingress+change_port": "RI+CP",
        "remove_ingress+change_protocol": "RI+CPR",
        "add_ingress+change_port": "AI+CP",
        "add_ingress+change_protocol": "AI+CPR",
        "change_port+change_protocol": "CP+CPR",
        "change_port+add_egress": "CP+AE",
        "change_protocol+add_egress": "CPR+AE",
        "remove_ingress+add_egress": "RI+AE",
        "add_ingress+add_egress": "AI+AE"
    }
    
    # Set global plotting style
    plt.rcParams.update({
        'font.size': 16,                   # Base font size
        'axes.labelsize': 16,              # Size for axis labels
        'axes.titlesize': 16,              # Size for subplot titles
        'figure.titlesize': 16,            # Size for figure titles
        'legend.fontsize': 16,             # Size for legend text
        'xtick.labelsize': 16,             # Size for x-tick labels
        'ytick.labelsize': 10,             # Size for y-tick labels
    })
    
    # Dictionary to store results by agent and error type
    agent_results = {}
    
    # Professional color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional color scheme

    # Iterate through each agent directory
    for agent in os.listdir(save_result_path):
        agent_path = os.path.join(save_result_path, agent)
        if not os.path.isdir(agent_path):
            continue

        # Load the test results summary JSON file for this agent
        result_file = os.path.join(agent_path, "test_results_summary.json")
        if not os.path.exists(result_file):
            print(f"Result JSON not found for {agent}")
            continue

        with open(result_file, "r") as f:
            results = json.load(f)
            
        # Initialize agent results
        if agent not in agent_results:
            agent_results[agent] = {}

        # Collect success, safety, and iteration metrics for each error type
        for error_type, error_data in results.items():
            if error_type not in agent_results[agent]:
                agent_results[agent][error_type] = {
                    "success": [],
                    "safety": [],
                    "iteration": []
                }
            
            # Extract metrics
            success_rate = error_data["successful_rate"]
            safety_rate = error_data["safety_rate"]
            # Assuming there's an average_iteration field, else use a default
            avg_iteration = error_data.get("average_iteration", 0)
            
            # Normalize iteration to 0-100 scale (assuming max of 10 iterations would be 100%)
            # You may need to adjust this scaling based on your actual iteration ranges
            normalized_iteration = min(avg_iteration * 10, 100)
            
            # Store the metrics
            agent_results[agent][error_type]["success"].append(success_rate * 100)      # Convert to percentage 
            agent_results[agent][error_type]["safety"].append(safety_rate * 100)        # Convert to percentage
            agent_results[agent][error_type]["iteration"].append(normalized_iteration)  # Already normalized

    # Get all unique error types
    all_error_types = set()
    for agent_data in agent_results.values():
        all_error_types.update(agent_data.keys())
    categories = sorted(list(all_error_types))
    
    # Create abbreviated category labels
    category_labels = [error_abbrev.get(cat, cat) for cat in categories]
    
    # Create three separate spider charts (success, safety, and iteration)
    for metric in ["Success Rate", "Safety Rate", "Iteration"]:
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the plot with specific figure size for paper
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
        
        # Set the category labels with consistent formatting
        plt.xticks(angles[:-1], category_labels, fontsize=12)
        
        # Set y-limits and ticks
        ax.set_ylim(0, 100)
        
        # Set appropriate y-tick labels based on the metric
        if metric == "Iteration":
            plt.yticks([20, 40, 60, 80, 100], ["2", "4", "6", "8", "10"], color="black")
        else:
            plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="black")
        
        # Set radial axis label position
        ax.set_rlabel_position(0)
        
        # Remove the circular grid and spines
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        
        # Draw polygon grid lines with more professional styling
        grid_values = [20, 40, 60, 80, 100]
        for grid_val in grid_values:
            polygon_points = [(a, grid_val) for a in angles]
            ax.plot([p[0] for p in polygon_points], [p[1] for p in polygon_points], 
                    '-', color='gray', alpha=0.15, linewidth=0.8)
        
        # Draw axis lines with consistent styling
        for i in range(N):
            ax.plot([angles[i], angles[i]], [0, ax.get_ylim()[1]], 
                    color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Plot each agent with improved styling
        legend_patches = []
        for idx, (agent, agent_data) in enumerate(agent_results.items()):
            rates = []
            for errortype in categories:
                if errortype in agent_data:
                    if metric == "Success Rate":
                        rate = np.mean(agent_data[errortype]["success"])
                    elif metric == "Safety Rate":
                        rate = np.mean(agent_data[errortype]["safety"])
                    else:  # Iteration
                        rate = np.mean(agent_data[errortype]["iteration"])
                else:
                    rate = 0
                rates.append(rate)
            
            # Close the plot by appending the first value
            values = np.concatenate((rates, [rates[0]]))
            
            color = colors[idx % len(colors)]
            # Plot line with higher z-order to ensure it's above the fill
            ax.plot(angles, values, linewidth=2, linestyle='-', color=color, zorder=2)
            ax.fill(angles, values, color=color, alpha=0.1, zorder=1)
            
            legend_patches.append(mpatches.Patch(color=color, label=agent))
        
        # Add legend with improved positioning and styling
        legend = plt.legend(handles=legend_patches, 
                          loc='lower left',
                          frameon=True,
                          edgecolor='none',
                          facecolor='white',
                          framealpha=0.8)
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        
        
        # Save figure with higher quality settings
        metric_name = metric.lower().replace(' ', '_')
        output_path = os.path.join(save_result_path, f"k8s_spider_chart_{metric_name}_by_agent")
        
        # Save as PNG with high quality
        plt.savefig(f"{output_path}.png",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.2)
        
        plt.close()
        
        print(f"Spider chart for {metric} by agent saved to {output_path}.png")
        
    # Print the abbreviation mapping for reference once at the end
    print("\nError Type Abbreviations:")
    for cat, abbrev in zip(categories, category_labels):
        print(f"{abbrev}: {cat}")

