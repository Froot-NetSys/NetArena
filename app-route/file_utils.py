import os
import json
import matplotlib.pyplot as plt
import shutil
from scipy import stats
import numpy as np
import matplotlib.patches as mpatches
import math

def prepare_file(file_path):
    """
    Prepares the specified file for use. If it exists, clears its content.
    If it doesn't exist, creates the file and any missing directories.

    Parameters:
        file_path (str): The path to the file.
    """
    if os.path.exists(file_path):
        # Clear the file's content
        with open(file_path, "w") as f:
            pass
    else:
        # Create missing directories and the file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("")

def delete_result_folder(folder_path):
    """
    Deletes the specified folder and its contents if it exists.

    Parameters:
        folder_path (str): The path to the folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def initialize_json_file(json_path):
    """
    Create an empty JSON file and its directory if they don't exist.
    
    Args:
        json_path (str): The path of the JSON file to initialize.
    """
    # Ensure the directory exists
    directory = os.path.dirname(json_path)
    if directory:  # Check if the path includes a directory
        os.makedirs(directory, exist_ok=True)

    # Create the JSON file if it doesn't exist
    if not os.path.exists(json_path):
        with open(json_path, 'w') as json_file:
            json.dump([], json_file, indent=4)

def write_query_result(json_path, result_data):
    """
    Write the result data to a JSON file. If the file exists, append the data;
    otherwise, create a new file.

    Args:
        json_path (str): The path to the JSON file.
        result_data (dict): The result data to write.
    """
    if os.path.exists(json_path):
        with open(json_path, 'r+') as json_file:
            data = json.load(json_file)
            data.append(result_data)
            json_file.seek(0)
            json.dump(data, json_file, indent=4)
    else:
        with open(json_path, 'w') as json_file:
            json.dump([result_data], json_file, indent=4)

def write_log_content(log_path, content):
    """
    Write content to a log file. If the file exists, append the content;
    otherwise, create a new file.

    Args:
        log_path (str): The path to the log file.
        content (str): The content to write to the log file.
    """
    with open(log_path, 'a') as log_file:
        log_file.write(content + '\n')

def static_summarize_results(json_folder, output_file):
    """Summarize results from multiple JSON files in a folder and write to a new JSON file."""
    total_success = 0
    total_iterations = 0
    total_average_time = 0
    total_safe_files = 0
    total_packet_loss_reduction = 0
    total_files_with_reduction = 0

    # Get all JSON file paths in the folder
    json_files = [
        os.path.join(json_folder, f)
        for f in os.listdir(json_folder)
        if f.endswith(".json") and os.path.isfile(os.path.join(json_folder, f))
    ]
    total_files = len(json_files)

    if total_files == 0:
        raise ValueError("No valid JSON files found in the specified folder.")

    for json_path in json_files:
        with open(json_path, "r") as file:
            data = json.load(file)

        # Analyze the last entry for success
        last_entry = data[-1]
        success = 1 if last_entry.get("packet_loss", -1) == 0 else 0

        # Count total successes
        total_success += success

        # Calculate average elapsed time and iterations
        elapsed_times = [entry["elapsed_time"] for entry in data if "elapsed_time" in entry]
        average_time = sum(elapsed_times) / len(elapsed_times)

        # Check if the file is safe (packet_loss is non-increasing)
        packet_losses = [entry["packet_loss"] for entry in data if "packet_loss" in entry]
        is_safe = all(x >= y for x, y in zip(packet_losses, packet_losses[1:]))
        total_safe_files += 1 if is_safe else 0

        # Calculate packet loss reduction
        if len(packet_losses) >= 2:
            initial_packet_loss = packet_losses[0]
            final_packet_loss = packet_losses[-1]
            packet_loss_reduction = initial_packet_loss - final_packet_loss
            total_packet_loss_reduction += packet_loss_reduction
            total_files_with_reduction += 1

        total_average_time += average_time
        total_iterations += len(elapsed_times)

    # Calculate overall metrics
    success_rate = total_success / total_files
    overall_average_time = total_average_time / total_files
    average_iterations = total_iterations / total_files
    safety_rate = total_safe_files / total_files

    # Calculate average packet loss reduction
    average_packet_loss_reduction = (
        total_packet_loss_reduction / total_files_with_reduction
        if total_files_with_reduction > 0 else 0
    )

    # Prepare the summary
    summary = {
        "success_rate": round(success_rate, 2),
        "overall_average_time": round(overall_average_time, 2),
        "average_iterations": round(average_iterations, 2),
        "safety_rate": round(safety_rate, 2),
        "average_packet_loss_reduction": round(average_packet_loss_reduction, 2)
    }

    # Write the summary to the output file
    with open(output_file, "w") as out_file:
        json.dump(summary, out_file, indent=4)



def plot_metrics(result_dir, error_types):
    """
    Plot success rates, safety rates, and average iterations for different error types.

    Parameters:
        result_dir (str): Directory containing the result JSON files.
        error_types (list): List of error types to plot.
    """
    success_rates = []
    safety_rates = []
    average_iterations = []

    for error_type in error_types:
        json_path = os.path.join(result_dir, error_type, f'{error_type}_result.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            success_rates.append(data.get('success_rate', 0))
            safety_rates.append(data.get('safety_rate', 0))
            average_iterations.append(data.get('average_iterations', 0))

    # Plot success rates
    plt.figure(figsize=(12, 6))
    plt.bar(error_types, success_rates, color='blue')
    plt.xlabel('Error Type')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Error Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'success_rate.png'))
    plt.close()

    # Plot safety rates
    plt.figure(figsize=(12, 6))
    plt.bar(error_types, safety_rates, color='green')
    plt.xlabel('Error Type')
    plt.ylabel('Safety Rate')
    plt.title('Safety Rate by Error Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'safety_rate.png'))
    plt.close()

    # Plot average iterations
    plt.figure(figsize=(12, 6))
    plt.bar(error_types, average_iterations, color='red')
    plt.xlabel('Error Type')
    plt.ylabel('Average Iterations')
    plt.title('Average Iterations by Error Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_iterations.png'))
    plt.close()

def plot_combined_error_metrics(result_dir, error_combinations):
    """
    Plot success rates, safety rates, and average iterations for combined error types.

    Parameters:
        result_dir (str): Directory containing the result JSON files.
        error_combinations (list): List of tuples containing combined error types.
    """
    success_rates = []
    safety_rates = []
    average_iterations = []
    error_types = ['disable_routing', 'disable_interface', 'remove_ip', 'drop_traffic_to_from_subnet', 'wrong_routing_table']

    for error_type in error_types:
        json_path = os.path.join(result_dir, 'result', error_type, f'{error_type}_result.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            success_rates.append(data.get('success_rate', 0))
            safety_rates.append(data.get('safety_rate', 0))
            average_iterations.append(data.get('average_iterations', 0))

    result_dir = os.path.join(result_dir, 'result', 'combined_test_results')
    for i, (error1, error2) in enumerate(error_combinations):
        json_path = os.path.join(result_dir, f'test_{i+1}', f'result_{i+1}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            # Ignore the last entry
            data = data[:-1]

            success_rate = sum(1 for entry in data if entry.get('packet_loss', 0) == 0) / len(data)
            safety_rate = all(data[i]['packet_loss'] <= data[i-1]['packet_loss'] for i in range(1, len(data)))
            average_iteration = len(data)
            success_rates.append(success_rate)
            safety_rates.append(1 if safety_rate else 0)
            average_iterations.append(average_iteration)

    # Combine error types and error combinations for labels
    labels = error_types + [f'{error1} + {error2}' for error1, error2 in error_combinations]

    # Plot success rates
    plt.figure(figsize=(10, 6))
    plt.bar(labels, success_rates, color='skyblue')
    plt.xlabel('Error Combinations')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(success_rates) * 1.1)  # Adjust y-axis limit
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'success_rate.png'), dpi=300)
    plt.close()

    # Plot safety rates
    plt.figure(figsize=(10, 6))
    plt.bar(labels, safety_rates, color='green')
    plt.xlabel('Error Combinations')
    plt.ylabel('Safety Rate')
    plt.title('Safety Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(safety_rates) * 1.1)  # Adjust y-axis limit
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'safety_rate.png'), dpi=300)
    plt.close()

    # Plot average iterations
    plt.figure(figsize=(10, 6))
    plt.bar(labels, average_iterations, color='orange')
    plt.xlabel('Error Combinations')
    plt.ylabel('Average Iterations')
    plt.title('Average Iterations by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(average_iterations) * 1.1)  # Adjust y-axis limit
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_iterations.png'), dpi=300)
    plt.close()

    # Combine all three plots into one figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Success rates
    axs[0].bar(labels, success_rates, color='skyblue')
    axs[0].set_ylabel('Success Rate')
    axs[0].set_title('Success Rate by Error Combinations')
    axs[0].set_ylim(0, min(max(success_rates) * 1.5, 1))  # Adjust y-axis limit

    # Safety rates
    axs[1].bar(labels, safety_rates, color='green')
    axs[1].set_ylabel('Safety Rate')
    axs[1].set_title('Safety Rate by Error Combinations')
    axs[1].set_ylim(0, min(max(safety_rates) * 1.5, 1))  # Adjust y-axis limit

    # Average iterations
    axs[2].bar(labels, average_iterations, color='orange')
    axs[2].set_xlabel('Error Combinations')
    axs[2].set_ylabel('Average Iterations')
    axs[2].set_title('Average Iterations by Error Combinations')
    axs[2].set_ylim(0, min(max(average_iterations) * 1.5, 15))  # Adjust y-axis limit

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # fig.suptitle('Reuslts for Qwen Model', fontsize=16)
    plt.savefig(os.path.join(result_dir, 'combined_metrics.png'), dpi=300)
    plt.close()


def static_plot_metrics(root_dir):
    labels = []
    success_rates = []
    safety_rates = []
    average_iterations = []  # Using average_iterations from the JSON data

    # Iterate over each subdirectory in the root_dir
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
            if os.path.exists(json_result_path):
                try:
                    with open(json_result_path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {json_result_path}: {e}")
                    continue

                labels.append(subdir)
                success_rates.append(data.get("success_rate", 0))
                safety_rates.append(data.get("safety_rate", 0))
                average_iterations.append(data.get("average_iterations", 0))
            else:
                print(f"File {json_result_path} does not exist.")

    if not labels:
        print("No valid data found. Exiting plotting function.")
        return

    # Create an output directory for the plots
    result_dir = os.path.join(root_dir, "plots")
    os.makedirs(result_dir, exist_ok=True)

    # -------------------------------------------
    # Individual plots for each metric (optional)
    # -------------------------------------------
    # Plot success rates
    plt.figure(figsize=(10, 6))
    plt.bar(labels, success_rates, color='skyblue')
    plt.xlabel('Error Combinations')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(success_rates) * 1.1 if success_rates else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'success_rate.png'), dpi=300)
    plt.close()

    # Plot safety rates
    plt.figure(figsize=(10, 6))
    plt.bar(labels, safety_rates, color='green')
    plt.xlabel('Error Combinations')
    plt.ylabel('Safety Rate')
    plt.title('Safety Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(safety_rates) * 1.1 if safety_rates else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'safety_rate.png'), dpi=300)
    plt.close()

    # Plot average iterations
    plt.figure(figsize=(10, 6))
    plt.bar(labels, average_iterations, color='orange')
    plt.xlabel('Error Combinations')
    plt.ylabel('Average Iterations')
    plt.title('Average Iterations by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(average_iterations) * 1.1 if average_iterations else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_iterations.png'), dpi=300)
    plt.close()

    # -------------------------------------------
    # Combine all three plots into one figure with subplots
    # -------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Success Rate subplot
    axs[0].bar(labels, success_rates, color='skyblue')
    axs[0].set_ylabel('Success Rate')
    axs[0].set_title('Success Rate by Error Combinations')
    axs[0].set_ylim(0, min(max(success_rates) * 1.5, 1) if success_rates else 1)

    # Safety Rate subplot
    axs[1].bar(labels, safety_rates, color='green')
    axs[1].set_ylabel('Safety Rate')
    axs[1].set_title('Safety Rate by Error Combinations')
    axs[1].set_ylim(0, min(max(safety_rates) * 1.5, 1) if safety_rates else 1)

    # Average Iterations subplot
    axs[2].bar(labels, average_iterations, color='orange')
    axs[2].set_xlabel('Error Combinations')
    axs[2].set_ylabel('Average Iterations')
    axs[2].set_title('Average Iterations by Error Combinations')
    axs[2].set_ylim(0, min(max(average_iterations) * 1.5, 15) if average_iterations else 1)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    combined_path = os.path.join(result_dir, 'combined_metrics.png')
    plt.savefig(combined_path, dpi=300)
    plt.close()

    print(f"Combined plot saved to: {combined_path}")

def process_results(save_result_path):
    """
    Process all promptagent folders under the specified path, extract commands, packet_loss, etc.,
    and merge them with the content of error_config.json to save the results.

    Args:
        save_result_path (str): Root directory path.
    """
    # Load the shared error_config.json
    global_error_config_path = os.path.join(save_result_path, "error_config.json")
    if not os.path.exists(global_error_config_path):
        raise FileNotFoundError(f"Global error_config.json not found in {save_result_path}")

    with open(global_error_config_path, "r") as f:
        global_error_config = json.load(f)

    # Get the queries list
    queries = global_error_config.get("queries", [])
    if not queries:
        raise ValueError("No queries found in error_config.json")

    # Iterate through each promptagent folder
    for promptagent in os.listdir(save_result_path):
        promptagent_path = os.path.join(save_result_path, promptagent)
        if not os.path.isdir(promptagent_path):
            continue

        # Initialize the results list
        results = [None] * len(queries)  # Store results in order

        # Iterate through subfolders
        for subfolder in os.listdir(promptagent_path):
            subfolder_path = os.path.join(promptagent_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Look for txt files starting with "result"
            for file in os.listdir(subfolder_path):
                if file.startswith("result") and file.endswith(".txt"):
                    txt_path = os.path.join(subfolder_path, file)
                    json_path = txt_path.replace(".txt", ".json")

                    # Parse the file index
                    try:
                        query_index = int(file.split("_")[1].split(".")[0]) - 1
                    except (IndexError, ValueError):
                        print(f"Invalid file name format: {file}")
                        continue

                    # Check if query_index is out of range
                    if query_index < 0 or query_index >= len(queries):
                        print(f"Query index {query_index} out of range for queries in error_config.json (file: {file})")
                        continue

                    # Read commands
                    with open(txt_path, "r") as txt_file:
                        commands = []
                        machines = []  # To store the values of Machine:
                        safe_count = 0
                        total_count = 0  # Total number of queries
                        for line in txt_file:
                            if line.startswith("=================================================="):
                                # Skip the next line
                                next(txt_file, None)
                                # Extract Machine value
                                machine_line = next(txt_file, "").strip()
                                machines.append(machine_line.replace("Machine:", "").strip())
                                # Extract Commands value
                                command_line = next(txt_file, "").strip()
                                commands.append(command_line.replace("Commands:", "").strip())
                                # Check the result line
                                result_line = next(txt_file, "").strip()
                                total_count += 1
                                if not result_line.startswith("Pingall result:"):
                                    safe_count += 1

                    # Read the JSON file
                    if not os.path.exists(json_path):
                        print(f"JSON file not found for {txt_path}")
                        continue

                    with open(json_path, "r") as json_file:
                        data = json.load(json_file)

                    # Extract packet_loss
                    packet_losses = [entry.get("packet_loss", -1) for entry in data]

                    # Determine success 
                    success = 1 if packet_losses[-1] == 0 else 0
                    
                    # Calculate the new safe value as a ratio
                    if total_count > 0:
                        if success == 1:
                            safe = (1 + safe_count) / total_count
                        else:
                            safe = safe_count / total_count
                        # if safe > 1, set to 1
                        safe = min(safe, 1)
                    else:
                        safe = 0

                    # Get the corresponding query
                    error_detail = queries[query_index]

                    # Save the result to the corresponding index
                    results[query_index] = {
                        "commands": commands,
                        "machines": machines,  # Store the values of Machine:
                        "packet_loss": packet_losses,
                        "success": success,
                        "safe": safe,
                        "detail": error_detail
                    }

        # Remove unprocessed entries (if some queries do not have corresponding result files)
        results = [result for result in results if result is not None]

        # Save results as a JSON file
        output_path = os.path.join(save_result_path, f"{promptagent}.json")
        with open(output_path, "w") as output_file:
            json.dump(results, output_file, indent=4)
        print(f"Results saved to {output_path}")

def plot_results(save_result_path, sample_num):
    """
    Plot the success rate and safety rate for each promptagent with publication-ready styling.
    
    Args:
        save_result_path (str): Root directory path.
        sample_num (int): Number of samples to select for each error type.
    """
    summary_results = {}

    # Iterate through each promptagent folder
    for promptagent in os.listdir(save_result_path):
        promptagent_path = os.path.join(save_result_path, promptagent)
        if not os.path.isdir(promptagent_path):
            continue

        # Load the result JSON file
        result_path = os.path.join(save_result_path, f"{promptagent}.json")
        if not os.path.exists(result_path):
            print(f"Result JSON not found for {promptagent}")
            continue

        with open(result_path, "r") as f:
            results = json.load(f)

        # Group by error type
        errortype_groups = {}
        for result in results:
            errortype = result["detail"]["errortype"]

            # If errortype is a list, convert it to a string or tuple
            if isinstance(errortype, list):
                errortype = tuple(errortype)  # Convert to tuple to ensure hashability

            if errortype not in errortype_groups:
                errortype_groups[errortype] = []
            errortype_groups[errortype].append(result)

        # Select only the top sample_num results for each error type
        filtered_results = []
        for errortype, group in errortype_groups.items():
            filtered_results.extend(group[:sample_num])
        print(f"Filtered results for {promptagent}: {len(filtered_results)} entries")

        # Statistics
        total_queries = len(filtered_results)
        total_success = sum(1 for result in filtered_results if result["success"] == 1)
        total_safe = sum(1 for result in filtered_results if result["safe"] == 1)

        # Calculate success rate and safety rate
        success_rate = (total_success / total_queries) * 100 if total_queries > 0 else 0
        safety_rate = (total_safe / total_queries) * 100 if total_queries > 0 else 0

        # Calculate standard error (SEM)
        success_binary_outcomes = [1] * total_success + [0] * (total_queries - total_success)
        safety_binary_outcomes = [1] * total_safe + [0] * (total_queries - total_safe)

        success_sem = stats.sem(success_binary_outcomes, ddof=0) * 100 if len(success_binary_outcomes) > 1 else 0
        safety_sem = stats.sem(safety_binary_outcomes, ddof=0) * 100 if len(safety_binary_outcomes) > 1 else 0

        # Calculate 95% confidence intervals
        success_margin = 1.96 * success_sem
        safety_margin = 1.96 * safety_sem

        # Save statistics
        summary_results[promptagent] = {
            "success_rate": success_rate,
            "safety_rate": safety_rate,
            "success_margin": success_margin,
            "safety_margin": safety_margin
        }
        print(f"Processed {promptagent}: Success Rate = {success_rate:.2f}%, Safety Rate = {safety_rate:.2f}%,success_margin = {success_margin:.2f}, safety_margin = {safety_margin:.2f}")
    
    # Create figure with higher DPI and specific size
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Professional color palette - Option 1: Scientific color scheme
    colors = ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67']
    
    # Plot each point
    for i, (folder, folder_stats) in enumerate(summary_results.items()):
        x = folder_stats["safety_rate"] / 100
        y = folder_stats["success_rate"] / 100
        x_err = folder_stats["safety_margin"] / 100
        y_err = folder_stats["success_margin"] / 100
        
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
    
    # Set labels and title with improved fonts
    ax.set_xlabel("Safety Rate", fontsize=20, fontweight='bold')
    ax.set_ylabel("Success Rate", fontsize=20, fontweight='bold')
    # ax.set_title(f"Success vs. Safety Analysis\n(Top {sample_num} samples per error type)",
    #              fontsize=20, 
    #              fontweight='bold', 
    #              pad=20)

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
    output_image_path = os.path.join(save_result_path, f"summary_plot_top_{sample_num}.png")
    plt.savefig(output_image_path, 
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

    print(f"Plot saved to {output_image_path}")

def plot_spider_charts(save_result_path, sample_num):
    """
    Create spider/radar charts for safety and success rates based on error types,
    with different agents plotted on the same chart for comparison.
    
    Args:
        save_result_path (str): Root directory path.
        sample_num (int): Number of samples to select for each error type.
    """

    # Error type abbreviation mapping
    error_abbrev = {
        "disable_routing": "level-1,DR",
        "disable_interface": "level-1,DI",
        "remove_ip": "level-1,RI",
        "drop_traffic_to_from_subnet": "level-1,DT",
        "wrong_routing_table": "level-1,WR",
        # For combined error types
        "disable_routing+disable_interface": "level-2,DR+DI",
        "disable_routing+remove_ip": "level-2,DR+RI",
        "disable_routing+drop_traffic_to_from_subnet": "level-3,DR+DT",
        "disable_routing+wrong_routing_table": "level-2,DR+WR",
        "disable_interface+remove_ip": "level-3,DI+RI",
        "disable_interface+drop_traffic_to_from_subnet": "level-2,DI+DT",
        "disable_interface+wrong_routing_table": "level-3,DI+WR",
        "remove_ip+drop_traffic_to_from_subnet": "level-3,RI+DT",
        "remove_ip+wrong_routing_table": "level-2,RI+WR",
        "drop_traffic_to_from_subnet+wrong_routing_table": "level-2,DT+WR"
    }

    # Set global plotting style
    # plt.style.use('seaborn-v0_8-white')  # Clean, professional base style
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
    
    # Process each promptagent folder
    for promptagent in os.listdir(save_result_path):
        promptagent_path = os.path.join(save_result_path, promptagent)
        if not os.path.isdir(promptagent_path):
            continue
            
        # Load the result JSON file
        result_path = os.path.join(save_result_path, f"{promptagent}.json")
        if not os.path.exists(result_path):
            print(f"Result JSON not found for {promptagent}")
            continue
            
        with open(result_path, "r") as f:
            results = json.load(f)
            
        # Initialize agent results
        if promptagent not in agent_results:
            agent_results[promptagent] = {}
            
        # Group results by error type for this agent
        for result in results:
            errortype = result["detail"]["errortype"]
            if isinstance(errortype, list):
                errortype = "+".join(errortype)  # Convert list to string
                
            if errortype not in agent_results[promptagent]:
                agent_results[promptagent][errortype] = {
                    "success": [],
                    "safety": []
                }
                
            agent_results[promptagent][errortype]["success"].append(result["success"])
            agent_results[promptagent][errortype]["safety"].append(result["safe"])
    
    # Get all unique error types
    all_error_types = set()
    for agent_data in agent_results.values():
        all_error_types.update(agent_data.keys())
    categories = sorted(list(all_error_types))
    
    # Create abbreviated category labels
    category_labels = [error_abbrev.get(cat, cat) for cat in categories]
    
    # Create two separate spider charts (success and safety)
    for metric in ["Success Rate", "Safety Rate"]:
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
                        rate = np.mean(agent_data[errortype]["success"]) * 100
                    else:  # Safety Rate
                        rate = np.mean(agent_data[errortype]["safety"]) * 100
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
        output_path = os.path.join(save_result_path, f"route_spider_chart_{metric.lower().replace(' ', '_')}_by_agent")
        
        # # Save as PDF for better quality
        # plt.savefig(f"{output_path}.pdf", 
        #             dpi=300, 
        #             bbox_inches='tight',
        #             pad_inches=0.2)
        
        # Also save as PNG for easy viewing
        plt.savefig(f"{output_path}.png",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.2)
        
        plt.close()
        
        print(f"Spider chart for {metric} by agent saved to {output_path}.pdf and {output_path}.png")
        
        # Print the abbreviation mapping for reference
        print("\nError Type Abbreviations:")
        for cat, abbrev in zip(categories, category_labels):
            print(f"{abbrev}: {cat}")

if __name__ == "__main__":
    save_result_path = "/home/ubuntu/nemo_benchmark/app-route/result/GPT-Agent/agenttest/20250421-182502"
    # for subdir in os.listdir(result_path):
    #     subdir_path = os.path.join(result_path, subdir)
    #     if os.path.isdir(subdir_path):
    #         json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
    #         static_summarize_results(subdir_path, json_result_path)

    # static_plot_metrics(result_path)
    process_results(save_result_path)
    plot_results(save_result_path, 10)
    plot_results(save_result_path, 50)
    plot_results(save_result_path, 150)
    # plot_spider_charts(save_result_path, 150)
    plot_spider_charts(save_result_path, 150)