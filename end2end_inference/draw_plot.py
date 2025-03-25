# Copyright 2025 The SpInfer Authors. All rights reserved.
# Copyright 2023 The FLash-LLM Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

import re

def parse_our_flash_ft_log(file_path):
    """
    parse the log file of our/flash-llm/ft, extract the inference time (ms).
    
    Args:
    file_path (str): the path of the log file
    
    Returns:
    float: inference time (ms), return None if not found
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # match the time value after FT-CPP-decoding-beamsearch-time
            match = re.search(r'FT-CPP-decoding-beamsearch-time\s*([\d.]+)\s*ms', content)
            if match:
                return float(match.group(1))
            else:
                print(f"Warning: Could not find 'FT-CPP-decoding-beamsearch-time' in {file_path}")
                return None
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
        return None

def parse_ds_log(file_path):
    """
    parse the log file of ds, extract the inference time (ms).
    
    Args:
    file_path (str): the path of the log file
    
    Returns:
    float: inference time (ms), return None if not found
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # match the time value after generation time is
            match = re.search(r'generation time is\s*([\d.]+)\s*sec', content)
            if match:
                # convert seconds to milliseconds
                return float(match.group(1)) * 1000
            else:
                print(f"Warning: Could not find 'generation time is' in {file_path}")
                return None
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
        return None
import os

def load_data(base_path, output_lengths, batch_sizes, log_parser):
    """
    load data from the log file.
    
    Args:
    base_path (str): the base path of the log file
    output_lengths (list): the list of output lengths
    batch_sizes (list): the list of batch sizes
    log_parser (function): the log parser function
    
    Returns:
    list: the same data format as the data
    """
    data = []
    for i, output_length in enumerate(output_lengths):
        row = []
        for j, batch_size in enumerate(batch_sizes):
            log_file = f"{base_path}/output_batch_{batch_size}_output_len_{output_length}.log"
            latency = log_parser(log_file)
            row.append(latency)
        data.append(row)
    return data

# define the paths
SpInfer_HOME = os.getenv('SpInfer_HOME')
FlashLLM_HOME = os.getenv('FlashLLM_HOME')
FT_HOME = os.getenv('FT_HOME')

# data
batch_sizes = [8, 16, 32]
output_lengths = [64, 128, 256, 512, 1024]

# load the data of our/flash-llm/ft
our_1gpu_13B = load_data(f"{SpInfer_HOME}/third_party/FasterTransformer/Result_13B/1-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)
our_2gpus_13B = load_data(f"{SpInfer_HOME}/third_party/FasterTransformer/Result_13B/2-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)
our_2gpus = load_data(f"{SpInfer_HOME}/third_party/FasterTransformer/Result_30B/2-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)
our_4gpus = load_data(f"{SpInfer_HOME}/third_party/FasterTransformer/Result_30B/4-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)

flash_llm_1gpu_13B = load_data(f"{FlashLLM_HOME}/third_party/FasterTransformer/Result_13B/1-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)
flash_llm_2gpus_13B = load_data(f"{FlashLLM_HOME}/third_party/FasterTransformer/Result_13B/2-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)
flash_llm_2gpus = load_data(f"{FlashLLM_HOME}/third_party/FasterTransformer/Result_30B/2-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)
flash_llm_4gpus = load_data(f"{FlashLLM_HOME}/third_party/FasterTransformer/Result_30B/4-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)

ft_2gpus_13B = load_data(f"{FT_HOME}/FasterTransformer/Result_13B/2-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)
ft_4gpus = load_data(f"{FT_HOME}/FasterTransformer/Result_13B/2-gpu", output_lengths, batch_sizes, parse_our_flash_ft_log)

# load the data of ds
ds_2gpus_13B = load_data(f"{SpInfer_HOME}/end2end_inference/ds_scripts/ds_result/2-gpu", output_lengths, batch_sizes, parse_ds_log)
ds_4gpus = load_data(f"{SpInfer_HOME}/end2end_inference/ds_scripts/ds_result/4-gpu", output_lengths, batch_sizes, parse_ds_log)

# modified compute_throughput function
def compute_throughput(batch_size, output_length, latency, num_gpus):
    if latency is None:
        return None
    return (batch_size * output_length) / (latency * num_gpus) * 1000

# modified compute_y_range function
def compute_y_range(*datasets, num_gpus):
    y_values = []
    for data in datasets:
        for i, output_length in enumerate(output_lengths):
            y_values += [
                compute_throughput(bs, output_length, data[i][j], num_gpus) 
                for j, bs in enumerate(batch_sizes)
            ]
    y_min = np.nanmin([v for v in y_values if v is not None])
    y_max = np.nanmax([v for v in y_values if v is not None])
    return y_min, y_max

# Calculate Y-axis ranges for each row
y_min_13B_1gpu, y_max_13B_1gpu = compute_y_range(our_1gpu_13B, flash_llm_1gpu_13B, num_gpus=1)
y_min_13B_2gpus, y_max_13B_2gpus = compute_y_range(our_2gpus_13B, flash_llm_2gpus_13B, ft_2gpus_13B, ds_2gpus_13B, num_gpus=2)
y_min_30B_2gpus, y_max_30B_2gpus = compute_y_range(our_2gpus, flash_llm_2gpus, num_gpus=2)
y_min_30B_4gpus, y_max_30B_4gpus = compute_y_range(our_4gpus, flash_llm_4gpus, ft_4gpus, ds_4gpus, num_gpus=4)

# Generate 4x5 plot array with extra space on top for the legend
fig = plt.figure(figsize=(25, 22))  # Made figure slightly taller

# Create a gridspec that reserves space for the legend at the top
gs = fig.add_gridspec(5, 5, height_ratios=[0.2, 1, 1, 1, 1])

# Adjust font and marker sizes
font_size = 30
marker_size = 18
tick_size = 28
legend_size = font_size + 4

# Create the main axes array (4x5)
axes = [[fig.add_subplot(gs[i+1, j]) for j in range(5)] for i in range(4)]

# Create legend at the top
legend_ax = fig.add_subplot(gs[0, :])
legend_handles = [
    mlines.Line2D([], [], color=f'C{i}', marker='o', markersize=marker_size, label=label, linestyle='-')
    for i, label in enumerate(['SpInfer (ours)', 'Flash-LLM', 'FasterTransformer', 'DeepSpeed'])
]
legend_ax.legend(handles=legend_handles, loc='center', ncol=4, fontsize=legend_size)
legend_ax.axis('off')

# Plot each subplot
for i, output_length in enumerate(output_lengths):
    # OPT-13B (1 GPU)
    ax = axes[0][i]
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, our_1gpu_13B[i][j], 1) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, flash_llm_1gpu_13B[i][j], 1) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.set_title(f'Output Length {output_length}', fontsize=font_size + 2)
    
    if i == 0:
        ax.set_ylabel('#Tokens/s', fontsize=font_size+2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xticks(batch_sizes)
    ax.set_ylim([y_min_13B_1gpu * 0.9, y_max_13B_1gpu * 1.1])
    if i == 0:
        ax.text(0.1, 0.5, '1 GPU', fontsize=font_size + 2, va='center', ha='center', rotation='vertical', transform=ax.transAxes, fontweight='bold')
        ax.text(0.5, 0.05, 'OPT-13B', fontsize=font_size, va='bottom', ha='left', transform=ax.transAxes, fontweight='bold')

    # OPT-13B (2 GPUs)
    ax = axes[1][i]
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, our_2gpus_13B[i][j], 2) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, flash_llm_2gpus_13B[i][j], 2) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, ft_2gpus_13B[i][j], 2) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, ds_2gpus_13B[i][j], 2) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    if i == 0:
        ax.set_ylabel('#Tokens/s', fontsize=font_size+2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xticks(batch_sizes)
    ax.set_ylim([y_min_13B_2gpus * 0.9, y_max_13B_2gpus * 1.1])
    if i == 0:
        ax.text(0.1, 0.5, '2 GPUs', fontsize=font_size + 2, va='center', ha='center', rotation='vertical', transform=ax.transAxes, fontweight='bold')
        ax.text(0.5, 0.05, 'OPT-13B', fontsize=font_size, va='bottom', ha='left', transform=ax.transAxes, fontweight='bold')

    # OPT-30B (2 GPUs)
    ax = axes[2][i]
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, our_2gpus[i][j], 2) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, flash_llm_2gpus[i][j], 2) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    if i == 0:
        ax.set_ylabel('#Tokens/s', fontsize=font_size+2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xticks(batch_sizes)
    ax.set_ylim([y_min_30B_2gpus * 0.9, y_max_30B_2gpus * 1.1])
    if i == 0:
        ax.text(0.1, 0.5, '2 GPUs', fontsize=font_size + 2, va='center', ha='center', rotation='vertical', transform=ax.transAxes, fontweight='bold')
        ax.text(0.5, 0.05, 'OPT-30B', fontsize=font_size, va='bottom', ha='left', transform=ax.transAxes, fontweight='bold')

    # OPT-30B (4 GPUs)
    ax = axes[3][i]
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, our_4gpus[i][j], 4) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, flash_llm_4gpus[i][j], 4) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, ft_4gpus[i][j], 4) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    ax.plot(batch_sizes, [compute_throughput(bs, output_length, ds_4gpus[i][j], 4) for j, bs in enumerate(batch_sizes)], marker='o', markersize=marker_size)
    if i == 0:
        ax.set_ylabel('#Tokens/s', fontsize=font_size+2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xticks(batch_sizes)
    ax.set_ylim([y_min_30B_4gpus * 0.9, y_max_30B_4gpus * 1.1])
    if i == 0:
        ax.text(0.1, 0.5, '4 GPUs', fontsize=font_size + 2, va='center', ha='center', rotation='vertical', transform=ax.transAxes, fontweight='bold')
        ax.text(0.5, 0.05, 'OPT-30B', fontsize=font_size, va='bottom', ha='left', transform=ax.transAxes, fontweight='bold')
    
    # Add "Batch Size" label to bottom row
    ax.set_xlabel('Batch Size', fontsize=font_size+2)

# Adjust layout and save
plt.tight_layout()
plt.savefig('throughput_comparison_OPT-13B_30B_4x5_tokens_per_sec_with_internal_gpu_labels.png', dpi=300, bbox_inches='tight')
plt.close()
