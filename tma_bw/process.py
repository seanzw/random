import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import re
from matplotlib.patches import Rectangle

def parse_benchmark_file(filename):
    """Parse the benchmark file and extract bandwidth data"""
    chunk_sizes = []
    stages_list = []
    bw_data = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_chunk = None
    for line in lines:
        line = line.strip()
        
        # Check for chunk size header
        chunk_match = re.search(r'CHUNK=(\d+) B', line)
        if chunk_match:
            current_chunk = int(chunk_match.group(1))
            if current_chunk not in chunk_sizes:
                chunk_sizes.append(current_chunk)
            bw_data[current_chunk] = {}
            continue
        
        # Check for stage data line
        stage_match = re.search(r'Stages=\s*(\d+)\s*\|\s*Chunk=\s*(\d+)\s*\|\s*Time=([\d.]+)\s*ms\s*\|\s*BW=([\d.]+)\s*GB/s', line)
        if stage_match and current_chunk is not None:
            stage = int(stage_match.group(1))
            bw = float(stage_match.group(4))
            if stage not in stages_list:
                stages_list.append(stage)
            bw_data[current_chunk][stage] = bw
            continue
            
        # Check for "Out of SHMEM" cases
        out_of_shmem_match = re.search(r'Stages=\s*(\d+).*Out of SHMEM', line)
        if out_of_shmem_match and current_chunk is not None:
            stage = int(out_of_shmem_match.group(1))
            if stage not in stages_list:
                stages_list.append(stage)
            bw_data[current_chunk][stage] = np.nan
    
    # Sort chunk sizes and stages
    chunk_sizes.sort()
    stages_list.sort()
    
    # Create matrix
    bw_matrix = np.full((len(chunk_sizes), len(stages_list)), np.nan)
    
    for i, chunk in enumerate(chunk_sizes):
        for j, stage in enumerate(stages_list):
            if chunk in bw_data and stage in bw_data[chunk]:
                bw_matrix[i, j] = bw_data[chunk][stage]
    
    return chunk_sizes, stages_list, bw_matrix

def main():
    parser = argparse.ArgumentParser(description='Plot TMA benchmark results')
    parser.add_argument('-i', '--input', required=True, help='Input benchmark file')
    parser.add_argument('-o', '--output', required=True, help='Output PNG file')
    parser.add_argument('--peak-bw', type=float, default=5.6, help='Peak bandwidth in TB/s (default: 5.6)')
    
    args = parser.parse_args()
    
    # Parse the input file
    chunk_sizes, stages, bw_matrix = parse_benchmark_file(args.input)
    
    print(f"Parsed data: {len(chunk_sizes)} chunk sizes, {len(stages)} stages")
    print(f"Chunk sizes: {chunk_sizes}")
    print(f"Stages: {stages}")
    
    # Convert to TB/s for utilization calculation
    peak_bw_tb = args.peak_bw
    bw_util_matrix = bw_matrix / 1000 / peak_bw_tb  # Convert GB/s to TB/s and divide by peak

    # In-flight bytes matrix
    inflight_matrix = np.zeros_like(bw_matrix)
    for i, chunk in enumerate(chunk_sizes):
        for j, stage in enumerate(stages):
            if not np.isnan(bw_matrix[i, j]):
                inflight_matrix[i, j] = chunk * stage

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Bandwidth heatmap
    im1 = ax1.imshow(bw_matrix, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages)
    ax1.set_yticks(range(len(chunk_sizes)))
    ax1.set_yticklabels([f'{cs}B' for cs in chunk_sizes])
    ax1.set_xlabel('Number of Stages')
    ax1.set_ylabel('Chunk Size')
    ax1.set_title('Bandwidth (GB/s) - Hopper TMA Bulk Transfer')

    # Add text annotations for bandwidth
    for i in range(len(chunk_sizes)):
        for j in range(len(stages)):
            if not np.isnan(bw_matrix[i, j]):
                ax1.text(j, i, f'{bw_matrix[i, j]:.0f}', 
                        ha='center', va='center', fontweight='bold',
                        fontsize=16,
                        color='white' if bw_matrix[i, j] > np.nanmax(bw_matrix)/2 else 'black')

    # Plot 2: Bandwidth utilization heatmap
    im2 = ax2.imshow(bw_util_matrix, cmap='plasma', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages)
    ax2.set_yticks(range(len(chunk_sizes)))
    ax2.set_yticklabels([f'{cs}B' for cs in chunk_sizes])
    ax2.set_xlabel('Number of Stages')
    ax2.set_ylabel('Chunk Size')
    ax2.set_title(f'Bandwidth Utilization (Fraction of {peak_bw_tb} TB/s)')

    # Add text annotations for utilization and in-flight bytes
    for i in range(len(chunk_sizes)):
        for j in range(len(stages)):
            if not np.isnan(bw_util_matrix[i, j]):
                util_percent = bw_util_matrix[i, j] * 100
                inflight_kb = inflight_matrix[i, j] / 1024
                text = f'{util_percent:.1f}%\n({inflight_kb:.0f}KB)'
                ax2.text(j, i, text, 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color='white' if bw_util_matrix[i, j] > 0.5 else 'black')

    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Bandwidth (GB/s)')
    plt.colorbar(im2, ax=ax2, label='Utilization Fraction')

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output}")
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Peak bandwidth achieved: {np.nanmax(bw_matrix):.2f} GB/s")
    print(f"Peak utilization: {np.nanmax(bw_util_matrix)*100:.1f}% of {peak_bw_tb} TB/s")
    
    # Find best configuration
    best_idx = np.nanargmax(bw_matrix)
    best_chunk = chunk_sizes[best_idx // len(stages)]
    best_stage = stages[best_idx % len(stages)]
    print(f"Best configuration: {best_chunk}B chunk, {best_stage} stages")

if __name__ == "__main__":
    main()