import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import re
from matplotlib.patches import Rectangle

def parse_benchmark_file(filename):
    """Parse the benchmark file and extract bandwidth data for all benchmark types"""
    benchmark_data = {}
    current_benchmark = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_chunk = None
    chunk_sizes = []
    stages_list = []
    
    for line in lines:
        line = line.strip()
        
        # Check for benchmark type headers
        if "TMA bulk" in line:
            current_benchmark = "TMA"
        elif "cp.async Bandwidth Test (1 Producer Warp)" in line:
            current_benchmark = "cp.async_1_warp"
        elif "cp.async Bandwidth Test (2 Producer Warps)" in line:
            current_benchmark = "cp.async_2_warps"  
        elif "cp.async Bandwidth Test (4 Producer Warps)" in line:
            current_benchmark = "cp.async_4_warps"
        
        # Initialize benchmark data structure if not exists
        if current_benchmark and current_benchmark not in benchmark_data:
            benchmark_data[current_benchmark] = {}
        
        # Check for chunk size header
        chunk_match = re.search(r'CHUNK=(\d+) B', line)
        if chunk_match and current_benchmark:
            current_chunk = int(chunk_match.group(1))
            if current_chunk not in chunk_sizes:
                chunk_sizes.append(current_chunk)
            benchmark_data[current_benchmark][current_chunk] = {}
            continue
        
        # Check for stage data line
        stage_match = re.search(r'Stages=\s*(\d+)\s*\|\s*Chunk=\s*(\d+)\s*\|\s*Time=([\d.]+)\s*ms\s*\|\s*BW=([\d.]+)\s*GB/s', line)
        if stage_match and current_chunk is not None and current_benchmark:
            stage = int(stage_match.group(1))
            bw = float(stage_match.group(4))
            if stage not in stages_list:
                stages_list.append(stage)
            benchmark_data[current_benchmark][current_chunk][stage] = bw
            continue
            
        # Check for "Out of SHMEM" cases
        out_of_shmem_match = re.search(r'Stages=\s*(\d+).*Out of SHMEM', line)
        if out_of_shmem_match and current_chunk is not None and current_benchmark:
            stage = int(out_of_shmem_match.group(1))
            if stage not in stages_list:
                stages_list.append(stage)
            benchmark_data[current_benchmark][current_chunk][stage] = np.nan
    
    # Sort chunk sizes and stages
    chunk_sizes.sort()
    stages_list.sort()
    
    # Create matrices for each benchmark
    result_data = {}
    for benchmark in benchmark_data:
        bw_matrix = np.full((len(chunk_sizes), len(stages_list)), np.nan)
        
        for i, chunk in enumerate(chunk_sizes):
            for j, stage in enumerate(stages_list):
                if chunk in benchmark_data[benchmark] and stage in benchmark_data[benchmark][chunk]:
                    bw_matrix[i, j] = benchmark_data[benchmark][chunk][stage]
        
        result_data[benchmark] = bw_matrix
    
    return chunk_sizes, stages_list, result_data

def main():
    parser = argparse.ArgumentParser(description='Plot TMA and cp.async benchmark results')
    parser.add_argument('-i', '--input', required=True, help='Input benchmark file')
    parser.add_argument('-o', '--output', required=True, help='Output PNG file')
    parser.add_argument('--peak-bw', type=float, default=5.6, help='Peak bandwidth in TB/s (default: 5.6)')
    
    args = parser.parse_args()
    
    # Parse the input file
    chunk_sizes, stages, benchmark_data = parse_benchmark_file(args.input)
    
    print(f"Parsed data: {len(chunk_sizes)} chunk sizes, {len(stages)} stages")
    print(f"Chunk sizes: {chunk_sizes}")
    print(f"Stages: {stages}")
    print(f"Benchmarks found: {list(benchmark_data.keys())}")
    
    # Convert to TB/s for utilization calculation
    peak_bw_tb = args.peak_bw

    # Create the visualization with 4x2 subplots
    # Each row represents a benchmark type, columns are bandwidth and utilization
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    
    # Define subplot titles and data mapping
    subplot_config = [
        ("TMA", "TMA"),
        ("cp.async_1_warp", "cp.async (1 Producer)"),
        ("cp.async_2_warps", "cp.async (2 Producers)"),
        ("cp.async_4_warps", "cp.async (4 Producers)")
    ]
    
    for row, (benchmark_key, title) in enumerate(subplot_config):
        if benchmark_key in benchmark_data:
            bw_matrix = benchmark_data[benchmark_key]
            util_matrix = bw_matrix / 1000 / peak_bw_tb * 100  # Convert GB/s to TB/s, then to percentage
            
            # Bandwidth heatmap - Column 0
            ax_bw = axes[row, 0]
            im_bw = ax_bw.imshow(bw_matrix, cmap='viridis', aspect='auto')
            ax_bw.set_xticks(range(len(stages)))
            ax_bw.set_xticklabels(stages)
            ax_bw.set_yticks(range(len(chunk_sizes)))
            ax_bw.set_yticklabels([f'{cs}B' for cs in chunk_sizes])
            ax_bw.set_xlabel('Number of Stages')
            ax_bw.set_ylabel('Chunk Size')
            ax_bw.set_title(f'{title} - Bandwidth (GB/s)')
            
            # Add text annotations for bandwidth
            for i in range(len(chunk_sizes)):
                for j in range(len(stages)):
                    if not np.isnan(bw_matrix[i, j]):
                        ax_bw.text(j, i, f'{bw_matrix[i, j]:.0f}', 
                                ha='center', va='center', fontweight='bold',
                                fontsize=10,
                                color='white' if bw_matrix[i, j] > np.nanmax(bw_matrix)/2 else 'black')
            
            # Add colorbar for bandwidth
            plt.colorbar(im_bw, ax=ax_bw, label='Bandwidth (GB/s)')
            
            # Utilization heatmap - Column 1
            ax_util = axes[row, 1]
            im_util = ax_util.imshow(util_matrix, cmap='plasma', aspect='auto', vmin=0, vmax=100)
            ax_util.set_xticks(range(len(stages)))
            ax_util.set_xticklabels(stages)
            ax_util.set_yticks(range(len(chunk_sizes)))
            ax_util.set_yticklabels([f'{cs}B' for cs in chunk_sizes])
            ax_util.set_xlabel('Number of Stages')
            ax_util.set_ylabel('Chunk Size')
            ax_util.set_title(f'{title} - Utilization (%)')
            
            # Add text annotations for utilization
            for i in range(len(chunk_sizes)):
                for j in range(len(stages)):
                    if not np.isnan(util_matrix[i, j]):
                        ax_util.text(j, i, f'{util_matrix[i, j]:.1f}%', 
                                ha='center', va='center', fontweight='bold',
                                fontsize=10,
                                color='white' if util_matrix[i, j] > 50 else 'black')
            
            # Add colorbar for utilization
            plt.colorbar(im_util, ax=ax_util, label='Utilization (%)')
            
        else:
            # If benchmark data not found, show empty plots with message
            for col in range(2):
                ax = axes[row, col]
                col_title = "Bandwidth" if col == 0 else "Utilization"
                ax.text(0.5, 0.5, f'No data found for\n{title}\n{col_title}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
                ax.set_title(f'{title} - {col_title}')
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    # plt.suptitle('Memory Bandwidth and Utilization Comparison', fontsize=16, y=0.98)
    
    # Save the figure
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output}")
    
    # Print summary statistics for each benchmark
    print("\nPerformance Summary:")
    for benchmark_key, title in subplot_config:
        if benchmark_key in benchmark_data:
            bw_matrix = benchmark_data[benchmark_key]
            peak_bw = np.nanmax(bw_matrix)
            peak_util = peak_bw / 1000 / peak_bw_tb * 100
            
            # Find best configuration
            best_idx = np.nanargmax(bw_matrix)
            best_chunk = chunk_sizes[best_idx // len(stages)]
            best_stage = stages[best_idx % len(stages)]
            
            print(f"\n{title}:")
            print(f"  Peak bandwidth: {peak_bw:.2f} GB/s")
            print(f"  Peak utilization: {peak_util:.1f}% of {peak_bw_tb} TB/s")
            print(f"  Best config: {best_chunk}B chunk, {best_stage} stages")

if __name__ == "__main__":
    main()