import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Usage: python plot_cpp_benchmark.py /path/to/cpp_benchmark_results_*.csv

def plot_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    
    # Check if this is an average file (has StdDev columns)
    is_average = 'Write_Time_ms_Avg' in df.columns
    
    if is_average:
        # Plot average results with error bars
        block_sizes = df['Block_Size_MB'].tolist()  # This column now contains "16KB", "64KB", etc.
        block_dim_0 = df['Block_Dim_0'].tolist()
        block_dim_1 = df['Block_Dim_1'].tolist()
        write_times = df['Write_Time_ms_Avg'].tolist()
        read_times = df['Read_Time_ms_Avg'].tolist()
        write_stddev = df['Write_Time_ms_StdDev'].tolist()
        read_stddev = df['Read_Time_ms_StdDev'].tolist()
        
        x = np.arange(len(block_sizes))
        width = 0.35
        plt.figure(figsize=(12, 6))
        
        # Plot bars with error bars
        plt.bar(x - width/2, write_times, width, label='Write', color='tab:red', 
                yerr=write_stddev, capsize=5, alpha=0.8)
        plt.bar(x + width/2, read_times, width, label='Read', color='tab:blue', 
                yerr=read_stddev, capsize=5, alpha=0.8)
        
        plt.xlabel('Block Size')
        plt.ylabel('Time (ms)')
        plt.title('N5 Performance by Block Size (Average with StdDev)')
        labels = [f"{size}\n({d0}x{d1})" for size, d0, d1 in zip(block_sizes, block_dim_0, block_dim_1)]
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Output path
        base_dir = os.path.dirname(csv_file)
        match = re.search(r'(\d+)MB', csv_file)
        if match:
            totalsize = match.group(1)
            # Check if this is an average file
            if 'average' in csv_file:
                out_path = os.path.join(base_dir, f"cpp_benchmark_results_avg_{totalsize}MB.png")
            else:
                # Extract run number for individual runs
                run_match = re.search(r'run(\d+)', csv_file)
                if run_match:
                    run_num = run_match.group(1)
                    out_path = os.path.join(base_dir, f"cpp_benchmark_results_run{run_num}_{totalsize}MB.png")
                else:
                    out_path = os.path.join(base_dir, f"cpp_benchmark_results_{totalsize}MB.png")
        else:
            out_path = csv_file.replace('.csv', '_plot.png')
        plt.savefig(out_path)
        print(f"Average plot saved to {out_path}")
        
    else:
        # Plot individual run results
        block_sizes = df['Block_Size_MB'].tolist()  # This column now contains "16KB", "64KB", etc.
        block_dim_0 = df['Block_Dim_0'].tolist()
        block_dim_1 = df['Block_Dim_1'].tolist()
        write_times = df['Write_Time_ms'].tolist()
        read_times = df['Read_Time_ms'].tolist()

        x = np.arange(len(block_sizes))
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, write_times, width, label='Write', color='tab:red')
        plt.bar(x + width/2, read_times, width, label='Read', color='tab:blue')
        plt.xlabel('Block Size')
        plt.ylabel('Time (ms)')
        plt.title('N5 Performance by Block Size')
        labels = [f"{size}\n({d0}x{d1})" for size, d0, d1 in zip(block_sizes, block_dim_0, block_dim_1)]
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Output path
        base_dir = os.path.dirname(csv_file)
        match = re.search(r'(\d+)MB', csv_file)
        if match:
            totalsize = match.group(1)
            # Extract run number for individual runs
            run_match = re.search(r'run(\d+)', csv_file)
            if run_match:
                run_num = run_match.group(1)
                out_path = os.path.join(base_dir, f"cpp_benchmark_results_run{run_num}_{totalsize}MB.png")
            else:
                out_path = os.path.join(base_dir, f"cpp_benchmark_results_{totalsize}MB.png")
        else:
            out_path = csv_file.replace('.csv', '_plot.png')
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_cpp_benchmark.py /path/to/cpp_benchmark_results_*.csv")
        sys.exit(1)
    csv_file = sys.argv[1]
    plot_from_csv(csv_file)

if __name__ == "__main__":
    main() 