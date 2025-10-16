#!/usr/bin/env python3
"""
phase 4: isolated testing of t5x optimizations
tests each optimization independently with statistical analysis
"""

import torch
import time
import os
import numpy as np
import tensorstore as ts
import matplotlib.pyplot as plt
import json
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Dict, List, Tuple
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# configuration
# ============================================================================

class Config:
    """centralized configuration for testing"""
    MODEL_NAME = "openlm-research/open_llama_3b"
    SAVE_DIR = "saved_models/phase4_tests/"
    NUM_RUNS = 3  # run each test 3 times for statistical accuracy
    
    # optimization parameters to test
    CHUNK_SIZES = [64, 1024, 16384, 1048576]  # 64, 1KB, 16KB, 1MB in elements
    CONCURRENCY_LIMITS = [1, 8, 32, 128]
    COMPRESSION_OPTIONS = [None, 'gzip']

# ============================================================================
# core testing function
# ============================================================================

def save_load_tensorstore(
    model_state: Dict,
    test_name: str,
    chunk_size_elements: int = 64,
    concurrency_limit: int = 1,
    compression: str = None,
    run_number: int = 1
) -> Tuple[float, float, float]:
    """
    save and load model with specific tensorstore configuration
    
    returns: (save_time, load_time, file_size) in seconds and gb
    """
    
    save_dir = f"{Config.SAVE_DIR}{test_name}_run{run_number}/"
    os.makedirs(save_dir, exist_ok=True)
    
    # create tensorstore context with specified concurrency
    ts_context = ts.Context({'file_io_concurrency': {'limit': concurrency_limit}})
    
    # ========== save phase ==========
    start_time = time.time()
    
    saved_count = 0
    for param_name, param_tensor in model_state.items():
        if param_tensor.device.type == 'meta':
            continue
            
        param_np = param_tensor.detach().cpu().float().numpy()
        
        # calculate chunk shape based on target elements
        chunk_shape = calculate_chunk_shape(param_np.shape, chunk_size_elements)
        
        safe_name = param_name.replace('.', '_').replace('/', '_')
        
        # build tensorstore spec
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': f"{save_dir}{safe_name}.zarr"
            },
            'metadata': {
                'shape': list(param_np.shape),
                'dtype': '<f4',
                'chunks': chunk_shape
            }
        }
        
        # add compression if specified
        if compression:
            spec['metadata']['compressor'] = {'id': compression}
        
        # save with specified context
        store = ts.open(spec, create=True, delete_existing=True, context=ts_context).result()
        store.write(param_np).result()
        saved_count += 1
    
    save_time = time.time() - start_time
    
    # save metadata
    metadata = {
        'param_names': [k for k in model_state.keys() if model_state[k].device.type != 'meta'],
        'total_params': saved_count,
        'chunk_size_elements': chunk_size_elements,
        'concurrency_limit': concurrency_limit,
        'compression': compression
    }
    with open(f"{save_dir}metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    # calculate file size
    total_size = 0
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    file_size = total_size / (1024**3)
    
    # ========== load phase ==========
    start_time = time.time()
    
    with open(f"{save_dir}metadata.json", 'r') as f:
        metadata = json.load(f)
    
    loaded_state = {}
    for param_name in metadata['param_names']:
        safe_name = param_name.replace('.', '_').replace('/', '_')
        zarr_path = f"{save_dir}{safe_name}.zarr"
        
        if os.path.exists(zarr_path):
            spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': zarr_path
                }
            }
            
            store = ts.open(spec, context=ts_context).result()
            param_np = store.read().result()
            loaded_state[param_name] = torch.from_numpy(param_np.copy()).half()
    
    load_time = time.time() - start_time
    
    # cleanup
    del loaded_state
    gc.collect()
    
    return save_time, load_time, file_size

def calculate_chunk_shape(shape: List[int], target_elements: int) -> List[int]:
    """calculate optimal chunk shape for given target size"""
    if target_elements < 1:
        target_elements = 1
    if not shape:
        return [1]
    
    chunk_shape = list(shape)
    
    # iteratively reduce largest dimension
    while np.prod(chunk_shape) > target_elements and max(chunk_shape) > 1:
        max_idx = chunk_shape.index(max(chunk_shape))
        chunk_shape[max_idx] = max(1, chunk_shape[max_idx] // 2)
    
    return chunk_shape

# ============================================================================
# statistical analysis
# ============================================================================

def run_multiple_times(
    model_state: Dict,
    test_name: str,
    chunk_size: int,
    concurrency: int,
    compression: str,
    num_runs: int = 3
) -> Dict:
    """run test multiple times and compute statistics"""
    
    save_times = []
    load_times = []
    file_sizes = []
    
    print(f"\n  running {test_name} ({num_runs} iterations)...")
    
    for run in range(1, num_runs + 1):
        save_t, load_t, size = save_load_tensorstore(
            model_state, test_name, chunk_size, concurrency, compression, run
        )
        save_times.append(save_t)
        load_times.append(load_t)
        file_sizes.append(size)
        print(f"    run {run}/{num_runs}: save={save_t:.2f}s, load={load_t:.2f}s, size={size:.2f}gb")
    
    return {
        'test_name': test_name,
        'save_mean': np.mean(save_times),
        'save_std': np.std(save_times),
        'load_mean': np.mean(load_times),
        'load_std': np.std(load_times),
        'size_mean': np.mean(file_sizes),
        'size_std': np.std(file_sizes),
        'chunk_size': chunk_size,
        'concurrency': concurrency,
        'compression': compression
    }

# ============================================================================
# main testing workflow
# ============================================================================

def main():
    print("="*80)
    print("PHASE 4: ISOLATED T5X OPTIMIZATION TESTING")
    print("="*80)
    
    # setup
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\ndevice: {device}")
    
    # load model
    print(f"\nloading model: {Config.MODEL_NAME}")
    tokenizer = LlamaTokenizer.from_pretrained(Config.MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    model_state = {}
    for name, param in model.named_parameters():
        if param.device.type != 'meta':
            model_state[name] = param
    
    print(f"model loaded: {len(model_state)} parameters")
    
    # ========================================================================
    # test suite: isolate each optimization
    # ========================================================================
    
    results = []
    
    print("\n" + "="*80)
    print("TEST SUITE: ISOLATED OPTIMIZATIONS")
    print("="*80)
    
    # baseline: basic tensorstore (no optimizations)
    print("\n[1/6] baseline: basic tensorstore")
    results.append(run_multiple_times(
        model_state, "baseline", 
        chunk_size=64, concurrency=1, compression=None, num_runs=Config.NUM_RUNS
    ))
    
    # test 1: concurrency only
    print("\n[2/6] optimization: high concurrency (128)")
    results.append(run_multiple_times(
        model_state, "concurrency_128",
        chunk_size=64, concurrency=128, compression=None, num_runs=Config.NUM_RUNS
    ))
    
    # test 2: large chunks only
    print("\n[3/6] optimization: large chunks (1mb = 262144 elements)")
    results.append(run_multiple_times(
        model_state, "chunks_1mb",
        chunk_size=262144, concurrency=1, compression=None, num_runs=Config.NUM_RUNS
    ))
    
    # test 3: compression only
    print("\n[4/6] optimization: gzip compression")
    results.append(run_multiple_times(
        model_state, "compression_gzip",
        chunk_size=64, concurrency=1, compression='gzip', num_runs=Config.NUM_RUNS
    ))
    
    # test 4: concurrency + chunks (no compression)
    print("\n[5/6] combined: concurrency + large chunks")
    results.append(run_multiple_times(
        model_state, "concurrency_chunks",
        chunk_size=262144, concurrency=128, compression=None, num_runs=Config.NUM_RUNS
    ))
    
    # test 5: all optimizations (t5x full)
    print("\n[6/6] combined: all t5x optimizations")
    results.append(run_multiple_times(
        model_state, "t5x_full",
        chunk_size=262144, concurrency=128, compression='gzip', num_runs=Config.NUM_RUNS
    ))
    
    # ========================================================================
    # save results
    # ========================================================================
    
    with open(f"{Config.SAVE_DIR}results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'test':<25} {'save (s)':<15} {'load (s)':<15} {'size (gb)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['test_name']:<25} "
              f"{r['save_mean']:>6.2f}±{r['save_std']:>4.2f}   "
              f"{r['load_mean']:>6.2f}±{r['load_std']:>4.2f}   "
              f"{r['size_mean']:>5.2f}±{r['size_std']:>4.2f}")
    
    # ========================================================================
    # generate plots
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    generate_plots(results)
    
    print("\n✓ phase 4 complete!")
    print(f"  results saved to: {Config.SAVE_DIR}results.json")
    print(f"  plots saved to: {Config.SAVE_DIR}optimization_analysis.png")

# ============================================================================
# visualization
# ============================================================================

def generate_plots(results: List[Dict]):
    """generate comprehensive 6-way comparison plots"""
    
    fig = plt.figure(figsize=(20, 12))
    
    test_names = [r['test_name'] for r in results]
    save_means = [r['save_mean'] for r in results]
    save_stds = [r['save_std'] for r in results]
    load_means = [r['load_mean'] for r in results]
    load_stds = [r['load_std'] for r in results]
    size_means = [r['size_mean'] for r in results]
    size_stds = [r['size_std'] for r in results]
    
    # colors for each test
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # ========== plot 1: save time with error bars ==========
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(range(len(test_names)), save_means, yerr=save_stds, 
                   capsize=5, color=colors, alpha=0.8)
    ax1.set_title('save time comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('time (seconds)', fontsize=12)
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # add value labels
    for i, (mean, std) in enumerate(zip(save_means, save_stds)):
        ax1.text(i, mean + std + max(save_means)*0.02, 
                f'{mean:.1f}s', ha='center', fontsize=9, fontweight='bold')
    
    # ========== plot 2: load time with error bars ==========
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(range(len(test_names)), load_means, yerr=load_stds,
                   capsize=5, color=colors, alpha=0.8)
    ax2.set_title('load time comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('time (seconds)', fontsize=12)
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(load_means, load_stds)):
        ax2.text(i, mean + std + max(load_means)*0.02,
                f'{mean:.1f}s', ha='center', fontsize=9, fontweight='bold')
    
    # ========== plot 3: file size with error bars ==========
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.bar(range(len(test_names)), size_means, yerr=size_stds,
                   capsize=5, color=colors, alpha=0.8)
    ax3.set_title('file size comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('size (gb)', fontsize=12)
    ax3.set_xticks(range(len(test_names)))
    ax3.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(size_means, size_stds)):
        ax3.text(i, mean + std + max(size_means)*0.02,
                f'{mean:.2f}gb', ha='center', fontsize=9, fontweight='bold')
    
    # ========== plot 4: save time improvement vs baseline ==========
    ax4 = plt.subplot(2, 3, 4)
    baseline_save = save_means[0]
    improvements = [(baseline_save - s) / baseline_save * 100 for s in save_means]
    bars = ax4.bar(range(len(test_names)), improvements, color=colors, alpha=0.8)
    ax4.set_title('save time improvement vs baseline', fontsize=14, fontweight='bold')
    ax4.set_ylabel('improvement (%)', fontsize=12)
    ax4.set_xticks(range(len(test_names)))
    ax4.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(axis='y', alpha=0.3)
    
    for i, imp in enumerate(improvements):
        color = 'green' if imp > 0 else 'red'
        ax4.text(i, imp + (5 if imp > 0 else -5),
                f'{imp:+.1f}%', ha='center', fontsize=9, fontweight='bold', color=color)
    
    # ========== plot 5: load time improvement vs baseline ==========
    ax5 = plt.subplot(2, 3, 5)
    baseline_load = load_means[0]
    improvements = [(baseline_load - l) / baseline_load * 100 for l in load_means]
    bars = ax5.bar(range(len(test_names)), improvements, color=colors, alpha=0.8)
    ax5.set_title('load time improvement vs baseline', fontsize=14, fontweight='bold')
    ax5.set_ylabel('improvement (%)', fontsize=12)
    ax5.set_xticks(range(len(test_names)))
    ax5.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax5.grid(axis='y', alpha=0.3)
    
    for i, imp in enumerate(improvements):
        color = 'green' if imp > 0 else 'red'
        ax5.text(i, imp + (5 if imp > 0 else -5),
                f'{imp:+.1f}%', ha='center', fontsize=9, fontweight='bold', color=color)
    
    # ========== plot 6: optimization impact heatmap ==========
    ax6 = plt.subplot(2, 3, 6)
    
    # create impact matrix
    metrics = ['save time\nimprovement', 'load time\nimprovement', 'size\nreduction']
    impact_matrix = []
    
    for r in results:
        save_imp = (baseline_save - r['save_mean']) / baseline_save * 100
        load_imp = (baseline_load - r['load_mean']) / baseline_load * 100
        size_red = (size_means[0] - r['size_mean']) / size_means[0] * 100
        impact_matrix.append([save_imp, load_imp, size_red])
    
    impact_matrix = np.array(impact_matrix).T
    
    im = ax6.imshow(impact_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    ax6.set_xticks(range(len(test_names)))
    ax6.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax6.set_yticks(range(len(metrics)))
    ax6.set_yticklabels(metrics, fontsize=10)
    ax6.set_title('optimization impact heatmap', fontsize=14, fontweight='bold')
    
    # add text annotations
    for i in range(len(metrics)):
        for j in range(len(test_names)):
            text = ax6.text(j, i, f'{impact_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    plt.colorbar(im, ax=ax6, label='improvement (%)')
    
    plt.tight_layout()
    plt.savefig(f'{Config.SAVE_DIR}optimization_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ plots saved to: {Config.SAVE_DIR}optimization_analysis.png")

if __name__ == "__main__":
    main()
