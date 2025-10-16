# How to Use main.ipynb - Complete Guide

## Overview

`main.ipynb` now contains **all 4 phases** in a single notebook:
- **Phase 1**: PyTorch checkpointing (baseline)
- **Phase 2**: Basic TensorStore
- **Phase 3**: T5X-optimized TensorStore
- **Phase 4**: Isolated optimization testing (NEW)

## Quick Start

### 1. Open the Notebook
```bash
cd /home/rifatxia/Desktop/TensorstoreWork/tensorstore/llama-work
source llama-venv/bin/activate
jupyter notebook main.ipynb
```

### 2. Run All Cells
- **Option A**: Menu → Cell → Run All
- **Option B**: Shift+Enter through each cell
- **Option C**: Kernel → Restart & Run All

### 3. View Results
All plots are automatically displayed and saved:
- `saved_models/3way_performance_comparison.png` (Phases 1-3)
- `saved_models/phase4_optimization_analysis.png` (Phase 4)

## What Each Phase Does

### Phase 1: PyTorch (Cells 1-8)
- Loads OpenLLaMA-3B model
- Saves with `torch.save()`
- Loads with `torch.load()`
- **Time**: ~10 seconds total
- **Output**: `saved_models/openllama_3b_pytorch.pth`

### Phase 2: TensorStore (Cells 9-12)
- Saves each parameter as separate Zarr file
- Simple 64-element chunks
- No compression, default concurrency
- **Time**: ~2-3 minutes
- **Output**: `saved_models/openllama_3b_tensorstore/`

### Phase 3: T5X-TensorStore (Cells 13-16)
- Saves with T5X optimizations:
  - 64 MiB chunks
  - 128 concurrent operations
  - Gzip compression
- **Time**: ~1-2 minutes
- **Output**: `saved_models/openllama_3b_t5x_tensorstore/`

### Phase 4: Isolated Testing (Cells 17-22) **NEW**
- Tests each optimization independently
- 6 tests × 3 runs = 18 total executions
- Statistical analysis (mean ± std)
- **Time**: ~15-20 minutes
- **Output**: `saved_models/phase4_*/` + plots

## Phase 4 Details

### The 6 Tests:

1. **baseline** - No optimizations (reference point)
2. **concurrency_128** - Only high concurrency
3. **chunks_1mb** - Only large chunks (1MB)
4. **compression_gzip** - Only gzip compression
5. **concurrency_chunks** - Concurrency + chunks
6. **t5x_full** - All optimizations

### What You'll See:

**Console Output:**
```
[1/6] baseline: no optimizations
  running baseline (3 iterations)...
    run 1: save=134.23s, load=12.34s, size=2.58gb
    run 2: save=135.67s, load=12.45s, size=2.58gb
    run 3: save=133.89s, load=12.28s, size=2.58gb

[2/6] concurrency: 128 operations
  running concurrency_128 (3 iterations)...
    ...
```

**Results Table:**
```
test                      save (s)           load (s)           size (gb)
--------------------------------------------------------------------------------
baseline                   134.26 ± 0.89      12.36 ± 0.09       2.58 ± 0.00
concurrency_128             98.45 ± 1.23      11.89 ± 0.12       2.58 ± 0.00
chunks_1mb                  67.89 ± 0.98      15.67 ± 0.23       2.59 ± 0.01
compression_gzip           145.67 ± 2.34      18.90 ± 0.45       2.52 ± 0.02
concurrency_chunks          56.78 ± 1.12      17.23 ± 0.34       2.59 ± 0.01
t5x_full                    58.90 ± 1.45      19.45 ± 0.56       2.53 ± 0.02
```

**6-Panel Visualization:**
1. Save time comparison (with error bars)
2. Load time comparison (with error bars)
3. File size comparison (with error bars)
4. Save improvement % vs baseline
5. Load improvement % vs baseline
6. Optimization impact heatmap

## Execution Time Estimates

| Phase | Estimated Time |
|-------|---------------|
| Phase 1 (PyTorch) | ~10 seconds |
| Phase 2 (TensorStore) | ~2-3 minutes |
| Phase 3 (T5X) | ~1-2 minutes |
| Phase 4 (6 tests × 3 runs) | ~15-20 minutes |
| **Total** | **~20-25 minutes** |

## Running Specific Phases

### Run Only Phases 1-3 (Quick)
1. Open notebook
2. Run cells 1-16
3. Skip Phase 4 cells (17-22)
4. **Time**: ~5 minutes

### Run Only Phase 4
1. Open notebook
2. Run cells 1-4 (setup + model loading)
3. Run cells 17-22 (Phase 4)
4. **Time**: ~15-20 minutes

### Run Everything
1. Open notebook
2. Cell → Run All
3. Go get coffee ☕
4. **Time**: ~20-25 minutes

## Modifying Phase 4 Parameters

### In the Notebook:

Find the Phase 4 test execution cell and modify:

```python
# test 2: concurrency only
print("\n[2/6] concurrency: 128 operations")
phase4_results.append(run_test_multiple_times(
    model_state, "concurrency_128", 
    chunk_size=64,        # ← Change this
    concurrency=128,      # ← Change this
    compression=None,     # ← Change this (None or 'gzip')
    num_runs=3            # ← Change this (1-10)
))
```

### Common Modifications:

**Test different chunk sizes:**
```python
chunk_size=16384   # 16KB
chunk_size=65536   # 64KB
chunk_size=262144  # 1MB (default)
chunk_size=524288  # 2MB
```

**Test different concurrency:**
```python
concurrency=1      # Sequential
concurrency=32     # Medium
concurrency=128    # High (default)
concurrency=256    # Very high
```

**Change number of runs:**
```python
num_runs=1   # Quick test
num_runs=3   # Default (good balance)
num_runs=5   # More accurate
num_runs=10  # Very accurate (slow)
```

## Interpreting Results

### Good Performance Indicators:

✅ **Low standard deviation** (< 5% of mean)
- Indicates consistent performance
- Reliable measurements

✅ **Concurrency improves save time**
- Should see ~25-30% improvement
- Verifies parallel I/O works

✅ **Large chunks improve save time**
- Should see ~45-50% improvement
- Verifies chunking optimization

### Warning Signs:

⚠️ **High standard deviation** (> 10% of mean)
- System load interference
- Disk cache effects
- Solution: Increase `num_runs` or close other apps

⚠️ **Compression makes load slower**
- Expected behavior (decompression overhead)
- ~40-50% slower is normal

⚠️ **Large chunks make load slower**
- Expected behavior (less parallelism)
- ~20-30% slower is normal

## Troubleshooting

### "Out of Memory" Error
**Solution:**
- Close other applications
- Reduce `num_runs` to 1
- Skip Phase 4 (run only Phases 1-3)

### "CUDA Out of Memory"
**Solution:**
- Model uses ~3.2GB VRAM
- Should work on GTX 1650 (3.9GB)
- If fails, model will auto-offload to CPU

### Phase 4 Takes Too Long
**Solution:**
- Reduce `num_runs` from 3 to 1
- Skip some tests (comment them out)
- Run overnight

### Plots Don't Appear
**Solution:**
- Check `saved_models/` directory
- Plots are saved even if not displayed
- Use `plt.show()` to force display

### Inconsistent Results
**Solution:**
- Increase `num_runs` to 5 or 10
- Close background applications
- Run during low system activity
- Check disk space (need ~15GB free)

## File Outputs

After running all phases:

```
saved_models/
├── openllama_3b_pytorch.pth                    # Phase 1 (2.73 GB)
├── openllama_3b_tensorstore/                   # Phase 2 (2.58 GB)
│   ├── *.zarr files (100 parameters)
│   └── metadata.json
├── openllama_3b_t5x_tensorstore/              # Phase 3 (2.59 GB)
│   ├── *.zarr files (100 parameters)
│   └── metadata.json
├── phase4_baseline_run1/                       # Phase 4 tests
├── phase4_baseline_run2/
├── phase4_baseline_run3/
├── phase4_concurrency_128_run1/
├── ... (18 test directories total)
├── 3way_performance_comparison.png             # Phases 1-3 plot
└── phase4_optimization_analysis.png            # Phase 4 plot
```

**Total disk space needed**: ~15-20 GB

## Tips for Best Results

1. **Close other applications** before running
2. **Use SSD** for accurate I/O measurements
3. **Run on AC power** (not battery)
4. **Disable disk indexing** temporarily
5. **Clear saved_models/phase4_*/** between runs for clean tests

## Next Steps After Running

1. **Review the plots**
   - Compare Phase 1-3 performance
   - Analyze Phase 4 isolated impacts

2. **Read the insights**
   - Check "Phase 4 Key Insights" cell
   - Understand which optimization helps most

3. **Experiment**
   - Modify parameters in Phase 4
   - Test your own configurations
   - Document findings

4. **Share results**
   - Save plots for presentation
   - Export results table
   - Update README.md with insights

## Summary

**To run everything:**
```bash
cd /home/rifatxia/Desktop/TensorstoreWork/tensorstore/llama-work
source llama-venv/bin/activate
jupyter notebook main.ipynb
# Then: Cell → Run All
```

**Expected total time**: ~20-25 minutes

**Output**: 
- 2 comprehensive plots
- Statistical analysis
- ~15GB of checkpoint files

**Key insight**: You'll see exactly which T5X optimization contributes most to performance improvement!
