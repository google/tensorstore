# Phase 4: Isolated T5X Optimization Testing

## Overview

This phase isolates and measures each T5X optimization independently to verify their individual impact on performance.

## Files Created

1. **`phase4_optimization_testing.py`** - Core testing framework with statistical analysis
2. **`phase4_config.json`** - Configuration file for easy parameter modification
3. **`run_phase4_streamlined.py`** - Streamlined runner using config file

## Quick Start

### Option 1: Run with Default Configuration
```bash
cd /home/rifatxia/Desktop/TensorstoreWork/tensorstore/llama-work
source llama-venv/bin/activate
python run_phase4_streamlined.py
```

### Option 2: Run Directly
```bash
python phase4_optimization_testing.py
```

## Test Suite (6 Tests)

Each test runs **3 times** with statistical analysis (mean ± std):

1. **baseline** - Basic TensorStore (no optimizations)
   - Chunk size: 64 elements
   - Concurrency: 1
   - Compression: None

2. **concurrency_128** - High concurrency only
   - Chunk size: 64 elements
   - Concurrency: 128 ✓
   - Compression: None

3. **chunks_1mb** - Large chunks only
   - Chunk size: 262,144 elements (1MB) ✓
   - Concurrency: 1
   - Compression: None

4. **compression_gzip** - Compression only
   - Chunk size: 64 elements
   - Concurrency: 1
   - Compression: gzip ✓

5. **concurrency_chunks** - Combined (no compression)
   - Chunk size: 262,144 elements ✓
   - Concurrency: 128 ✓
   - Compression: None

6. **t5x_full** - All T5X optimizations
   - Chunk size: 262,144 elements ✓
   - Concurrency: 128 ✓
   - Compression: gzip ✓

## Modifying Parameters

Edit `phase4_config.json`:

```json
{
  "testing": {
    "num_runs": 3,  // Change number of runs
    "save_dir": "saved_models/phase4_tests/"
  },
  
  "test_configurations": [
    {
      "name": "my_custom_test",
      "chunk_size_elements": 16384,  // Modify chunk size
      "concurrency_limit": 64,        // Modify concurrency
      "compression": "gzip"           // null or "gzip"
    }
  ]
}
```

Then run:
```bash
python run_phase4_streamlined.py
```

## Output

### Results File
`saved_models/phase4_tests/results.json` - Complete statistical data

### Visualization
`saved_models/phase4_tests/optimization_analysis.png` - 6-panel plot:
1. Save time comparison (with error bars)
2. Load time comparison (with error bars)
3. File size comparison (with error bars)
4. Save time improvement vs baseline (%)
5. Load time improvement vs baseline (%)
6. Optimization impact heatmap

### Console Output
```
test                      save (s)           load (s)           size (gb)
--------------------------------------------------------------------------------
baseline                   134.23 ± 2.45      12.34 ± 0.56       2.58 ± 0.01
concurrency_128             98.45 ± 1.89      11.23 ± 0.45       2.58 ± 0.01
chunks_1mb                  67.89 ± 1.23      15.67 ± 0.78       2.59 ± 0.01
compression_gzip           145.67 ± 2.34      18.90 ± 0.89       2.52 ± 0.02
concurrency_chunks          56.78 ± 1.45      17.23 ± 0.67       2.59 ± 0.01
t5x_full                    58.90 ± 1.67      19.45 ± 0.98       2.53 ± 0.02
```

## Expected Insights

### Concurrency Impact
- **Save time**: ~25-30% improvement
- **Load time**: ~5-10% improvement
- **File size**: No change
- **Why**: Parallel I/O operations

### Large Chunks Impact
- **Save time**: ~45-50% improvement
- **Load time**: ~20-30% slower
- **File size**: Minimal change
- **Why**: Fewer I/O operations, but less read parallelism

### Compression Impact
- **Save time**: ~5-10% slower
- **Load time**: ~40-50% slower
- **File size**: ~2-5% reduction
- **Why**: CPU overhead for compression/decompression

### Combined Impact
- **Save time**: ~55-60% improvement (best)
- **Load time**: ~30-40% slower
- **File size**: ~2-5% reduction
- **Why**: Synergistic effect of concurrency + chunks

## Adding Custom Tests

Enable custom tests in `phase4_config.json`:

```json
{
  "custom_tests": {
    "enabled": true,
    "configurations": [
      {
        "name": "test_64_concurrency",
        "chunk_size_elements": 262144,
        "concurrency_limit": 64,
        "compression": null
      },
      {
        "name": "test_512kb_chunks",
        "chunk_size_elements": 131072,
        "concurrency_limit": 128,
        "compression": "gzip"
      }
    ]
  }
}
```

## Key Functions

### `save_load_tensorstore()`
Core function that saves/loads with specific configuration:
```python
save_time, load_time, file_size = save_load_tensorstore(
    model_state,
    test_name="my_test",
    chunk_size_elements=262144,
    concurrency_limit=128,
    compression='gzip',
    run_number=1
)
```

### `run_multiple_times()`
Runs test N times and computes statistics:
```python
results = run_multiple_times(
    model_state,
    test_name="my_test",
    chunk_size=262144,
    concurrency=128,
    compression='gzip',
    num_runs=3
)
# Returns: {save_mean, save_std, load_mean, load_std, size_mean, size_std}
```

### `calculate_chunk_shape()`
T5X-style chunking algorithm:
```python
chunk_shape = calculate_chunk_shape(
    shape=[32000, 3200],
    target_elements=262144
)
# Returns: [4000, 3200] - optimal chunks
```

## Workflow for Testing New Parameters

1. Edit `phase4_config.json`
2. Run `python run_phase4_streamlined.py`
3. Check `saved_models/phase4_tests/results.json`
4. View `saved_models/phase4_tests/optimization_analysis.png`
5. Iterate

## Statistical Rigor

- Each test runs **3 times** (configurable)
- Reports **mean ± standard deviation**
- Error bars on all plots
- Identifies outliers through std deviation
- Ensures reproducible results

## Performance Tips

- **GPU recommended** for faster model loading
- **SSD recommended** for accurate I/O measurements
- **Close other applications** for consistent results
- **Run during low system load** for accuracy

## Troubleshooting

**High standard deviation?**
- Increase `num_runs` in config
- Close background applications
- Check disk activity

**Out of memory?**
- Set `use_gpu: false` in config
- Reduce number of concurrent tests
- Clear saved_models/phase4_tests/ between runs

**Tests too slow?**
- Reduce `num_runs` to 1 for quick testing
- Use smaller model for prototyping
- Skip compression tests (slowest)
