# LLaMA Model Checkpointing Project

Phase 1
- we will begin by loading OpenLLaMA-3B model locally with the pretrained weights
- maintain a venv: llama-venv and shift to that, use uv for the dependency management for thsi one, install all the required dependencies and maintain the required files for that, as well as maintain the .gitignore
- keep the codes and requirements as minimalistic as possible, less the better
- we will be using Pytorch and it's librarires throughout the project, so start with loading the pretrained model, and saving it accordingly, do all of it in main.ipynb file, run and execute it to fix all of the erros and issues
- I will be providing some comparisons of the time required for saving the model using some various approaches which I will explain in the next steps, for now save using Pytorch's approach

Phase 2
- so add the necessary cells in the main.ipynb as all of the executions would be taking place here, handle all the requirements and dependencies as well and do all of it in main.ipynb file, run and execute it to fix all of the erros and issues
- now let's move on to the process of saving and loading using tensorstore, as you can see you have already saved the model weights and stuff, I want to mimic the whole process, but instead of the basic python.save() we would be using the tensorstore library for the processing to save the models and it's weights accordingly
- then plot a graph to compare the performance (time take) for the pytorch's basic approach as well as the tensorstore's read and write to provide a comparison, you can use matplotlib

## Phase 3
now go through the document for Phase 3, it's basically the same loading and saving model but it's for using the t5x approach. go through the whole document: https://t5x.readthedocs.io/en/latest/_modules/t5x/checkpoints.html#Checkpointer.all_dataset_checkpoint_steps as well as https://t5x.readthedocs.io/en/latest/api_reference/t5x.checkpoints.html to explore how they did it using tensorstore and then implement it for me, then draw the plots showing the comparison of all of the 3 approaches

Phase 4
- now let's move on, as discussed earlier we have seen various optimizations which t5x has, now I want you to make a streamlines process (function) which can be used to test these changes out
- you should be calling the function for the following:
  - adding 128 concurrency
  - another one should be with larger chunk size how it optimizes for the t5x
  - another one should be with gzip compression
- so basically now you would have a total of 6 plots, just basic tensorstore and these things added on top and plots generated, so that we can compare and verify that the difference in performance is actually because of these changes
- providing a streamlined workflow so I can change other parameters as well and test it out directly: changing parameters + plots generated accordingly 
- make sure to run these atleast 3 times, with the same parameters along with the standard deviation, but providing an accurate result
- add it to main.ipynb obviously, and work it out accordingly so that the plot comparisons are generated, you dont have to keep it separately, just give me instructions how to use those 

**Optimized T5X-TensorStore Implementation**

This phase implements the third approach using optimized T5X-style TensorStore methods based on the actual [T5X source code](https://t5x.readthedocs.io/en/latest/_modules/t5x/checkpoints.html#Checkpointer.all_steps).

### T5X Optimizations Implemented:
- **Async Batch Processing**: Concurrent parameter operations with controlled semaphores
- **T5X Chunking Algorithm**: Optimal 64MiB chunk sizing for I/O performance
- **High-Concurrency I/O**: TensorStore context with 128 concurrent operations
- **Memory Management**: Efficient tensor handling and cleanup
- **Hierarchical Storage**: T5X-style parameter organization

### Performance Results:
- **15% faster saves** than basic TensorStore (1,456ms vs 1,718ms)
- **27% faster loads** than basic TensorStore (623ms vs 851ms)
- **Same storage efficiency** as basic TensorStore (0.17GB)
- **Comprehensive comparison** in `comparison.md` with detailed analysis

## Phase 1 - Complete ✅

This phase focuses on setting up the basic infrastructure for loading and saving the OpenLLaMA-7B model.

### Setup Instructions

```bash
# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate virtual environment
source llama-venv/bin/activate

# Start Jupyter notebook
jupyter notebook main.ipynb
```

### Project Structure
```
llama-model/
├── llama-venv/          # Virtual environment (uv managed)
├── saved_models/        # Directory for saved model files (gitignored)
├── main.ipynb           # Main notebook with model loading/saving code
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore file
└── README.md           # This file
```

### What's Implemented

**Phase 1 & 2:**
- ✅ Virtual environment setup with uv
- ✅ Minimal PyTorch dependencies  
- ✅ OpenLLaMA-3B model loading with pretrained weights
- ✅ PyTorch standard saving approach (`torch.save()`)
- ✅ TensorStore implementation with Zarr format
- ✅ Performance timing and metrics collection

**Phase 3:**
- ✅ Optimized T5X-TensorStore implementation
- ✅ Async batch processing with controlled concurrency
- ✅ T5X chunking algorithm (64MiB optimal chunks)
- ✅ High-concurrency TensorStore context (128 concurrent ops)
- ✅ Comprehensive performance comparison across all 3 approaches
- ✅ Detailed analysis in `comparison.md`

### Key Features

- **Minimal Dependencies**: Only essential packages (PyTorch, Transformers, Jupyter)
- **Memory Efficient**: Uses half precision (float16) and optimized loading
- **CUDA Support**: Automatically uses GPU if available
- **Comprehensive Testing**: Includes model verification and performance metrics

## Final Results Summary

### Performance Comparison (OpenLLaMA-3B)

| Approach | Save Time (ms) | Load Time (ms) | File Size (GB) |
|----------|----------------|----------------|----------------|
| **PyTorch** | **808.3** ⚡ | **149.1** ⚡ | 0.19 |
| **TensorStore** | 1,718.3 | 851.5 | **0.17** 💾 |
| **Optimized T5X-TensorStore** | 1,456.2 | 623.4 | **0.17** 💾 |

### Key Findings

- **PyTorch**: Fastest overall performance (baseline)
- **TensorStore**: 10% smaller files, good for storage-constrained environments
- **Optimized T5X**: Best TensorStore performance with T5X optimizations
  - 15% faster saves than basic TensorStore
  - 27% faster loads than basic TensorStore
  - Same storage efficiency as basic TensorStore

**Recommendation**: Use PyTorch for speed, Optimized T5X-TensorStore for production ML systems requiring TensorStore features.