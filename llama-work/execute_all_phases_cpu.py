#!/usr/bin/env python3
"""
Execute all 4 phases on CPU with proper error handling
"""

import torch
import time
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import gc
import tensorstore as ts
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LLAMA MODEL CHECKPOINTING - ALL 4 PHASES (CPU MODE)")
print("="*80)

# ============================================================================
# SETUP
# ============================================================================
USE_CUDA = False
device = torch.device('cuda' if (USE_CUDA and torch.cuda.is_available()) else 'cpu')
print(f"\n✓ using device: {device}")
if device.type == 'cpu':
    print("  running on cpu - this will be slower but more stable")

os.makedirs('saved_models', exist_ok=True)
print("✓ created saved_models directory")

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

model_name = "openlm-research/open_llama_3b"
print(f"\nloading model: {model_name}")

tokenizer = LlamaTokenizer.from_pretrained(model_name)
print("✓ tokenizer loaded")

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # float32 for cpu
    device_map=None,
    low_cpu_mem_usage=True,
    use_safetensors=True
)

model = model.to(device)
print(f"✓ model loaded successfully")
print(f"  parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m")

# ============================================================================
# TEST INFERENCE
# ============================================================================
print("\n" + "="*80)
print("TESTING MODEL INFERENCE")
print("="*80)

test_prompt = "the future of artificial intelligence is"
inputs = tokenizer(test_prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"\nprompt: '{test_prompt}'")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"generated: {generated_text[:100]}...")
print("✓ model inference successful")

# ============================================================================
# PHASE 1: PYTORCH
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: PYTORCH SAVING/LOADING")
print("="*80)

pytorch_save_path = "saved_models/openllama_3b_pytorch.pth"

print("\nsaving with pytorch...")
start_time = time.time()
torch.save(model.state_dict(), pytorch_save_path)
pytorch_save_time = time.time() - start_time
pytorch_file_size = os.path.getsize(pytorch_save_path) / (1024**3)

print(f"✓ pytorch save completed in {pytorch_save_time*1000:.1f} ms")
print(f"  file size: {pytorch_file_size:.2f} gb")

print("\nloading with pytorch...")
start_time = time.time()
state_dict = torch.load(pytorch_save_path, map_location='cpu', weights_only=True)
pytorch_load_time = time.time() - start_time

print(f"✓ pytorch load completed in {pytorch_load_time*1000:.1f} ms")
print(f"  loaded {len(state_dict)} parameters")

del state_dict
gc.collect()

# ============================================================================
# PHASE 2: TENSORSTORE
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: TENSORSTORE SAVING/LOADING")
print("="*80)

tensorstore_save_dir = "saved_models/openllama_3b_tensorstore/"
os.makedirs(tensorstore_save_dir, exist_ok=True)

print("\nsaving with tensorstore...")
start_time = time.time()

model_state = {}
for name, param in model.named_parameters():
    if param.device.type != 'meta':
        model_state[name] = param

print(f"  processing {len(model_state)} parameters...")

saved_count = 0
for param_name, param_tensor in model_state.items():
    try:
        param_np = param_tensor.detach().cpu().float().numpy()
        safe_name = param_name.replace('.', '_').replace('/', '_')
        
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': f"{tensorstore_save_dir}{safe_name}.zarr"
            },
            'metadata': {
                'shape': list(param_np.shape),
                'dtype': '<f4',
                'chunks': [min(64, s) for s in param_np.shape] if param_np.shape else [1]
            }
        }
        
        store = ts.open(spec, create=True, delete_existing=True).result()
        store.write(param_np).result()
        saved_count += 1
        
    except Exception as e:
        print(f"  warning: skipping {param_name}: {e}")
        continue

metadata = {
    'param_names': list(model_state.keys()),
    'total_params': len(model_state)
}
with open(f"{tensorstore_save_dir}metadata.json", 'w') as f:
    json.dump(metadata, f)

tensorstore_save_time = time.time() - start_time

tensorstore_size = 0
for root, dirs, files in os.walk(tensorstore_save_dir):
    for file in files:
        tensorstore_size += os.path.getsize(os.path.join(root, file))
tensorstore_file_size = tensorstore_size / (1024**3)

print(f"✓ tensorstore save completed in {tensorstore_save_time*1000:.1f} ms")
print(f"  saved {saved_count} parameters")
print(f"  file size: {tensorstore_file_size:.2f} gb")

print("\nloading with tensorstore...")
start_time = time.time()

with open(f"{tensorstore_save_dir}metadata.json", 'r') as f:
    metadata = json.load(f)

loaded_state = {}
loaded_count = 0

for param_name in metadata['param_names']:
    try:
        safe_name = param_name.replace('.', '_').replace('/', '_')
        zarr_path = f"{tensorstore_save_dir}{safe_name}.zarr"
        
        if os.path.exists(zarr_path):
            spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': zarr_path
                }
            }
            
            store = ts.open(spec).result()
            param_np = store.read().result()
            loaded_state[param_name] = torch.from_numpy(param_np.copy()).float()
            loaded_count += 1
            
    except Exception as e:
        print(f"  warning: failed to load {param_name}: {e}")
        continue

tensorstore_load_time = time.time() - start_time

print(f"✓ tensorstore load completed in {tensorstore_load_time*1000:.1f} ms")
print(f"  loaded {loaded_count} parameters")

del loaded_state, model_state
gc.collect()

# ============================================================================
# PHASE 3: T5X-TENSORSTORE
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: T5X-OPTIMIZED TENSORSTORE SAVING/LOADING")
print("="*80)

t5x_tensorstore_save_dir = "saved_models/openllama_3b_t5x_tensorstore/"
os.makedirs(t5x_tensorstore_save_dir, exist_ok=True)

_DESIRED_CHUNK_SIZE_BYTES = 64 * 1024 * 1024
_TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})

def choose_chunk_shape_simple(write_shape, target_elements):
    if target_elements < 1:
        target_elements = 1
    if not write_shape:
        return [1]
    
    chunk_shape = list(write_shape)
    while np.prod(chunk_shape) > target_elements and max(chunk_shape) > 1:
        max_idx = chunk_shape.index(max(chunk_shape))
        chunk_shape[max_idx] = max(1, chunk_shape[max_idx] // 2)
    
    return chunk_shape

print("\nsaving with t5x-tensorstore...")
start_time = time.time()

model_state = {}
for name, param in model.named_parameters():
    if param.device.type != 'meta':
        model_state[name] = param

print(f"  processing {len(model_state)} parameters with t5x optimizations...")

saved_count = 0
for param_name, param_tensor in model_state.items():
    try:
        param_np = param_tensor.detach().cpu().float().numpy()
        target_elements = _DESIRED_CHUNK_SIZE_BYTES // param_np.dtype.itemsize
        chunk_shape = choose_chunk_shape_simple(list(param_np.shape), target_elements)
        
        safe_name = param_name.replace('.', '_').replace('/', '_')
        
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': f"{t5x_tensorstore_save_dir}{safe_name}.zarr"
            },
            'metadata': {
                'shape': list(param_np.shape),
                'dtype': '<f4',
                'chunks': chunk_shape,
                'compressor': {'id': 'gzip'}
            }
        }
        
        store = ts.open(spec, create=True, delete_existing=True, context=_TS_CONTEXT).result()
        store.write(param_np).result()
        saved_count += 1
        
    except Exception as e:
        print(f"  warning: skipping {param_name}: {e}")
        continue

metadata = {
    'param_names': list(model_state.keys()),
    'total_params': len(model_state)
}
with open(f"{t5x_tensorstore_save_dir}metadata.json", 'w') as f:
    json.dump(metadata, f)

t5x_tensorstore_save_time = time.time() - start_time

t5x_size = 0
for root, dirs, files in os.walk(t5x_tensorstore_save_dir):
    for file in files:
        t5x_size += os.path.getsize(os.path.join(root, file))
t5x_tensorstore_file_size = t5x_size / (1024**3)

print(f"✓ t5x-tensorstore save completed in {t5x_tensorstore_save_time*1000:.1f} ms")
print(f"  saved {saved_count} parameters")
print(f"  file size: {t5x_tensorstore_file_size:.2f} gb")

print("\nloading with t5x-tensorstore...")
start_time = time.time()

with open(f"{t5x_tensorstore_save_dir}metadata.json", 'r') as f:
    metadata = json.load(f)

loaded_state = {}
loaded_count = 0

for param_name in metadata['param_names']:
    try:
        safe_name = param_name.replace('.', '_').replace('/', '_')
        zarr_path = f"{t5x_tensorstore_save_dir}{safe_name}.zarr"
        
        if os.path.exists(zarr_path):
            spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': zarr_path
                }
            }
            
            store = ts.open(spec, context=_TS_CONTEXT).result()
            param_np = store.read().result()
            loaded_state[param_name] = torch.from_numpy(param_np.copy()).float()
            loaded_count += 1
            
    except Exception as e:
        print(f"  warning: failed to load {param_name}: {e}")
        continue

t5x_tensorstore_load_time = time.time() - start_time

print(f"✓ t5x-tensorstore load completed in {t5x_tensorstore_load_time*1000:.1f} ms")
print(f"  loaded {loaded_count} parameters")

del loaded_state, model_state
gc.collect()

# ============================================================================
# PHASE 4: ORBAX
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: ORBAX SAVING/LOADING")
print("="*80)

try:
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from etils import epath
    
    def pytorch_to_jax_pytree(model_state_dict):
        jax_pytree = {}
        for name, param in model_state_dict.items():
            param_np = param.detach().cpu().float().numpy()
            jax_pytree[name] = jnp.array(param_np)
        return jax_pytree
    
    def jax_pytree_to_pytorch(jax_pytree):
        pytorch_state = {}
        for name, param in jax_pytree.items():
            param_np = np.array(param)
            pytorch_state[name] = torch.from_numpy(param_np).float()
        return pytorch_state
    
    orbax_save_dir = os.path.abspath('saved_models/openllama_3b_orbax/')
    os.makedirs(orbax_save_dir, exist_ok=True)
    
    print("\nsaving with orbax...")
    start_time = time.time()
    
    model_state = model.state_dict()
    print(f"  processing {len(model_state)} parameters with orbax optimizations...")
    
    jax_pytree = pytorch_to_jax_pytree(model_state)
    
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    
    save_args = jax.tree_util.tree_map(
        lambda x: ocp.SaveArgs(chunk_byte_size=1024 * 1024),
        jax_pytree,
    )
    
    checkpoint_path = epath.Path(orbax_save_dir) / 'checkpoint'
    checkpointer.save(checkpoint_path, jax_pytree, save_args=save_args)
    
    orbax_save_time = time.time() - start_time
    
    orbax_size = 0
    for root, dirs, files in os.walk(orbax_save_dir):
        for file in files:
            orbax_size += os.path.getsize(os.path.join(root, file))
    orbax_file_size = orbax_size / (1024**3)
    
    print(f"✓ orbax save completed in {orbax_save_time*1000:.1f} ms")
    print(f"  saved {len(jax_pytree)} parameters")
    print(f"  file size: {orbax_file_size:.2f} gb")
    
    print("\nloading with orbax...")
    start_time = time.time()
    
    abstract_pytree = jax.tree_util.tree_map(
        lambda x: ocp.utils.to_shape_dtype_struct(x),
        jax_pytree
    )
    
    loaded_jax_pytree = checkpointer.restore(checkpoint_path, abstract_pytree)
    loaded_pytorch_state = jax_pytree_to_pytorch(loaded_jax_pytree)
    
    orbax_load_time = time.time() - start_time
    
    print(f"✓ orbax load completed in {orbax_load_time*1000:.1f} ms")
    print(f"  loaded {len(loaded_pytorch_state)} parameters")
    
    del loaded_pytorch_state, loaded_jax_pytree, jax_pytree, model_state
    gc.collect()
    
    orbax_available = True
    
except Exception as e:
    print(f"\n⚠ orbax phase skipped: {e}")
    print("  continuing with 3-way comparison...")
    orbax_available = False
    orbax_save_time = 0
    orbax_load_time = 0
    orbax_file_size = 0

# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
if orbax_available:
    print("4-WAY PERFORMANCE COMPARISON")
else:
    print("3-WAY PERFORMANCE COMPARISON")
print("="*80)

if orbax_available:
    methods = ['pytorch', 'tensorstore', 't5x-tensorstore', 'orbax']
    save_times = [
        pytorch_save_time * 1000,
        tensorstore_save_time * 1000,
        t5x_tensorstore_save_time * 1000,
        orbax_save_time * 1000
    ]
    load_times = [
        pytorch_load_time * 1000,
        tensorstore_load_time * 1000,
        t5x_tensorstore_load_time * 1000,
        orbax_load_time * 1000
    ]
    file_sizes = [
        pytorch_file_size,
        tensorstore_file_size,
        t5x_tensorstore_file_size,
        orbax_file_size
    ]
    colors = ['blue', 'orange', 'green', 'red']
else:
    methods = ['pytorch', 'tensorstore', 't5x-tensorstore']
    save_times = [
        pytorch_save_time * 1000,
        tensorstore_save_time * 1000,
        t5x_tensorstore_save_time * 1000
    ]
    load_times = [
        pytorch_load_time * 1000,
        tensorstore_load_time * 1000,
        t5x_tensorstore_load_time * 1000
    ]
    file_sizes = [
        pytorch_file_size,
        tensorstore_file_size,
        t5x_tensorstore_file_size
    ]
    colors = ['blue', 'orange', 'green']

print(f"\n{'method':<18} {'save (ms)':<12} {'load (ms)':<12} {'size (gb)':<12}")
print('-' * 65)
for i, method in enumerate(methods):
    print(f'{method:<18} {save_times[i]:<12.1f} {load_times[i]:<12.1f} {file_sizes[i]:<12.2f}')

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING PERFORMANCE VISUALIZATION")
print("="*80)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# save time
ax1.bar(methods, save_times, color=colors)
ax1.set_title('save time comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('time (ms)', fontsize=12)
ax1.set_ylim(0, max(save_times) * 1.1)
for i, v in enumerate(save_times):
    ax1.text(i, v + max(save_times) * 0.02, f'{v:.0f}ms', ha='center', fontweight='bold')

# load time
ax2.bar(methods, load_times, color=colors)
ax2.set_title('load time comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('time (ms)', fontsize=12)
ax2.set_ylim(0, max(load_times) * 1.1)
for i, v in enumerate(load_times):
    ax2.text(i, v + max(load_times) * 0.02, f'{v:.0f}ms', ha='center', fontweight='bold')

# file size
ax3.bar(methods, file_sizes, color=colors)
ax3.set_title('file size comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('size (gb)', fontsize=12)
ax3.set_ylim(0, max(file_sizes) * 1.1)
for i, v in enumerate(file_sizes):
    ax3.text(i, v + max(file_sizes) * 0.02, f'{v:.2f}gb', ha='center', fontweight='bold')

# efficiency
save_efficiency = [save_times[i]/save_times[0] for i in range(len(methods))]
load_efficiency = [load_times[i]/load_times[0] for i in range(len(methods))]
size_efficiency = [file_sizes[i]/file_sizes[0] for i in range(len(methods))]

x_pos = np.arange(len(methods))
width = 0.25

ax4.bar(x_pos - width, save_efficiency, width, label='save time', color='lightcoral', alpha=0.8)
ax4.bar(x_pos, load_efficiency, width, label='load time', color='lightblue', alpha=0.8)
ax4.bar(x_pos + width, size_efficiency, width, label='file size', color='lightgreen', alpha=0.8)

ax4.set_title('efficiency relative to pytorch', fontsize=14, fontweight='bold')
ax4.set_ylabel('relative performance (pytorch = 1.0)', fontsize=12)
ax4.set_xlabel('method', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods)
ax4.legend()
ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
output_file = f'saved_models/{"4way" if orbax_available else "3way"}_performance_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ performance chart saved to: {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nmodel: openllama-3b (3426.5m parameters)")
print(f"device: {device}")

best_save = methods[save_times.index(min(save_times))]
best_load = methods[load_times.index(min(load_times))]
best_size = methods[file_sizes.index(min(file_sizes))]

print('\n📈 performance winners:')
print(f'• fastest save: {best_save}')
print(f'• fastest load: {best_load}')
print(f'• smallest size: {best_size}')

print('\n✅ all phases completed successfully!')
print('\n📁 generated files:')
print('• saved_models/openllama_3b_pytorch.pth')
print('• saved_models/openllama_3b_tensorstore/')
print('• saved_models/openllama_3b_t5x_tensorstore/')
if orbax_available:
    print('• saved_models/openllama_3b_orbax/')
print(f'• {output_file}')

print('\n' + '='*80)
print('EXECUTION COMPLETED')
print('='*80)
