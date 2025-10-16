#!/usr/bin/env python3
"""
Fix main.ipynb to centralize device selection and force CPU usage
"""

import nbformat as nbf
import json

# Read the existing notebook
with open('main.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Find and update the device setup cell
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        # Update device setup cell
        if 'setup device and check cuda availability' in cell.source:
            cell.source = """# setup device - centralized device selection (force cpu for stability)
# set USE_CUDA = False to force cpu, True to use gpu if available
USE_CUDA = False  # change to True if you want to use gpu
device = torch.device('cuda' if (USE_CUDA and torch.cuda.is_available()) else 'cpu')
print(f"using device: {device}")

if device.type == 'cuda':
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"cuda memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} gb")
    print(f"cuda memory free: {torch.cuda.memory_reserved(0) / 1e9:.2f} gb")
else:
    print("running on cpu - this will be slower but more stable")"""
        
        # Update model loading cell
        elif 'LlamaForCausalLM.from_pretrained' in cell.source and 'device_map' in cell.source:
            cell.source = """# load openllama-3b model with pretrained weights
model_name = "openlm-research/open_llama_3b"
print(f"loading model: {model_name}")

# load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)
print("tokenizer loaded successfully")

# load model with memory optimization
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32 if device.type == 'cpu' else torch.float16,  # float32 for cpu, float16 for gpu
    device_map=None,  # don't use device_map to avoid meta tensors
    low_cpu_mem_usage=True,
    use_safetensors=True
)

# move model to device
model = model.to(device)

print(f"model loaded successfully")
print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m")

if device.type == 'cuda':
    print(f"cuda memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} gb")"""
        
        # Update inference test cell
        elif 'test model inference' in cell.source and 'inputs = {k: v.to(device)' in cell.source:
            cell.source = """# test model inference to verify it's working
test_prompt = "the future of artificial intelligence is"
inputs = tokenizer(test_prompt, return_tensors="pt")

# move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"testing model with prompt: '{test_prompt}'")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"generated text: {generated_text}")
print("model inference test successful")"""
        
        # Update final summary cell to use device variable consistently
        elif 'torch.cuda.get_device_name(0) if torch.cuda.is_available()' in cell.source:
            cell.source = cell.source.replace(
                'torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"',
                'torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"'
            )

# Write updated notebook
with open('main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Successfully updated main.ipynb!")
print("- Centralized device selection with USE_CUDA flag")
print("- Set to CPU mode by default")
print("- Removed device_map='auto' to avoid meta tensors")
print("- Consistent device usage throughout notebook")
