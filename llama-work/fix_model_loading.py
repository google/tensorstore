#!/usr/bin/env python3
"""
fix model loading in main.ipynb to handle device_map properly
"""

import nbformat as nbf

# read notebook
with open('main.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# find and fix the model loading cell
for cell in nb.cells:
    if cell.cell_type == 'code' and 'LlamaForCausalLM.from_pretrained' in cell.source:
        # replace the model loading code
        cell.source = """# load openllama-3b model with pretrained weights
model_name = "openlm-research/open_llama_3b"
print(f"loading model: {model_name}")

# load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)
print("tokenizer loaded successfully")

# load model with memory optimization
model = LlamaForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,  # use half precision for memory efficiency
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
    use_safetensors=True,  # use safetensors format
    offload_folder="saved_models/offload"  # folder for disk offload if needed
)

print(f"model loaded successfully")
print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}m")

if torch.cuda.is_available():
    print(f"cuda memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} gb")"""
        print("✓ fixed model loading cell")
        break

# write updated notebook
with open('main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✓ notebook fixed successfully")
