#!/usr/bin/env python3
"""
execute main.ipynb cell by cell with error handling
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

print("="*80)
print("EXECUTING MAIN.IPYNB")
print("="*80)

# read notebook
with open('main.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

print(f"\nTotal cells: {len(nb.cells)}")
print("Starting execution...\n")

# create executor
ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')

try:
    # execute notebook
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    
    # save executed notebook
    with open('main_executed.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print("\n" + "="*80)
    print("✓ EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nOutput saved to: main_executed.ipynb")
    
except Exception as e:
    print("\n" + "="*80)
    print("✗ EXECUTION FAILED")
    print("="*80)
    print(f"\nError: {e}")
    
    # save partial execution
    with open('main_partial.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print("\nPartial execution saved to: main_partial.ipynb")
    print("\nTrying to identify the failing cell...")
    
    # find which cell failed
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if 'outputs' in cell and cell['outputs']:
                for output in cell['outputs']:
                    if output.get('output_type') == 'error':
                        print(f"\n✗ Cell {i+1} failed:")
                        print(f"   {cell['source'][:100]}...")
                        print(f"\n   Error: {output.get('ename')}: {output.get('evalue')}")
                        break
    
    sys.exit(1)
