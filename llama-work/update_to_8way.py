#!/usr/bin/env python3
"""update notebook to 8-way comparison"""
import nbformat as nbf

# read notebook
with open('main.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# remove old phase 4 cells
filtered_cells = []
in_phase4 = False
for cell in nb.cells:
    if cell.cell_type == 'markdown' and 'phase 4' in cell.source.lower():
        in_phase4 = True
        continue
    if in_phase4 and cell.cell_type == 'markdown' and cell.source.startswith('#') and 'phase 4' not in cell.source.lower():
        in_phase4 = False
    if not in_phase4:
        filtered_cells.append(cell)

nb.cells = filtered_cells

# add new phase 4 cells from separate file
import add_8way_cells
nb.cells.extend(add_8way_cells.get_cells())

# write updated notebook
with open('main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✓ Updated to 8-way comparison!")
