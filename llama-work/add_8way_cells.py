#!/usr/bin/env python3
import nbformat as nbf

def get_cells():
    """return all phase 4 cells for 8-way comparison"""
    return [
        # header
        nbf.v4.new_markdown_cell("""# phase 4: 8-way comprehensive comparison

**all approaches save the same 238 parameters for fair comparison**

**8 approaches:**
1. pytorch - native pytorch (baseline)
2. tensorstore - basic, no optimizations
3. t5x - all optimizations
4. ts+concurrency - only 128 concurrent ops
5. ts+chunks - only 1mb chunks
6. ts+compression - only gzip
7. ts+float16 - float16 dtype (2 bytes)
8. ts+ocdbt - ocdbt driver"""),
