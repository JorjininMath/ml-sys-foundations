# 01 Tensor Basics

## Module Goal

Build intuition for tensor internals beyond "just a matrix":

- shape
- stride
- contiguous vs non-contiguous memory
- view vs reshape vs permute
- dtype and device basics

## Documentation Strategy

This module currently uses a script-first workflow:

- runnable Python scripts for experiments
- concise README guidance
- external article drafts for publishing

JupyterBook can be added later when all scripts are stable.

## File Map

- `tensor_shapes_and_strides.py`
  - Reads tensor metadata and shows how transpose changes stride/contiguity.
- `contiguous_vs_noncontiguous.py`
  - Demonstrates behavior differences for `view`, `reshape`, and `contiguous()`.
- `view_reshape_permute_demo.py`
  - Focuses on shape-changing API differences and failure cases.
- `dtype_device_demo.py`
  - Explores dtype casting and device placement basics.

## How To Run

From repository root:

```bash
python 01_tensor_basics/tensor_shapes_and_strides.py
python 01_tensor_basics/contiguous_vs_noncontiguous.py
python 01_tensor_basics/view_reshape_permute_demo.py
python 01_tensor_basics/dtype_device_demo.py
```

## Suggested Order

1. `tensor_shapes_and_strides.py`
2. `contiguous_vs_noncontiguous.py`
3. `view_reshape_permute_demo.py`
4. `dtype_device_demo.py`
