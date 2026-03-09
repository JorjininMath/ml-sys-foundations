# Tensor Basics Part 1 (Draft)

## Working Title

Shape, Stride, and Contiguity: Why Tensor Layout Matters

## Target Audience

Readers who already know basic tensor creation, but do not yet understand memory layout behavior.

## Core Questions

1. What does `stride` actually mean?
2. Why does `transpose` often produce non-contiguous tensors?
3. Why can `view` fail while `reshape` succeeds?

## Key Scripts

- `01_tensor_basics/tensor_shapes_and_strides.py`
- `01_tensor_basics/contiguous_vs_noncontiguous.py`

## Narrative Outline

### 1) Problem Setup

Most beginners track tensor `shape`, but ignore `stride`.  
That is why operations like `view()` can fail unexpectedly after `transpose()`.

### 2) Experiment A: Read Tensor Metadata

Use `tensor_shapes_and_strides.py` to compare:

- base tensor `a`
- transposed tensor `b`
- contiguous copy `c`

Focus on:

- shape stays meaningful at the logical level
- stride changes after transpose
- contiguity status flips from `True` to `False` and back to `True`

### 3) Experiment B: Behavior Difference

Use `contiguous_vs_noncontiguous.py` to verify:

- `torch.equal(y, z)` is `True` (same logical values)
- `y.view(-1)` fails on non-contiguous layout
- `z.view(-1)` succeeds after `contiguous()`
- `y.reshape(-1)` succeeds because reshape may copy when needed

### 4) Mechanism Explanation

- `shape` describes logical dimensions
- `stride` describes memory jump size per dimension
- `view` does not reorder memory; it only reinterprets existing layout
- non-contiguous memory often cannot satisfy `view` requirements

### 5) Common Pitfalls

- assuming transpose physically reorders data
- treating `view` and `reshape` as always equivalent
- checking shape only, without checking stride/contiguity

### 6) Practical Checklist

Before debugging shape/layout errors:

1. print `shape`
2. print `stride()`
3. print `is_contiguous()`
4. if needed, call `contiguous()` before `view`

### 7) Three Takeaways

1. Shape tells what a tensor looks like; stride tells how memory is traversed.
2. Transpose usually changes stride without immediate data copy.
3. Layout-sensitive ops (like `view`) depend on contiguity-compatible memory.

## Assets To Add Before Publish

- one screenshot of script output for Experiment A
- one screenshot of `view` failure vs success in Experiment B
- one simple diagram of "same values, different layout"

## Next Article Bridge

Next, focus on `view`, `reshape`, and `permute` in detail with explicit failure and copy behavior.
