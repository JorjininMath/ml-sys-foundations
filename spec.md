# ML Systems Foundations — Project Spec

## Goal

Build a solid, hands-on understanding of the core ideas behind modern deep learning systems.
This is a learning repository, not a product. Every piece of code here should deepen conceptual understanding.

## Core Thesis

Deep learning frameworks like PyTorch and JAX do not use plain NumPy as their primary abstraction because modern training requires a unified object that handles:

- heterogeneous devices (CPU / GPU / TPU / MPS)
- automatic differentiation
- compiled/high-performance kernels
- low-precision computation (FP16, BF16)
- memory management
- distributed communication

The goal of this repo is to understand *why* these things exist — not just how to use them.

---

## Modules

### 01 — Tensor Basics

**Key concepts:** shape, dtype, device, stride, contiguous vs non-contiguous, view vs reshape, transpose/permute, broadcasting

**Goal:** Build intuition for what a tensor is beyond "just a matrix."

**Files to build:**
- `tensor_shapes_and_strides.py` — explore shape/stride relationship
- `contiguous_vs_noncontiguous.py` — show memory layout effects
- `view_reshape_permute_demo.py` — when view fails vs reshape succeeds
- `dtype_device_demo.py` — dtype casting, device placement

---

### 02 — Autograd from Scratch

**Key concepts:** computational graph, forward pass, backward pass, chain rule in code, grad accumulation, topological sort

**Implementation path:**
1. Scalar autograd engine (`Value` class with `+`, `*`, `relu`, `backward`)
2. Extend to vector/matrix ops (`add`, `mul`, `matmul`, `sum`)
3. Train a tiny MLP end-to-end

**Goal:** Understand why `requires_grad`, `grad_fn`, and `backward()` exist.

**Files to build:**
- `scalar_engine.py` — scalar autograd with graph
- `graph_viz.py` — visualize the computational graph
- `tiny_ops.py` — vector/matrix ops with autograd
- `mlp_demo.py` — train a tiny MLP using the custom engine
- `tests/` — unit tests for gradient correctness

---

### 03 — PyTorch Core Mechanics

**Key concepts:** tensor creation, device transfer, `requires_grad`, `nn.Module`, loss, optimizer, `zero_grad`, `no_grad`, train vs eval mode

**Goal:** Connect the from-scratch autograd intuition to real PyTorch usage.

**Files to build:**
- `basic_tensor_ops.py`
- `autograd_demo.py`
- `nn_module_demo.py`
- `simple_training_loop.py`
- `eval_vs_train_demo.py`

---

### 04 — Training Mechanics

**Key concepts:** activation storage during forward, gradient flow during backward, optimizer state, batch size effects, gradient accumulation, why memory grows

**Goal:** Make training feel mechanical, not magical.

**Files to build:**
- `forward_backward_step.py` — trace a single full step
- `gradient_accumulation_demo.py`
- `optimizer_state_demo.py` — show Adam's m/v states
- `activation_memory_demo.py` — measure memory at each stage

---

### 05 — Memory and Precision

**Key concepts:** CPU vs GPU memory, host-to-device cost, FP32/FP16/BF16 tradeoffs, mixed precision, parameter/activation/optimizer memory, contiguous layout

**Goal:** Understand why performance is not only about FLOPs.

**Files to build:**
- `fp32_fp16_bf16_demo.py` — compare precision, range, speed
- `cpu_gpu_transfer_demo.py` — measure transfer overhead
- `memory_breakdown_demo.py` — break down memory usage per training stage
- `layout_and_performance_demo.py` — contiguous vs non-contiguous speed

---

### 06 — Performance Basics

**Key concepts:** NumPy vs torch CPU vs torch GPU, data transfer overhead, kernel launch overhead, tensor layout effects, keeping data on one device

**Goal:** Turn system claims into measurable experiments.

---

### 07 — Distributed Basics (Conceptual)

**Key concepts:** data parallelism, tensor parallelism, model parallelism, pipeline parallelism, AllReduce, communication as bottleneck

**Goal:** Build enough intuition for future work in LLM/distributed systems.

**Files:** Primarily `.md` concept docs, not code.

---

## What Does NOT Belong Here

- RAG pipelines
- Agent frameworks
- Multi-tool copilots
- Product or demo code
- API wrappers
- Frontend applications
- Application-level LLM experiments

Those belong in separate repositories.

---

## Current Progress Snapshot (Tensor Basics Part 1)

Completed:

- `01_tensor_basics/tensor_shapes_and_strides.py`
  - unified script structure with `main()`
  - metadata inspection (`shape`, `stride`, `dtype`, `device`, `is_contiguous`)
  - transpose and contiguous comparison
- `01_tensor_basics/contiguous_vs_noncontiguous.py`
  - contiguous vs non-contiguous layout demo
  - `view` failure on non-contiguous tensor
  - `contiguous() + view` success path
  - `reshape` behavior contrast
- `01_tensor_basics/README.md`
  - module goal, file map, run instructions
- Zhihu draft (Chinese)
  - `01_tensor_basics/zhihu_tensor_basics_part1_cn.md`

Next:

- finish `01_tensor_basics/view_reshape_permute_demo.py`
- finish `01_tensor_basics/dtype_device_demo.py`
- then move to `02_autograd_from_scratch/`
