# ML Systems Foundations

## Project Goal

This repository is **only for base/foundation learning**, not for application-layer projects like RAG, agents, or product demos.

The purpose is to build a solid understanding of the core ideas behind modern deep learning systems, especially:

- what a tensor really is
- why deep learning frameworks use tensors instead of plain NumPy arrays
- how autograd works
- how device / dtype / memory / layout affect performance
- what happens during training
- the basic ideas behind distributed training

This repo should feel like a **clean technical learning lab**, focused on fundamentals that support later work in LLM systems, agents, and AI engineering.

---

## Naming

Preferred root folder name:

`ml-systems-foundations`

Reason:
- more focused than a general `ai-foundations`
- matches the actual learning goal
- professional enough for long-term GitHub use
- broad enough to include tensor, autograd, training, memory, and distributed basics

---

## What this repo is NOT for

Do **not** put these in this repository:

- RAG projects
- agent workflow projects
- multi-tool copilots
- product demos
- API wrappers
- frontend apps
- application-level LLM experiments

Those should live in separate repositories.

This repo should remain focused on **base learning**.

---

## Background Motivation

We discussed a long article comparing NumPy and Tensor-based deep learning frameworks.

The main conclusion is:

Deep learning does not rely on plain NumPy as the main training abstraction not because NumPy is “bad,” but because modern deep learning needs a unified abstraction that can handle:

- heterogeneous devices (CPU / GPU / TPU / MPS)
- automatic differentiation
- compiled/high-performance kernels
- low-precision computation
- memory management
- distributed communication

In frameworks like PyTorch and JAX, that abstraction is typically a tensor/array object with extra semantics, not just a plain CPU array.

Important nuance:
- the article’s **main direction is mostly correct**
- but some claims are too absolute or exaggerated
- NumPy itself is CPU-oriented, but NumPy-like APIs can still power accelerator-based systems (for example JAX)
- the real point is not “NumPy is useless,” but that deep learning needs a richer computation abstraction

---

## Learning Priorities

The best learning sequence we identified is:

1. Tensor basics
2. Autograd
3. PyTorch training mechanics
4. Memory / dtype / device / layout
5. Performance basics
6. Distributed training basics

This order is important:
- first understand the data abstraction
- then understand gradient flow
- then understand how training actually works
- then move into systems/performance concepts

---

## Core Topics to Cover

### 1. Tensor Basics
Learn the real meaning of:
- shape
- dtype
- device
- stride
- contiguous vs non-contiguous memory
- view vs reshape
- transpose / permute
- broadcasting

Goal:
build intuition for what a tensor is beyond “just a matrix.”

---

### 2. Autograd from Scratch
This is one of the most important parts.

Learn:
- computational graph
- forward pass
- backward pass
- chain rule in code
- grad accumulation
- topological order in backward

Suggested implementation path:
- start with scalar autograd
- then extend to vector/matrix operations
- support a tiny set of ops like add, mul, matmul, relu, sum
- train a tiny MLP

Goal:
understand why `requires_grad`, `grad_fn`, and `backward()` exist.

---

### 3. PyTorch Core Mechanics
Learn how PyTorch expresses the ideas above in practice.

Topics:
- tensor creation
- moving tensors across devices
- `requires_grad`
- `nn.Module`
- loss computation
- optimizer step
- `zero_grad`
- training loop vs evaluation loop
- `torch.no_grad()`

Goal:
connect theory to real framework usage.

---

### 4. Training Mechanics
Understand what actually happens during neural network training.

Topics:
- forward pass
- activation storage
- backward pass
- optimizer update
- batch size
- gradient accumulation
- parameter states
- why memory grows during training

Goal:
make training feel mechanical and understandable, not magical.

---

### 5. Memory and Precision
This is the start of the systems view.

Topics:
- CPU memory vs GPU memory
- host-to-device transfer cost
- FP32 / FP16 / BF16 basics
- why mixed precision helps
- activation memory
- parameter memory
- optimizer state memory
- memory bottlenecks
- contiguous layout and performance implications

Goal:
understand why performance is not only about FLOPs.

---

### 6. Performance Basics
Learn how code structure affects actual runtime.

Topics:
- NumPy vs torch CPU vs torch GPU comparisons
- data transfer overhead
- kernel launch overhead
- effect of tensor layout
- why keeping data on one device matters
- common profiling mindset

Goal:
turn system claims into measurable experiments.

---

### 7. Distributed Basics
Only basic concepts for now, not full infrastructure engineering.

Topics:
- why single-GPU training is limited
- data parallelism
- tensor/model parallelism
- pipeline parallelism
- AllReduce intuition
- communication as a bottleneck

Goal:
build enough systems intuition for later LLM/distributed learning.

---

## Recommended Repository Structure

```text
ml-systems-foundations/
├── README.md
├── 00_notes/
│   ├── foundation_summary.md
│   ├── tensor_notes.md
│   ├── autograd_notes.md
│   ├── training_notes.md
│   └── distributed_notes.md
├── 01_tensor_basics/
│   ├── README.md
│   ├── tensor_shapes_and_strides.py
│   ├── contiguous_vs_noncontiguous.py
│   ├── view_reshape_permute_demo.py
│   └── dtype_device_demo.py
├── 02_autograd_from_scratch/
│   ├── README.md
│   ├── scalar_engine.py
│   ├── graph_viz.py
│   ├── tiny_ops.py
│   ├── mlp_demo.py
│   └── tests/
├── 03_pytorch_core/
│   ├── README.md
│   ├── basic_tensor_ops.py
│   ├── autograd_demo.py
│   ├── nn_module_demo.py
│   ├── simple_training_loop.py
│   └── eval_vs_train_demo.py
├── 04_training_mechanics/
│   ├── README.md
│   ├── forward_backward_step.py
│   ├── gradient_accumulation_demo.py
│   ├── optimizer_state_demo.py
│   └── activation_memory_demo.py
├── 05_memory_and_precision/
│   ├── README.md
│   ├── fp32_fp16_bf16_demo.py
│   ├── cpu_gpu_transfer_demo.py
│   ├── memory_breakdown_demo.py
│   └── layout_and_performance_demo.py
├── 06_distributed_basics/
│   ├── README.md
│   ├── distributed_concepts.md
│   ├── allreduce_intuition.md
│   └── parallelism_map.md
└── 99_notes/
    └── future_learning_ideas.md