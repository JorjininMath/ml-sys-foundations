# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

- Always respond to the user in **Chinese**
- All code, file names, comments, and written files (`.py`, `.md`, etc.) must be in **English**

## Project Purpose

This is a **personal learning lab** for ML systems foundations. The goal is deep understanding of the core mechanics behind modern deep learning — not building applications.

**In scope:** tensor internals, autograd, PyTorch training mechanics, memory/precision, performance, distributed basics.

**Out of scope:** RAG, agents, LLM applications, product demos, API wrappers, frontend.

## Repository Structure

```
00_notes/          # concept notes and summaries
01_tensor_basics/  # tensor shape, stride, contiguous, view, dtype, device
02_autograd_from_scratch/  # scalar → vector autograd engine, tiny MLP
03_pytorch_core/   # PyTorch tensor ops, nn.Module, training loop
04_training_mechanics/     # forward/backward/optimizer, gradient accumulation
05_memory_and_precision/   # FP32/FP16/BF16, CPU↔GPU transfer, memory breakdown
06_distributed_basics/     # data/model/pipeline parallelism concepts, AllReduce
99_notes/          # future learning ideas
```

## Learning Sequence

Follow this order when adding new content:
1. Tensor basics → 2. Autograd → 3. PyTorch core → 4. Training mechanics → 5. Memory & precision → 6. Performance → 7. Distributed basics

## Zhihu Article Writing Rules

Each module may have a `zhihu_*_cn.md` file — a Chinese knowledge article for publishing on Zhihu.

**Audience:** Chinese readers learning PyTorch/ML systems. They do not have access to this repo.

**Structure:**
- Title: concept-focused, not "script walkthrough" (e.g. "从 Shape 到 Stride：为什么转置后 Tensor 会变成 Non-Contiguous？")
- Sections numbered 一、二、三… in logical concept order
- End with a 总结 and a 下一篇 teaser

**Content rules:**
- Never reference `.py` files or repo paths — article must be self-contained
- Every code block must be `python` fenced (not plain text or bullet list snippets)
- Include `print()` output as comments inside the code block so readers see what to expect
- Use `text` fenced blocks only for memory layout diagrams
- Explain the *why* behind each concept, not just what the API does
- Use comparison tables when contrasting two or more operations (e.g. `view` vs `reshape`)

**Tone:** clear and direct, assumes the reader knows basic Python and NumPy but is new to PyTorch internals

## Running Code

Each module is a standalone Python script. Run directly:

```bash
python 01_tensor_basics/tensor_shapes_and_strides.py
python 02_autograd_from_scratch/scalar_engine.py
```

Tests (when present) live in `tests/` under each module:

```bash
python -m pytest 02_autograd_from_scratch/tests/
```
