# Zhihu Column Structure for ML Systems Foundations

## Purpose

This document defines a practical writing structure for publishing a Zhihu column based on this repository's learning path.
The goal is to keep writing aligned with implementation progress and make each article reproducible from runnable scripts.

## Publishing Strategy

- One completed script or concept milestone maps to one article draft.
- Prefer depth over breadth: each article answers one core systems question.
- Keep article length around 1,500-2,500 Chinese characters for readability.
- Use evidence-based writing: code output, measured behavior, and clear conclusions.

## Recommended 8-Article Series

### Article 1

- **Working title:** Why Deep Learning Frameworks Center on Tensors
- **Core question:** Why is a tensor abstraction needed beyond plain CPU arrays?
- **Primary module mapping:** Repo overview + system motivation from spec/project docs
- **Deliverable focus:** Build conceptual foundation and audience expectation

### Article 2

- **Working title:** Shape and Stride: The Hidden Geometry of Tensor Access
- **Core question:** How do shape and stride jointly determine tensor indexing behavior?
- **Primary module mapping:** `01_tensor_basics/tensor_shapes_and_strides.py`
- **Deliverable focus:** First hands-on technical article with runnable examples

### Article 3

- **Working title:** Contiguous vs Non-Contiguous: Why Memory Layout Matters
- **Core question:** What changes when tensor memory is contiguous or not?
- **Primary module mapping:** `01_tensor_basics/contiguous_vs_noncontiguous.py`
- **Deliverable focus:** Explain layout-performance intuition with concrete demos

### Article 4

- **Working title:** view, reshape, permute: Similar API, Different Semantics
- **Core question:** Why does `view` fail in some cases while `reshape` succeeds?
- **Primary module mapping:** `01_tensor_basics/view_reshape_permute_demo.py`
- **Deliverable focus:** Clarify common beginner confusion with minimal cases

### Article 5

- **Working title:** Building Autograd from Scratch (Scalar Engine)
- **Core question:** How does backward propagation emerge from graph + chain rule?
- **Primary module mapping:** `02_autograd_from_scratch/scalar_engine.py`
- **Deliverable focus:** Turn theory into executable internals

### Article 6

- **Working title:** From Scalar Engine to Tiny MLP
- **Core question:** How do we scale from scalar ops to matrix-style training behavior?
- **Primary module mapping:** `02_autograd_from_scratch/tiny_ops.py` + `mlp_demo.py`
- **Deliverable focus:** Bridge toy autograd to practical learning dynamics

### Article 7

- **Working title:** PyTorch Training Loop Under the Hood
- **Core question:** What exactly happens in `forward -> backward -> step -> zero_grad`?
- **Primary module mapping:** `03_pytorch_core/` + `04_training_mechanics/`
- **Deliverable focus:** Connect from-scratch intuition to real framework usage

### Article 8

- **Working title:** Precision, Memory, and Performance Trade-offs
- **Core question:** Why are FP16/BF16/memory layout/device transfer central to speed?
- **Primary module mapping:** `05_memory_and_precision/` + `06_performance_basics` concepts
- **Deliverable focus:** Systems view with measurable experiments

## Standard Article Template

Use this structure for every article:

1. **Problem Statement**
   - What concrete question does this article answer?
2. **Minimal Runnable Setup**
   - Which script(s) are used?
   - How to run them?
3. **Observation and Output**
   - What output/behavior is observed?
   - Which output lines are most important?
4. **Mechanism Explanation**
   - Why does this behavior happen internally?
5. **Common Pitfalls**
   - 2-4 mistakes and how to avoid them
6. **Takeaways**
   - 3 concise conclusions
7. **Next Article Link**
   - One-sentence handoff to the next topic

## Weekly Workflow (Simple and Sustainable)

- **Day 1-2:** Finish code experiment and keep clean script outputs.
- **Day 3:** Draft article using the standard template.
- **Day 4:** Add diagrams, output snippets, and tighten logic.
- **Day 5/Weekend:** Publish on Zhihu and log reader feedback notes.

## Article 2 Draft Skeleton (To Be Filled Next)

### Proposed title

Shape and Stride: The Hidden Geometry of Tensor Access

### Target outcome

By the end of this article, readers can explain:

- why same shape can still imply different memory traversal;
- how transpose changes stride without copying data (in many cases);
- why contiguous layout affects downstream operations.

### Placeholder sections

1. Opening problem scenario
2. Minimal tensor examples
3. Shape vs stride output walkthrough
4. Transpose/permute behavior explanation
5. `is_contiguous()` verification and implications
6. Practical debugging checklist
7. Bridge to Article 3

