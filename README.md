# ML Systems Foundations

Personal learning lab for understanding core deep learning systems mechanics.
This repository focuses on fundamentals, not application-layer projects.

## Scope

In scope:

- tensor internals
- autograd from scratch
- PyTorch training mechanics
- memory, precision, and performance basics
- distributed training fundamentals

Out of scope:

- RAG pipelines
- agent frameworks
- product demos
- frontend or API wrapper projects

## Project Structure

- `00_notes/`: concept notes and summaries
- `01_tensor_basics/`: shape, stride, contiguous, view/reshape, dtype, device
- `02_autograd_from_scratch/`: scalar/vector autograd and tiny MLP
- `03_pytorch_core/`: real PyTorch core mechanics
- `04_training_mechanics/`: forward/backward/optimizer details
- `05_memory_and_precision/`: FP32/FP16/BF16 and memory breakdown
- `06_distributed_basics/`: distributed concepts and communication basics
- `99_notes/`: future learning ideas

## Environment Setup

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
```

## Run

Run scripts directly:

```bash
python 01_tensor_basics/tensor_shapes_and_strides.py
python 02_autograd_from_scratch/scalar_engine.py
```

Run tests when available:

```bash
python -m pytest 02_autograd_from_scratch/tests/
```

## Learning Order

1. Tensor basics
2. Autograd
3. PyTorch core
4. Training mechanics
5. Memory and precision
6. Performance basics
7. Distributed basics
