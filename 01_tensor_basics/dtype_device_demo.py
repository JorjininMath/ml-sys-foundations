import torch


def print_meta(name: str, tensor: torch.Tensor) -> None:
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} dtype: {tensor.dtype}")
    print(f"{name} device: {tensor.device}")
    print(f"{name} is_contiguous: {tensor.is_contiguous()}")
    print(tensor)
    print()


def select_compute_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    # Step 1: start from an integer tensor on CPU.
    x = torch.arange(6).reshape(2, 3)
    print_meta("x (base int tensor)", x)

    # Step 2: dtype conversions.
    x_fp32 = x.to(dtype=torch.float32)
    x_fp16 = x.to(dtype=torch.float16)
    x_bf16 = x.to(dtype=torch.bfloat16)
    print_meta("x_fp32", x_fp32)
    print_meta("x_fp16", x_fp16)
    print_meta("x_bf16", x_bf16)

    # Step 2b: memory footprint — lower precision = fewer bytes per element.
    for name, t in [("fp32", x_fp32), ("fp16", x_fp16), ("bf16", x_bf16)]:
        print(f"{name}: element_size={t.element_size()} bytes, total nbytes={t.nbytes}")
    print()

    # Step 2c: fp16 has a narrow dynamic range (~±65504); values outside overflow to inf.
    # bf16 shares fp32's exponent range, so it handles large values without overflow.
    large = torch.tensor(65504.0)
    print(f"fp16 max (~65504): {large.to(torch.float16)}")
    print(f"fp16 overflow (65504 * 2): {(large * 2).to(torch.float16)}")   # -> inf
    print(f"bf16 same value  (65504 * 2): {(large * 2).to(torch.bfloat16)}")  # -> ok
    print()

    # Step 3: pick best available compute device.
    device = select_compute_device()
    print(f"selected device: {device}")
    print()

    # Step 4: move tensor to selected device.
    x_device = x_fp32.to(device=device)
    print_meta("x_device", x_device)

    # Step 5: convert dtype and device in one call.
    x_device_fp16 = x.to(device=device, dtype=torch.float16)
    print_meta("x_device_fp16 (single .to call)", x_device_fp16)

    # Step 6: common pitfall (integer tensors cannot require gradients).
    try:
        x.requires_grad_(True)
    except RuntimeError as error:
        print("int tensor requires_grad failed as expected:")
        print(error)
        print()

    x_grad_ok = x_fp32.clone().requires_grad_(True)
    print_meta("x_grad_ok (float tensor with gradients)", x_grad_ok)

    # Key takeaways:
    # - dtype controls numeric representation and precision
    # - device controls where computation happens
    # - use .to(device=..., dtype=...) for explicit and readable conversion


if __name__ == "__main__":
    main()
