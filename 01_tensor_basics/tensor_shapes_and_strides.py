import torch


def print_meta(name: str, tensor: torch.Tensor) -> None:
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} stride: {tensor.stride()}")
    print(f"{name} dtype: {tensor.dtype}")
    print(f"{name} device: {tensor.device}")
    print(f"{name} is_contiguous: {tensor.is_contiguous()}")
    print()


def main() -> None:
    # Step 1: start from a contiguous 2x3 tensor.
    a = torch.arange(6).reshape(2, 3)
    print_meta("a (base)", a)

    # Step 2: transpose changes logical axes and stride.
    b = a.transpose(0, 1)
    print_meta("b (transposed)", b)

    # Step 3: contiguous() creates a contiguous copy with same values.
    c = b.contiguous()
    print_meta("c (after contiguous)", c)

    # Key takeaways:
    # - shape describes logical layout
    # - stride describes memory steps per dimension
    # - transpose often changes stride and makes tensor non-contiguous


if __name__ == "__main__":
    main()