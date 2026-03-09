import torch


def print_meta(name: str, tensor: torch.Tensor) -> None:
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} stride: {tensor.stride()}")
    print(f"{name} dtype: {tensor.dtype}")
    print(f"{name} device: {tensor.device}")
    print(f"{name} is_contiguous: {tensor.is_contiguous()}")
    print()


def main() -> None:
    # Step 1: build a base contiguous tensor.
    x = torch.arange(12).reshape(3, 4)
    print_meta("x (base contiguous)", x)

    # Step 2: transpose creates a non-contiguous view in most cases.
    y = x.t()
    print_meta("y (transposed, usually non-contiguous)", y)

    # Step 3: contiguous() makes a contiguous copy.
    z = y.contiguous()
    print_meta("z (after contiguous)", z)

    # Same logical values, different memory layouts.
    print("y equals z:", torch.equal(y, z))
    print()

    # view is layout-sensitive.
    try:
        y_flat = y.view(-1)
        print("y.view succeeded unexpectedly:", y_flat.shape)
    except RuntimeError as error:
        print("y.view failed as expected:", error)

    z_flat = z.view(-1)
    print("z.view succeeded:", z_flat.shape)

    # reshape is more flexible and may copy if needed.
    y_reshape = y.reshape(-1)
    print("y.reshape succeeded:", y_reshape.shape)
    print()

    # Key takeaways:
    # - non-contiguous tensors can keep same values but different memory layout
    # - view requires compatible contiguous-like layout
    # - contiguous() can enable layout-sensitive ops such as view


if __name__ == "__main__":
    main()