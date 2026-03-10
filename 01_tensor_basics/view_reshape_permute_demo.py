import torch


def print_meta(name: str, tensor: torch.Tensor) -> None:
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} stride: {tensor.stride()}")
    print(f"{name} is_contiguous: {tensor.is_contiguous()}")
    print(tensor)
    print()


def main() -> None:
    # 1) Base contiguous tensor
    x = torch.arange(12).reshape(3, 4)
    print_meta("x (base)", x)

    # 2) view on contiguous tensor -> success, shares the same storage
    x_view = x.view(2, 6)
    print_meta("x_view = x.view(2, 6)", x_view)
    print(f"x and x_view share storage: {x.data_ptr() == x_view.data_ptr()}")
    print()

    # 3) Transpose -> usually non-contiguous
    y = x.transpose(0, 1)
    print_meta("y = x.transpose(0, 1)", y)

    # 4) view on non-contiguous tensor -> expected failure
    try:
        y_view = y.view(2, 6)
        print_meta("y_view = y.view(2, 6) (unexpected)", y_view)
    except RuntimeError as error:
        print("y.view(2, 6) failed as expected:")
        print(error)
        print()

    # 5) reshape on non-contiguous tensor -> can succeed (may copy)
    y_reshape = y.reshape(2, 6)
    print_meta("y_reshape = y.reshape(2, 6)", y_reshape)

    # 6) permute reorders dimensions — changes strides, result is usually non-contiguous
    z = x.reshape(2, 2, 3)
    print_meta("z = x.reshape(2, 2, 3)", z)

    z_perm = z.permute(2, 0, 1)
    print_meta("z_perm = z.permute(2, 0, 1)", z_perm)
    # permute, like transpose, only changes strides — does NOT guarantee contiguity

    # 7) must call contiguous() before view on a permuted tensor
    z_perm_contig = z_perm.contiguous()
    z_flat = z_perm_contig.view(-1)
    print_meta("z_flat = z_perm.contiguous().view(-1)", z_flat)
    print(f"z_perm and z_perm_contig share storage: {z_perm.data_ptr() == z_perm_contig.data_ptr()}")
    print()

    # 8) broadcasting: PyTorch expands tensors with size-1 dims to match larger shapes
    a = torch.ones(3, 1)           # shape (3, 1)
    b = torch.ones(1, 4)           # shape (1, 4)
    c = a + b                      # result shape (3, 4) — no data was copied
    print(f"broadcasting: {a.shape} + {b.shape} -> {c.shape}")
    print()

    # broadcasting aligns dims from the right; size-1 expands, missing dims treated as 1
    x_row = torch.arange(4).float()               # shape (4,)
    x_col = torch.arange(3).float().unsqueeze(1)  # shape (3, 1)
    result = x_row + x_col                        # shape (3, 4)
    print(f"x_row {x_row.shape} + x_col {x_col.shape} -> {result.shape}")
    print(result)
    print()

    # Key takeaways:
    # - view: zero-copy, requires contiguous-compatible layout
    # - reshape: copies if needed, always succeeds for same number of elements
    # - permute: reorders dims via stride changes, result often non-contiguous
    # - broadcasting: expands size-1 dims virtually (no data copy)


if __name__ == "__main__":
    main()