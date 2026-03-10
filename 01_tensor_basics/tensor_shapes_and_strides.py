import torch


def element_offset(indices: list[int], strides: tuple[int, ...]) -> int:
    """Compute the flat storage offset for a given index tuple using strides."""
    return sum(i * s for i, s in zip(indices, strides))


def main() -> None:
    # A tensor's memory position for element [r, c] is:
    #   storage_offset + r * stride[0] + c * stride[1]

    # Step 1: contiguous 3x4 tensor — row-major layout.
    x = torch.arange(12).reshape(3, 4)
    print("x =")
    print(x)
    print(f"shape:  {x.shape}")
    print(f"stride: {x.stride()}")
    print()

    # Step 2: manually verify stride formula matches actual values.
    print("Manual stride lookup vs tensor indexing:")
    for r in range(3):
        for c in range(4):
            offset = element_offset([r, c], x.stride())
            actual = x[r, c].item()
            storage_val = x.storage()[offset]
            print(f"  x[{r},{c}] = {actual}  |  stride formula offset={offset}  |  storage[{offset}]={storage_val:.0f}")
    print()

    # Step 3: after transpose, stride changes but storage does not.
    y = x.transpose(0, 1)
    print("y = x.transpose(0, 1)")
    print(f"shape:  {y.shape}")
    print(f"stride: {y.stride()}  <-- rows now jump by 1, cols jump by 4")
    print()

    # Step 4: verify transposed stride formula.
    print("Manual stride lookup on transposed tensor:")
    for r in range(4):
        for c in range(3):
            offset = element_offset([r, c], y.stride())
            actual = y[r, c].item()
            storage_val = x.storage()[offset]  # same storage as x
            print(f"  y[{r},{c}] = {actual}  |  stride formula offset={offset}  |  storage[{offset}]={storage_val:.0f}")
    print()

    # Step 5: 3D tensor — stride extends naturally to N dimensions.
    z = torch.arange(24).reshape(2, 3, 4)
    print("z = arange(24).reshape(2, 3, 4)")
    print(f"shape:  {z.shape}")
    print(f"stride: {z.stride()}  <-- [12, 4, 1]")
    print()

    # Verify: z[d, r, c] = d*12 + r*4 + c*1
    for d, r, c in [(0, 0, 0), (0, 1, 2), (1, 0, 3), (1, 2, 3)]:
        offset = element_offset([d, r, c], z.stride())
        actual = z[d, r, c].item()
        print(f"  z[{d},{r},{c}] = {actual}  |  {d}*12 + {r}*4 + {c}*1 = {offset}")
    print()

    # Key takeaways:
    # - stride[i] is the number of storage elements to skip when index[i] increases by 1
    # - for a contiguous C-order tensor: stride[i] = product of all dims after i
    # - transpose swaps strides, not data — same storage, different read rules
    # - element_offset = sum(index[i] * stride[i]) is the universal access formula


if __name__ == "__main__":
    main()
