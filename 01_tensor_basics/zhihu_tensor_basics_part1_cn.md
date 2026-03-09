# 从 Shape 到 Stride：为什么转置后 Tensor 会变成 Non-Contiguous？

很多人刚学 PyTorch 时，会把 Tensor 理解成"有 shape 的数组"。这个理解不算错，但不够完整。

真正决定 Tensor 很多行为差异的，往往是另一个属性：`stride`。

---

## 一、shape 和 stride 不是一回事

创建一个 2×3 的张量：

```python
import torch

a = torch.arange(6).reshape(2, 3)
print(a)
# tensor([[0, 1, 2],
#         [3, 4, 5]])

print(a.shape)          # torch.Size([2, 3])
print(a.stride())       # (3, 1)
print(a.is_contiguous()) # True
```

这里 `stride = (3, 1)` 的含义是：

- 沿第 0 维（行）前进一步，内存要跳过 **3 个元素**
- 沿第 1 维（列）前进一步，内存要跳过 **1 个元素**

这和底层内存布局完全吻合：`[0, 1, 2, 3, 4, 5]` 是连续排列的。

**`shape` 描述逻辑形状，`stride` 描述如何在内存里导航。**

---

## 二、转置只改 stride，不动内存

对 `a` 做转置：

```python
b = a.transpose(0, 1)
print(b)
# tensor([[0, 3],
#         [1, 4],
#         [2, 5]])

print(b.shape)           # torch.Size([3, 2])
print(b.stride())        # (1, 3)
print(b.is_contiguous()) # False
```

注意：**底层 storage 没有变**，`b` 和 `a` 共享同一块内存 `[0, 1, 2, 3, 4, 5]`。

PyTorch 只是把 stride 从 `(3, 1)` 改成了 `(1, 3)`，相当于换了一套"读取规则"：

```
原始 a：stride = (3, 1)
  a[0][0] → 内存位置 0
  a[0][1] → 内存位置 1   （+1）
  a[1][0] → 内存位置 3   （+3）

转置 b：stride = (1, 3)
  b[0][0] → 内存位置 0
  b[1][0] → 内存位置 1   （+1）
  b[0][1] → 内存位置 3   （+3）
```

读取 `b` 的同一行时，内存地址是跳跃的，不再连续 —— 这就是 `is_contiguous = False` 的原因。

---

## 三、`contiguous()` 做了什么

调用 `contiguous()` 会真正复制数据，把元素重新排成连续内存：

```python
c = b.contiguous()
print(c.stride())        # (2, 1)
print(c.is_contiguous()) # True
print(torch.equal(b, c)) # True，值相同，但底层 storage 不同
```

`b` 和 `c` 的值完全一样，但 `c` 拥有自己的独立内存，且布局是连续的。

---

## 四、为什么 `view` 会失败，`reshape` 却可能成功

`view` 要求 tensor 的内存必须是连续的，因为它本质上是在同一块内存上重新解释"形状"（零拷贝）。

```python
b.view(-1)    # RuntimeError: non-contiguous tensor 无法做零拷贝 view
c.view(-1)    # OK：tensor([0, 3, 1, 4, 2, 5])
```

`reshape` 则更灵活 —— 如果内存连续，它等同于 `view`（零拷贝）；如果不连续，它会先复制再重排：

```python
b.reshape(-1) # OK：内部会先 contiguous 再 view
```

一句话记住：

| 操作 | 要求 | 特点 |
|------|------|------|
| `view` | 必须 contiguous | 零拷贝，快 |
| `reshape` | 无要求 | 必要时复制，更灵活 |
| `contiguous()` | — | 强制整理成连续内存 |

---

## 五、调试清单

遇到 `RuntimeError: non-contiguous` 或布局相关报错时，先打印这四个属性：

```python
print(tensor.shape)
print(tensor.stride())
print(tensor.is_contiguous())
print(tensor.dtype, tensor.device)
```

如果是 `view` 报错，标准修法：

```python
tensor = tensor.contiguous()
tensor.view(...)  # 现在可以了
```

---

## 总结

1. **`shape`** 描述"看起来几行几列"，**`stride`** 描述"沿某一维走一步，内存要跳多少"。
2. `transpose()` 只改 stride，不复制数据，结果通常是 non-contiguous。
3. `view` 是零拷贝操作，依赖连续内存；`reshape` 在不连续时会自动复制。
4. `contiguous()` 是显式触发内存整理的方式。

---

## 下一篇

`view`、`reshape`、`permute` 三者的差异与拷贝行为，以及如何系统判断什么时候会触发内存复制。
