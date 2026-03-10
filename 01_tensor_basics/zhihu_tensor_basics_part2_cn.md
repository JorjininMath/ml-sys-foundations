# view、reshape、permute 到底有什么区别？

很多人第一次遇到这三个 API，会觉得它们都在“改形状”，应该差不多。  
但一上手就会踩坑：有的成功、有的报错、有的悄悄复制。

这篇只回答一个问题：**何时零拷贝，何时复制，何时会报错？**

---

## 一、先看 `view`：快，但严格

在连续内存上，`view` 很高效：

```python
import torch

x = torch.arange(12).reshape(3, 4)  # contiguous
x_view = x.view(2, 6)

print(x.is_contiguous())      # True
print(x_view.is_contiguous()) # True
print(x.data_ptr() == x_view.data_ptr())  # True，共享 storage
```

这说明 `view` 的本质是：  
**不搬数据，只改解释方式（零拷贝）**。

---

## 二、为什么 transpose 后 `view` 常报错

```python
y = x.transpose(0, 1)
print(y.is_contiguous())  # False

y.view(2, 6)  # RuntimeError
```

原因不是“元素个数不对”，而是**布局不兼容**。  
`view` 需要原有 stride 能支持目标形状；transpose 后 stride 变了，常常不满足。

---

## 三、`reshape` 为什么更稳

```python
y_reshape = y.reshape(2, 6)  # 成功
```

`reshape` 的策略是：

- 能零拷贝就走零拷贝
- 不能就走复制路径

所以它更“稳”，但在某些场景会多一次内存开销。

---

## 四、`permute` 的本质：重排维度顺序，不重排底层数据

```python
z = x.reshape(2, 2, 3)
z_perm = z.permute(2, 0, 1)

print(z.shape, z.stride())           # (2,2,3), e.g. stride (6,3,1)
print(z_perm.shape, z_perm.stride()) # (3,2,2), e.g. stride (1,6,3)
print(z_perm.is_contiguous())        # False (common)
```

`permute` 改的是“维度顺序 + stride 规则”，不是立刻拷贝数据。  
因此它和 transpose 一样，结果通常 non-contiguous。

---

## 五、`permute` 后为什么要 `contiguous()`

```python
z_perm_contig = z_perm.contiguous()
z_flat = z_perm_contig.view(-1)  # 现在稳定
```

这一步通常意味着“先整理布局，再做 layout-sensitive 操作（如 view）”。

---

## 六、一张表记住三者区别

| 操作 | 是否优先零拷贝 | 对布局要求 | 常见行为 |
|------|----------------|-----------|---------|
| `view` | 是 | 高（需布局兼容） | 快，但容易在 non-contiguous 上报错 |
| `reshape` | 尽量 | 低 | 更稳，必要时复制 |
| `permute` | 是（重解释） | 无 | 改维度顺序，结果常 non-contiguous |

---

## 七、实战优先级怎么选

- 追求性能、你确认布局安全：优先 `view`
- 日常业务代码、先保证健壮：优先 `reshape`
- 需要改维度顺序（NCHW/NHWC 等）：用 `permute`，后续视情况 `contiguous()`

---

## 总结

1. `view` 是零拷贝重解释，快但严格。  
2. `reshape` 更稳，因为它在必要时会复制。  
3. `permute` 主要是重排维度顺序，不保证连续布局。  
4. 真正踩坑时，优先检查 `shape/stride/is_contiguous`。

---

## 下一篇

下一篇讲 `dtype` 和 `device`：  
为什么它们会直接影响内存占用、数值行为和训练可行性。
