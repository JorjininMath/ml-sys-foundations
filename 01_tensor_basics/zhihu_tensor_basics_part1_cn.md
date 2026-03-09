# 从 Shape 到 Stride：为什么转置后 Tensor 会变成 Non-Contiguous？

很多人刚学 PyTorch 时，会把 Tensor 理解成“有 shape 的数组”。这个理解不算错，但不够。

真正决定很多行为差异的，往往是另一个信息：`stride`。

这篇用两个最小脚本来解释：

- `01_tensor_basics/tensor_shapes_and_strides.py`
- `01_tensor_basics/contiguous_vs_noncontiguous.py`

## 一、shape 和 stride 不是一回事

先看一个基础张量 `a`（2x3）：

```python
a = torch.arange(6).reshape(2, 3)
# shape = (2, 3)
# stride = (3, 1)
# is_contiguous = True
```

然后做转置：

```python
b = a.transpose(0, 1)
# shape = (3, 2)
# stride = (1, 3)
# is_contiguous = False
```

再做：

```python
c = b.contiguous()
# shape = (3, 2)
# is_contiguous = True
```

核心点：

- `shape` 描述逻辑维度。
- `stride` 描述每个维度索引 +1 时，内存要跳多少元素。

## 二、为什么 `view` 会失败，`reshape` 却可能成功

在 `contiguous_vs_noncontiguous.py` 里：

```python
y = x.t()          # non-contiguous
z = y.contiguous() # contiguous
torch.equal(y, z)  # True，语义值相同
```

然后比较行为：

```python
y.view(-1)    # RuntimeError：布局不满足零拷贝重解释
z.view(-1)    # OK
y.reshape(-1) # OK，必要时内部会复制
```

一句话记住：

- `view` 快但严格
- `reshape` 更灵活
- `contiguous()` 是布局整理步骤

## 三、为什么转置后常常 non-contiguous（直觉图）

原始 `a`：

```text
a = [[0, 1, 2],
     [3, 4, 5]]
memory = [0, 1, 2, 3, 4, 5]
stride = (3, 1)
```

转置后 `b = a.T`：

```text
b = [[0, 3],
     [1, 4],
     [2, 5]]
stride = (1, 3)
```

关键是：转置通常不立刻重排内存，而是改“读取规则”。  
所以读取同一行元素时会出现跳跃访问，不再是连续块，因此常见 `is_contiguous = False`。

## 四、调试清单（非常实用）

遇到布局相关问题时，先打印：

```python
print(tensor.shape)
print(tensor.stride())
print(tensor.is_contiguous())
print(tensor.dtype, tensor.device)
```

如果 `view` 在转置后报错，优先尝试：

```python
tensor = tensor.contiguous()
tensor.view(...)
```

## 五、三条结论

1. `shape` 决定“看起来的形状”，`stride` 决定“内存里怎么走”。
2. `transpose()` 常改 stride 而不是立即复制数据。
3. `view` 这类布局敏感操作依赖 contiguous 兼容内存。

## 下一篇

下一篇聚焦：`view`、`reshape`、`permute` 的差异与拷贝行为，给出更系统的判断方法。
