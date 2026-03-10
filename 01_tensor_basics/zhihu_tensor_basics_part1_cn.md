# 从 Shape 到 Stride：为什么转置后 Tensor 会变成 Non-Contiguous？

很多人学 PyTorch 时先记住了 `shape`，但一到 `view` 报错、`transpose` 结果奇怪，就会卡住。  
问题通常不在 shape，而在 `stride`。

这篇的目标是用可运行例子回答三个问题：

1. stride 到底表示什么？
2. 为什么 `transpose` 后常常 non-contiguous？
3. 通用的索引偏移公式如何在 2D 和 3D 中工作？

---

## 一、先建立一个核心公式

对任意 N 维 tensor，元素在底层 storage 中的偏移量可以写成：

```python
offset = sum(index[i] * stride[i] for i in dims)
```

直觉上：

- `shape` 说的是“张量看起来多大”
- `stride` 说的是“某一维索引 +1 时，内存跳多少元素”

---

## 二、2D 例子：为什么 `stride=(4,1)` 很自然

先构造一个连续的 `3x4` 张量：

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x.shape)    # torch.Size([3, 4])
print(x.stride()) # (4, 1)
```

`stride=(4,1)` 的意思是：

- 行索引 +1，要跳过 4 个元素
- 列索引 +1，要跳过 1 个元素

比如 `x[2,3]`：

- offset = `2*4 + 3*1 = 11`
- 正好对应最后一个元素 `11`

这和连续内存 `[0,1,2,...,11]` 完全一致。

---

## 三、transpose 为什么“改读法不改数据”

现在做转置：

```python
y = x.transpose(0, 1)
print(y.shape)          # torch.Size([4, 3])
print(y.stride())       # (1, 4)
print(y.is_contiguous()) # False
```

这里最关键的是：转置通常不会立刻重排底层数据。  
它只是把 stride 从 `(4,1)` 变成 `(1,4)`，等于换了一套读取规则。

因此：

- 同一块 storage 被“重新解释”
- 行/列访问会出现跳跃
- 常见结果就是 `is_contiguous=False`

---

## 四、`contiguous()` 在这里做了什么

```python
z = y.contiguous()
print(z.is_contiguous()) # True
```

这一步会在需要时分配新内存并复制数据，把当前逻辑顺序真正排成连续布局。  
所以 `y` 和 `z` 的值可以相同，但内存布局不同。

---

## 五、3D 例子：公式如何自然扩展

同样的公式直接推广到 3D：

```python
z3 = torch.arange(24).reshape(2, 3, 4)
print(z3.shape)   # torch.Size([2, 3, 4])
print(z3.stride()) # (12, 4, 1)
```

对于 `z3[d,r,c]`，偏移公式就是：

```python
offset = d*12 + r*4 + c*1
```

比如 `z3[1,2,3]`：

- offset = `1*12 + 2*4 + 3 = 23`
- 正好对应最后一个元素

这个例子很重要：它说明 stride 不是 2D 特例，而是 N 维通用规则。

---

## 六、实战调试清单

遇到布局相关问题（尤其 `view` 报错）时，先打印这几项：

```python
print(tensor.shape)
print(tensor.stride())
print(tensor.is_contiguous())
print(tensor.dtype, tensor.device)
```

一句口诀：**先看 shape，再看 stride，再看 contiguous。**

---

## 总结

1. `shape` 是逻辑结构，`stride` 是内存访问步长规则。  
2. 通用访问公式是 `offset = sum(index[i] * stride[i])`。  
3. `transpose` 常常只改 stride，不改底层 storage。  
4. `contiguous()` 会在必要时复制并整理成连续内存。  
5. stride 规则对 2D/3D/N 维都成立。

---

## 下一篇

下一篇会专门讲 `view`、`reshape`、`permute` 的区别：  
什么时候零拷贝，什么时候会复制，为什么同样改形状行为却不同。
