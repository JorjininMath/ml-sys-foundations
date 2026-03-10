# dtype 与 device：精度、内存和训练可行性的第一课

在 Tensor 基础里，`shape` 和 `stride` 解释了“怎么读数据”。  
这篇继续往系统层走一步：**dtype 和 device 决定你“用什么算、在哪算”**。

---

## 一、先看 dtype：同样形状，内存占用可以差很多

```python
import torch

x = torch.arange(6).reshape(2, 3)
x_fp32 = x.to(torch.float32)
x_fp16 = x.to(torch.float16)
x_bf16 = x.to(torch.bfloat16)

for name, t in [("fp32", x_fp32), ("fp16", x_fp16), ("bf16", x_bf16)]:
    print(name, t.element_size(), t.nbytes)
```

常见结果：

- `fp32`：4 bytes/element
- `fp16`：2 bytes/element
- `bf16`：2 bytes/element

同样元素个数，`fp16/bf16` 内存通常是 `fp32` 的一半。

---

## 二、为什么要区分 fp16 和 bf16

两者的关键区别在于**指数位宽**：

- `fp16`：5 位指数，11 位尾数 → 动态范围约 ±65504，大值更容易溢出
- `bf16`：8 位指数（与 fp32 相同），7 位尾数 → 动态范围与 fp32 一致，精度更低但不易溢出

```python
large = torch.tensor(65504.0)
print((large * 2).to(torch.float16))   # inf，超出 fp16 最大值
print((large * 2).to(torch.bfloat16))  # 131008.0，bf16 和 fp32 共享指数范围，可以表示
```

一句话总结：`fp16` 和 `bf16` 内存占用相同，但 `bf16` 借了 fp32 的指数位宽，换来更稳定的数值范围，代价是精度比 `fp16` 低。

---

## 三、device：代码在哪个硬件上执行

推荐写一个设备检测函数，避免硬编码：

```python
import torch

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = select_device()
print(device)  # cuda / mps / cpu，取决于当前机器
```

这样代码在不同机器上都能跑，不用每次手动改字符串。

---

## 四、推荐写法：一行同时改 dtype + device

```python
x_device_fp16 = x.to(device=device, dtype=torch.float16)
```

这种写法的优点：

- 显式：你一眼知道“搬到哪、转成啥”
- 稳定：减少多步转换造成的混乱

---

## 五、一个非常常见的坑：整数张量不能直接求梯度

```python
x = torch.arange(6).reshape(2, 3)
x.requires_grad_(True)  # RuntimeError
```

要参与梯度计算，通常需要浮点 dtype：

```python
x_grad_ok = x.to(torch.float32).requires_grad_(True)
```

这也是很多初学者在训练代码里第一次遇到的报错来源。

---

## 六、实战清单（写训练脚本前先过一遍）

1. 明确 dtype（`fp32` 还是 `fp16/bf16`）  
2. 明确 device（`cpu/cuda/mps`）  
3. 需要反向传播时，确保是浮点 dtype  
4. 打印 `tensor.dtype` 和 `tensor.device` 做 sanity check  

---

## 总结

1. `dtype` 影响数值表示和内存占用。  
2. `device` 决定计算发生在哪个硬件上。  
3. `.to(device=..., dtype=...)` 是最清晰的转换入口。  
4. 梯度计算场景里，优先使用浮点张量。  

---

## 下一步

Tensor 基础到这里就比较完整了。  
下一阶段可以进入 `02_autograd_from_scratch`，把“梯度是怎么被算出来的”真正跑通。
