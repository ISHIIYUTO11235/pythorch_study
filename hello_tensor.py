import torch

x = torch.tensor(2.0)
# w = 3.0(重み), b = 1.0（バイアス）

w = torch.tensor(3.0, requires_grad = True)
b = torch.tensor(1.0,requires_grad = True)

y = w*x+b
print(f"計算結果y;{y}")

#バックパブロケーション発動 
y.backward()

print(w.grad.item())
print(b.grad.item())