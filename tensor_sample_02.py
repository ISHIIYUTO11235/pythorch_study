import torch

x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
#これが学習すべきパラメータ　重みらしい
w = torch.randn(5, 3, requires_grad = True)
b = torch.randn(3, requires_grad = True)

y = x @ w + b
print(y.shape)

loss = y.sum()
loss.backward()

print(f"ロスの勾配(gradi)")
print(w.grad)
print(f"勾配の形{w.grad.shape}")


lr = 0.1
print(f"更新前のW{w}")

with torch.no_grad():
    w -= lr * w.grad


w.grad.zero_()
print(f"更新後のW{w}")