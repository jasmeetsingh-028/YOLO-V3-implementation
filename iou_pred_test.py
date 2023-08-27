import torch

x = torch.randn((2, 4))

print(x)
print(x[..., 0])
