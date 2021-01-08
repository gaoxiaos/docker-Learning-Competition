import random
import torch
device = torch.device("cuda")
a = random.random(size=(100,100))
b = random.random(size=(100,100))
a = a.to_device(device)
b = b.to_device(device)
c = torch.matmul(a,b)
print(c)
