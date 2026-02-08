import torch

li = []
ai = []
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([5, 6, 7, 8])
ai.append(a)
ai.append(b)
c = torch.cat(ai, dim = 0)
print(c.shape)
li.append(c)
print(len(li))