import numpy as np
import torch
a = np.array([[2, 3, 5], [5, 9, 1]])
a = torch.from_numpy(a).unsqueeze(0)
print(a.shape)
b = a.reshape(a.shape[0], a.shape[1]*a.shape[2])
print(b)
print(np.argmax(a.detach().numpy()))
print(np.argmax(b.detach().numpy()))
