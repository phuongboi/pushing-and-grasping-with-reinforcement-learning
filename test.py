import numpy as np
import torch
import torchvision
from torch import nn
a = np.array([[2, 3, 5], [5, 9, 1]])
a = torch.from_numpy(a).unsqueeze(0)
print(a.shape)
b = a.reshape(a.shape[0], a.shape[1]*a.shape[2])
print(b)
print(np.argmax(a.detach().numpy()))
print(np.argmax(b.detach().numpy()))
model = torchvision.models.resnet18(pretrained=False)
input  = torch.from_numpy(np.zeros((1, 3, 224, 224)).astype(np.float32))
feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
img_feat = feature_extractor(input)
avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

out = avg_pool(img_feat)
print(out.shape)
linear = nn.Linear(in_features=512, out_features=16, bias=True)
out_linear = linear(out)

print(out_linear.shape)
