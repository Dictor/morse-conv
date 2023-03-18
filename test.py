from model import MorseCNN
import numpy as np
import torch
import math

input = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
inputt = torch.from_numpy(input)

model = MorseCNN()
model.load_state_dict(torch.load("./model"))

out = model.layer1[0](inputt.float())
print(out)