from model import MorseCNN
import numpy as np
import torch
import math

# https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/deploying-convolutional-neural-network-on-cortex-m-with-cmsis-nn

def convert_weight(weight):
    weight = weight.detach().numpy()
    real_min = weight.min()
    real_max = weight.max()

    # assume that apply same quantization on whole layer
    # affine quantization with 0 zero point (symmetric quantization)
    quantization_target_bit = 8
    double_multiplier = (real_max - real_min) / (2 ** quantization_target_bit - 1)
    int_multiplier, shift = math.frexp(double_multiplier)
    int_multiplier = int(int_multiplier * 2147483648) # 2147483648 = 1 << 31

    quantized_weight = np.round(weight / double_multiplier)
    return (int_multiplier, shift, quantized_weight)

def format_1d_tensor(t):
    output = "{"
    for i in range(t.shape[0]):
        output += "{0:d}".format(t[i])
        if i != t.shape[0]-1:
            output += ", "
    output += "}"
    return output

def format_2d_tensor(t):
    height = t.shape[0]
    width = t.shape[1]

    output = "{"
    for j in range(height):
        for k in range(width):
            output += "{0:d}".format(t[j][k])
            if k != width-1 or j != height-1:
                output += ", "
    output += "}"
    return output

def format_3d_tensor(t):
    channel = t.shape[0]
    height = t.shape[1]
    width = t.shape[2]

    output = "{"
    for i in range(channel):
        for j in range(height):
            for k in range(width):
                output += "{0:d}".format(t[i][j][k])
                if k != width-1 or j != height-1 or i != channel-1:
                    output += ", "
    output += "}"
    return output


print("weight quantilizer & C array formatter")
model = MorseCNN()
model.load_state_dict(torch.load("./model"))

print("Conv 1 ############")
m, s, w = convert_weight(model.layer1[0].weight)
print("weight (m={}, s={}) : ".format(m, s), format_3d_tensor(
    np.int8(w)))
m, s, w = convert_weight(model.layer1[0].bias)
print("bias : (m={}, s={}) : ".format(m, s), format_1d_tensor(
    np.int8(w)))
print("###################")

print("Conv 2 ############")
m, s, w = convert_weight(model.layer2[0].weight)
print("weight (m={}, s={}) : ".format(m, s), format_3d_tensor(
    np.int8(w)))
m, s, w = convert_weight(model.layer2[0].bias)
print("bias : (m={}, s={}) : ".format(m, s), format_1d_tensor(
    np.int8(w)))
print("###################")

print("FC ############")
m, s, w = convert_weight(model.fc.weight)
print("weight (m={}, s={}) : ".format(m, s), format_2d_tensor(
    np.int8(w)))
m, s, w = convert_weight(model.fc.bias)
print("bias : (m={}, s={}) : ".format(m, s), format_1d_tensor(
    np.int8(w)))
print("###################")

