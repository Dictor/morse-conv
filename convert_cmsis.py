from model import MorseCNN
import numpy as np
import torch
import math

# https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/deploying-convolutional-neural-network-on-cortex-m-with-cmsis-nn

def convert_weight(weight):
    weight = weight.detach().numpy()
    real_min = weight.min()
    real_max = weight.max()

    # refer from https://github.com/ARM-software/CMSIS-NN/blob/main/Tests/UnitTest/generate_test_data.py 
    quant_min = -128
    quant_max = 127
    weight_scale = (real_max - real_min) / ((quant_max * 1.0) - quant_min)
    zeropoint = quant_min + int(-real_min / weight_scale + 0.5)
    zeropoint = max(quant_min, min(zeropoint, -quant_min))

    input_scale = 1 # just assume
    output_scale = 1
    real_scale = input_scale * weight_scale / output_scale
    int_multiplier, shift = math.frexp(real_scale)
    int_multiplier = round(int_multiplier * 2147483648) # 2147483648 = 1 << 31

    quantized_weight = np.round(weight / real_scale)
    return (int_multiplier, shift, zeropoint, real_scale, quantized_weight)

def convert_bias(bias, scale):
    quantized_bias = np.round(bias.detach().numpy() / scale)
    return quantized_bias

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

