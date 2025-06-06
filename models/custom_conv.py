import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Attempt to load the compiled extension. If not built yet, build on the fly.
_ext_path = os.path.join(os.path.dirname(__file__), '..', 'build')
try:
    custom_conv = load(
        name='custom_conv',
        build_directory=_ext_path,
        sources=[
            os.path.join('csrc', 'custom_conv.cpp'),
            os.path.join('csrc', 'custom_conv_kernel.cu'),
        ]
    )
except (RuntimeError, FileNotFoundError):
    custom_conv = load(
        name='custom_conv',
        sources=[
            os.path.join('csrc', 'custom_conv.cpp'),
            os.path.join('csrc', 'custom_conv_kernel.cu'),
        ]
    )

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        outputs = custom_conv.forward(input, weight, bias, stride, padding)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_input = torch.nn.grad.conv2d_input(
            input.shape, weight, grad_output, stride=stride, padding=padding
        )
        grad_weight = torch.nn.grad.conv2d_weight(
            input, weight.shape, grad_output, stride=stride, padding=padding
        )
        grad_bias = grad_output.sum((0, 2, 3)) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return CustomConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding)
