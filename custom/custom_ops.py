import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import custom_cpp


# Define a wrapper function
def add_tensors(a: Tensor, b: Tensor) -> Tensor:
    # Call the C++ function
    return custom_cpp.add_tensors(a, b)


class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: nn.Parameter = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights using Kaiming uniform initialization method
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return CustomLinearFunction.apply(x, self.weight, self.bias if self.bias is not None else None)


class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx: Function, input: Tensor, weights: Tensor, bias: Tensor | None = None) -> Tensor:
        ctx.save_for_backward(input, weights, bias)
        input = input.contiguous()
        weights = weights.t().contiguous()
        output: Tensor = custom_cpp.matrix_multiply(input, weights)
        if bias is not None:
            bias_2d = bias.contiguous()
            output = custom_cpp.matrix_add(output, bias_2d)
        return output

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        input, weights, bias = ctx.saved_tensors
        grad_inputs: Tensor | None = None
        grad_weights: Tensor | None = None
        grad_bias: Tensor | None = None

        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            weights = weights.contiguous()
            grad_inputs = custom_cpp.matrix_multiply(grad_output, weights)

        if ctx.needs_input_grad[1]:
            input = input.t().contiguous()
            grad_output = grad_output.contiguous()
            grad_weights = custom_cpp.matrix_multiply(input, grad_output)
            grad_weights = grad_weights.t()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_inputs, grad_weights, grad_bias


class CustomReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: Tensor) -> Tensor:
        inp_cont = inp.contiguous()
        return CustomReLUFunction.apply(inp_cont)

    def extra_repr(self) -> str:
        return "MPS-based ReLU"


class CustomReLUFunction(Function):
    @staticmethod
    def forward(ctx: Function, inp: Tensor) -> Tensor:
        ctx.save_for_backward(inp)
        inp_cont = inp.contiguous()
        return custom_cpp.relu(inp_cont)

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> Tensor:
        inp, = ctx.saved_tensors
        grad_input: Tensor = grad_output.clone()
        grad_input[inp < 0.0] = 0.0
        return grad_input


class CustomLeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: Tensor) -> Tensor:
        inp_cont = inp.contiguous()
        return CustomLeakyReLUFunction.apply(inp_cont)

    def extra_repr(self) -> str:
        return "MPS-based LeakyReLU"


class CustomLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx: Function, inp: Tensor) -> Tensor:
        ctx.save_for_backward(inp)
        inp_cont = inp.contiguous()
        return custom_cpp.leaky_relu(inp_cont)

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> Tensor:
        inp, = ctx.saved_tensors
        grad_input: Tensor = grad_output.clone()
        grad_input[inp < 0.0] = 0.01
        return grad_input
