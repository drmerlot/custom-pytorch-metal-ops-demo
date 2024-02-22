import math
import torch
import torch.nn as nn
from torch.autograd import Function
import my_extension_cpp


# Define a wrapper functions
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Call the C++ function
    return my_extension_cpp.add_tensors(a, b)


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Kaiming uniform initialization method
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Initialize bias to zero
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return CustomLinearFunction.apply(x, self.weight, self.bias if self.bias is not None else None)


class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias=None):
        ctx.save_for_backward(input, weights, bias)
        input = input.contiguous()
        weights = weights.t().contiguous()
        output = my_extension_cpp.matrix_multiply(input, weights)
        if bias is not None:
            bias_2d = bias.contiguous()
            # Assuming my_extension_cpp.matrix_add can handle broadcasting the bias
            output = my_extension_cpp.matrix_add(output, bias_2d)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, bias = ctx.saved_tensors
        grad_inputs = grad_weights = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            weights = weights.contiguous()
            grad_inputs = my_extension_cpp.matrix_multiply(grad_output, weights)

        if ctx.needs_input_grad[1]:
            input = input.t().contiguous()
            grad_output = grad_output.contiguous()
            grad_weights = my_extension_cpp.matrix_multiply(input, grad_output)
            grad_weights = grad_weights.t()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_inputs, grad_weights, grad_bias


class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        # No additional parameters to initialize

    def forward(self, inp):
        # Call the C++ function that invokes the Metal shader
        inp_cont = inp.contiguous()
        return CustomReLUFunction.apply(inp_cont)

    def extra_repr(self):
        return "MPS-based ReLU"


class CustomReLUFunction(Function):
    @staticmethod
    def forward(ctx, inp):
        # Store input for use in the backward pass
        ctx.save_for_backward(inp)
        inp_cont = inp.contiguous()
        return my_extension_cpp.relu(inp_cont)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the stored input
        inp, = ctx.saved_tensors
        # Create gradient tensor, only allowing gradients to flow where input > 0
        grad_input = grad_output.clone()
        grad_input[inp < 0.0] = 0.0
        return grad_input
