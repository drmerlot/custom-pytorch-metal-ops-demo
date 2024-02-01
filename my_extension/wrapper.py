import torch
import torch.nn as nn
from torch.autograd import Function
import my_extension_cpp


# Define a wrapper functions
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Call the C++ function
    return my_extension_cpp.add_tensors(a, b)


class CustomLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(CustomLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inp):
        inp_contiguous = inp.contiguous()
        return CustomLinearFunction.apply(inp_contiguous, self.weight)

    def extra_repr(self):
        return 'input_features={}, output_features={}'.format(
            self.input_features, self.output_features
        )


class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight):
        ctx.save_for_backward(inp, weight)
        inp_cont = inp.contiguous()
        weight_cont = weight.contiguous()
        return my_extension_cpp.matrix_multiply(inp_cont, weight_cont.t())

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            # Gradient with respect to input
            grad_input = my_extension_cpp.matrix_multiply(grad_output, weight.t())

        if ctx.needs_input_grad[1]:
            # Gradient with respect to weight
            inp_t = inp.t().contiguous()
            grad_output = grad_output.contiguous()
            grad_weight = my_extension_cpp.matrix_multiply(inp_t, grad_output)
        return grad_input, grad_weight


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
        #grad_input = grad_output.clone()
        #grad_input[inp < 0.0] = 0.0
        grad_output = grad_output.contiguous()
        grad_input = torch.where(inp > 0, grad_output, 0.01 * grad_output)
        return grad_input
