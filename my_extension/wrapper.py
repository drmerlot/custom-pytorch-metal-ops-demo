import torch
import my_extension_cpp


# Define a wrapper functions
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Call the C++ function
    return my_extension_cpp.add_tensors(a, b)


# Define a wrapper functions
def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Call the C++ function
    return my_extension_cpp.matrix_multiply(a, b)
