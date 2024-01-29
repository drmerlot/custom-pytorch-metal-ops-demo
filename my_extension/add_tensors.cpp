#include "dispatch_add_tensors.h"
#include <torch/extension.h>

// C++ op dispatching the Metal add tensors shader
torch::Tensor add_tensors(const torch::Tensor &a, const torch::Tensor &b) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    TORCH_CHECK(a.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(a.is_contiguous(), "input must be contiguous");

    // Check the supported data types for the function
    TORCH_CHECK(a.scalar_type() == torch::kFloat ||
                a.scalar_type() == torch::kHalf, "Unsupported data type: ", a.scalar_type());

    // Allocate the output, same shape as the a
    torch::Tensor output = torch::empty_like(a);

    return dispatchAddTensors(a, b, output);
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors);
}