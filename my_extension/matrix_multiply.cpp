#include "dispatch_matrix_multiply.h"
#include <torch/extension.h>

// C++ op dispatching the Metal add tensors shader
torch::Tensor add_tensors(const torch::Tensor &A,
                          const torch::Tensor &B) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    TORCH_CHECK(A.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(B.is_contiguous(), "input must be contiguous");

    // !! rethink this !!
    // // Check the supported data types for the function
    // TORCH_CHECK(a.scalar_type() == torch::kFloat ||
    //             a.scalar_type() == torch::kHalf, "Unsupported data type: ", a.scalar_type());

    // get the required dimentions for the remaining inputs and the output
    int widthA = A.size(0);
    int heightA = A.size(1);
    int widthB = B.size(1);


    // Allocate the output, with known dim from above
    torch::Tensor output = torch::empty({3, 4});

    return dispatchMatrixMultiply(A, B, widthA, heightA, widthB, output);
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors);
}