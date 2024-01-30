#include "dispatch_add_tensors.h"
#include "dispatch_matrix_multiply.h"
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

// C++ op dispatching the Metal add tensors shader
torch::Tensor matrix_multiply(const torch::Tensor &A,
                              const torch::Tensor &B) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    // this should go in utils too, find out how to wrap TORCK_CHECK in a function
    TORCH_CHECK(A.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(A.is_contiguous(), "input must be contiguous");

    // !! rethink this !!
    // // Check the supported data types for the function
    TORCH_CHECK(A.scalar_type() == torch::kFloat ||
                A.scalar_type() == torch::kHalf, "Unsupported data type: ", A.scalar_type());

    // // get the required dimentions for the remaining inputs and the output
    // int hA = A.size(0);
    // int wA = A.size(1);
    // int wB = B.size(1);

    // // convert ints to torch tensors to set buffers
    // auto widthA = torch::tensor({wA}, torch::dtype(torch::kInt32)).to(at::kMPS);
    // auto heightA = torch::tensor({hA}, torch::dtype(torch::kInt32)).to(at::kMPS);
    // auto widthB = torch::tensor({wB}, torch::dtype(torch::kInt32)).to(at::kMPS);


    // Allocate the output, with known dim from above
    //torch::Tensor output = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat));
    // Assuming A is m x n and B is n x p
    auto A_size = A.sizes();
    auto B_size = B.sizes();
    int m = A_size[0];
    int p = B_size[1];

    // Create an output tensor of size m x p, with the same data type and device as A
    torch::Tensor output = torch::empty({m, p}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));

    return dispatchMatrixMultiply(A, B, output);
    //return dispatchMatrixMultiply(A, B, widthA, heightA, widthB, result);
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_multiply", &matrix_multiply);
    m.def("add_tensors", &add_tensors);
}