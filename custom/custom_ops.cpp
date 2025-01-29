#include "dispatch_matrix_multiply.h"
#include "dispatch_matrix_add.h"
#include "dispatch_element_wise_matrix_op.h"
#include <torch/extension.h>
#include <string>


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
    int hA = A.size(0);
    int wA = A.size(1);
    int wB = B.size(1);

    // // convert ints to torch tensors to set buffers
    auto widthA = torch::tensor({wA}, torch::dtype(torch::kInt32)).to(at::kMPS);
    auto heightA = torch::tensor({hA}, torch::dtype(torch::kInt32)).to(at::kMPS);
    auto widthB = torch::tensor({wB}, torch::dtype(torch::kInt32)).to(at::kMPS);


    // Allocate the output, with known dim from above
    //torch::Tensor output = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat));
    // Assuming A is m x n and B is n x p
    auto A_size = A.sizes();
    auto B_size = B.sizes();
    int m = A_size[0];
    int p = B_size[1];

    // Create an output tensor of size m x p, with the same data type and device as A
    torch::Tensor output = torch::empty({m, p}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));

    //return dispatchMatrixMultiply(A, B, output);
    return dispatchMatrixMultiply(A, B, widthA, heightA, widthB, hA, wB, output);
}


// C++ op dispatching the Metal add tensors shader
torch::Tensor matrix_add(const torch::Tensor &A,
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
    int hA = A.size(0);
    int wA = A.size(1);

    // // convert ints to torch tensors to set buffers
    auto widthA = torch::tensor({wA}, torch::dtype(torch::kInt32)).to(at::kMPS);
    auto heightA = torch::tensor({hA}, torch::dtype(torch::kInt32)).to(at::kMPS);

    // Allocate the output, with known dim from above
    torch::Tensor output = torch::empty_like(A);

    return dispatchMatrixAdd(A, B, widthA, heightA, wA, hA, output);
}

// C++ op dispatching the Metal add tensors shader
torch::Tensor relu(const torch::Tensor &input) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    // this should go in utils too, find out how to wrap TORCK_CHECK in a function
    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // !! rethink this !!
    // // Check the supported data types for the function
    TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                input.scalar_type() == torch::kHalf, "Unsupported data type: ", input.scalar_type());

    // // get the required dimentions for the remaining inputs and the output
    int W = input.size(1);
    int H = input.size(0);

    // // convert ints to torch tensors to set buffers
    auto width = torch::tensor({W}, torch::dtype(torch::kInt32)).to(at::kMPS);
    auto height = torch::tensor({H}, torch::dtype(torch::kInt32)).to(at::kMPS);

    // Allocate the output, same shape as the input
    torch::Tensor output = torch::empty_like(input);

    // define the metal op to use
    std::string relu = "relu";

    //return dispatchMatrixMultiply(A, B, output);
    return dispatchElementWiseMatrixOp(input, width, height, relu, output);
}

// C++ op dispatching the Metal add tensors shader
torch::Tensor leaky_relu(const torch::Tensor &input) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    // this should go in utils too, find out how to wrap TORCK_CHECK in a function
    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // !! rethink this !!
    // // Check the supported data types for the function
    TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                input.scalar_type() == torch::kHalf, "Unsupported data type: ", input.scalar_type());

    // // get the required dimentions for the remaining inputs and the output
    int W = input.size(1);
    int H = input.size(0);

    // // convert ints to torch tensors to set buffers
    auto width = torch::tensor({W}, torch::dtype(torch::kInt32)).to(at::kMPS);
    auto height = torch::tensor({H}, torch::dtype(torch::kInt32)).to(at::kMPS);

    // Allocate the output, same shape as the input
    torch::Tensor output = torch::empty_like(input);

    // define the metal op to use
    std::string leaky_relu = "leakyRelu";

    //return dispatchMatrixMultiply(A, B, output);
    return dispatchElementWiseMatrixOp(input, width, height, leaky_relu, output);
}


// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_multiply", &matrix_multiply);
    m.def("matrix_add", &matrix_add);
    m.def("relu", &relu);
    m.def("leaky_relu", &relu);
}