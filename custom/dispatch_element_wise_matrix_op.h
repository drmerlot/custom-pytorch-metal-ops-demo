#ifndef DISPATCH_RELU_H
#define DISPATCH_RELU_H

#include <torch/extension.h>
#include <string>

// Define a function to add tensors using Metal
torch::Tensor& dispatchElementWiseMatrixOp(const torch::Tensor& input,
                                           const torch::Tensor& width,
                                           const torch::Tensor& height,
                                           const std::string& metalOpName,
                                           torch::Tensor& output);

#endif // DISPATCH_RELU_H