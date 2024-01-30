#ifndef DISPATCH_MATRIX_MULTIPLY_H
#define DISPATCH_MATRIX_MULTIPLY_H

#include <torch/extension.h>

// dispatch add tensors declaration
torch::Tensor& dispatchMatrixMultiply(const torch::Tensor& A,
                                      const torch::Tensor& B,
                                      const int& widthA,
                                      const int& heightA,
                                      const int& widthB,
                                      torch::Tensor& result);

#endif // DISPATCH_MATRIX_MULTIPLY_H