#ifndef DISPATCH_MATRIX_ADD_H
#define DISPATCH_MATRIX_ADD_H

#include <torch/extension.h>

// dispatch add tensors declaration
torch::Tensor& dispatchMatrixAdd(const torch::Tensor& A,
                                 const torch::Tensor& B,
                                 const torch::Tensor& widthA,
                                 const torch::Tensor& heightA,
                                 const int& wA,
                                 const int& hA,
                                 torch::Tensor& result);

#endif // DISPATCH_MATRIX_ADD_H