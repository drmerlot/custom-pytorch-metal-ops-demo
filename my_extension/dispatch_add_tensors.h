#ifndef DISPATCH_ADD_TENSORS_H
#define DISPATCH_ADD_TENSORS_H

#include <torch/extension.h>


// dispatch add tensors declaration
torch::Tensor& dispatchAddTensors(const torch::Tensor& a,
                                  const torch::Tensor& b,
                                  torch::Tensor& output);
#endif // DISPATCH_ADD_TENSORS_H