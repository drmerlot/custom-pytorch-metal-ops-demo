#ifndef UTILS_H
#define UTILS_H

#include <torch/extension.h>

// Declaration of readMetalShader function
const char* readMetalShader(const std::string& filename);

// Declaration of getMTLBufferStorage function
id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor);

#endif // UTILS_H
