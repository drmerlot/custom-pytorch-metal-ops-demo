#ifndef UTILS_H
#define UTILS_H

#include <torch/extension.h>
#import <Metal/Metal.h>

// Declaration of getMTLBufferStorage function
id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor);

// Declaration of readMetalShader function
const char* readMetalShader(const std::string& filename);

// Declaration of readCompliedMetalLibrary
id<MTLLibrary> readCompiledMetalLibrary(NSError *error, id<MTLDevice> device);

#endif // UTILS_H
