#include "utils.h"

#include <fstream>
#include <sstream>
#include <cstring>

// function for reading in metal shader files as const char
const char* readMetalShader(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open Metal shader file: " + filename);
    }
    // read the data from the buffer
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string str = buffer.str();

    // return as char* for use in newLibraryWithSource
    char* shader_code = new char[str.length() + 1];
    std::strcpy(shader_code, str.c_str());
    return shader_code;
}

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}