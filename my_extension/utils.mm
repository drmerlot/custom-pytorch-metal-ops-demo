#include "utils.h"
#include "../custom_metal_ops.h"
#import <Metal/Metal.h>

#include <fstream>
#include <sstream>
#include <cstring>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

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

// function for reading in compiled metal shader library
id<MTLLibrary> readCompiledMetalLibrary(NSError *error,
                                        id<MTLDevice> device) {
        // Issue: this is currently hard coded to var names within the
        // compiled lib
        //Get the compiled library data
        NSData* libraryData = [NSData dataWithBytesNoCopy:__custom_metal_ops_metallib
                                            length:__custom_metal_ops_metallib_len
                                      freeWhenDone:NO];
        // convert to dispatch_data_t
        const void *bytes = [libraryData bytes];
        size_t length = [libraryData length];
        dispatch_data_t dispatchData = dispatch_data_create(bytes, length, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), DISPATCH_DATA_DESTRUCTOR_DEFAULT);

        // define the customKernelLibrary
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithData:dispatchData error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        // get rid of the dispatchData if possible
        if (dispatchData) dispatch_release(dispatchData);

        // return the finished library
        return customKernelLibrary;
}