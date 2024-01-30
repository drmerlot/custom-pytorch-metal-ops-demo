#include "utils.h"
#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


// Define a function to multiply two matricies
torch::Tensor& dispatchMatrixMultiply(const torch::Tensor& A,
                                      const torch::Tensor& B,
                                      const int& widthA,
                                      const int& heightA,
                                      const int& widthB,
                                      torch::Tensor& result) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = A.numel(); // !! THIS IS NOT LIKELY RIGHT!

        // get the shader source code as
        const char* customKernel = readMetalShader("./my_extension/metal/matrix_multiply.metal");

        // Load the custom soft shrink shader.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:customKernel]
                                                                  options:nil
                                                                    error:&error];
        // free the memory from const char
        delete[] customKernel;

        // check the library was created with CUSTOM_KERNEL
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        // get the custom function
        std::string kernel_name = "matrixMultiply";
        id<MTLFunction> customMatrixMultiplyFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(customMatrixMultiplyFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> matrixMultiplyPSO = [device newComputePipelineStateWithFunction:customMatrixMultiplyFunction error:&error];
        TORCH_CHECK(matrixMultiplyPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:matrixMultiplyPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(A) offset:a.storage_offset() * A.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(B) offset:b.storage_offset() * B.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(widthA) offset:widthA.storage_offset() * widthA.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(heightA) offset:heightA.storage_offset() * heightA.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(widthB) offset:widthB.storage_offset() * widthB.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(result) offset:result.storage_offset() * result.element_size() atIndex:5];

            // set grid size
            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = matrixMultiplyPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numThreads) {
                threadGroupSize = numThreads;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

    return output;
}
