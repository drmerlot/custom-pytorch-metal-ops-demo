#include "utils.h"
#include "dispatch_matrix_multiply.h"
#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


// Define a function to multiply two matricies

torch::Tensor& dispatchMatrixMultiply(const torch::Tensor& A,
                                      const torch::Tensor& B,
                                      const torch::Tensor& widthA,
                                      const torch::Tensor& heightA,
                                      const torch::Tensor& widthB,
                                      const int& hA,
                                      const int& wB,
                                      torch::Tensor& result) {
    @autoreleasepool {
        // Retrieve the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Load the shader source and create the compute kernel library
        const char* customKernel = readMetalShader("./my_extension/metal/matrix_multiply.metal");
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:customKernel]
                                                                  options:nil
                                                                    error:&error];
        delete[] customKernel;  // Free the shader source memory
        TORCH_CHECK(customKernelLibrary, "Failed to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        // Create the compute kernel function
        std::string kernel_name = "matrixMultiply";
        id<MTLFunction> customMatrixMultiplyFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(customMatrixMultiplyFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object
        id<MTLComputePipelineState> matrixMultiplyPSO = [device newComputePipelineStateWithFunction:customMatrixMultiplyFunction error:&error];
        TORCH_CHECK(matrixMultiplyPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        // Dispatch the kernel
        dispatch_sync(serialQueue, ^(){
            // Start a compute pass
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object
            [computeEncoder setComputePipelineState:matrixMultiplyPSO];

            // Set the tensor buffers
            [computeEncoder setBuffer:getMTLBufferStorage(A) offset:A.storage_offset() * A.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(widthA) offset:widthA.storage_offset() * widthA.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(heightA) offset:heightA.storage_offset() * heightA.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(widthB) offset:widthB.storage_offset() * widthB.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(result) offset:result.storage_offset() * result.element_size() atIndex:5];

            // Set grid and thread group sizes
            //MTLSize gridSize = MTLSizeMake(widthB.item<int>(), heightA.item<int>(), 1);

            // hard set the gridSize and threadGroupSize
            // same dims as the output matrix
            MTLSize gridSize = MTLSizeMake(wB, hA, 1);

            // Query the maximum threads per thread group from the device
            NSUInteger maxThreadsPerThreadgroup = matrixMultiplyPSO.maxTotalThreadsPerThreadgroup;

            // Choose a standard thread group size (like 8x8 or 16x16)
            NSUInteger threadGroupWidth = 32;
            NSUInteger threadGroupHeight = 32;

            // // reduce by a factor of two until thread group fits under max threads per group
            while (threadGroupWidth * threadGroupHeight > maxThreadsPerThreadgroup) {
                threadGroupWidth /= 2;
                threadGroupHeight /= 2;
            }

            // Adjust thread group size to fit the gridSize if necessary
            if (wB % threadGroupWidth != 0) {
                threadGroupWidth = wB % threadGroupWidth;
            }
            if (hA % threadGroupHeight != 0) {
                threadGroupHeight = hA % threadGroupHeight;
            }

            MTLSize threadgroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);


            // Dispatch the compute command
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];

            // Commit the work
            torch::mps::synchronize();  // or commit? or synchronize?
        });
    }
    return result;
}
