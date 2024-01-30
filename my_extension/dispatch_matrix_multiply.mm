#include "utils.h"
#include "dispatch_matrix_multiply.h"
#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


// Define a function to multiply two matricies

torch::Tensor& dispatchMatrixMultiply(const torch::Tensor& A,
                                      const torch::Tensor& B,
                                      // const torch::Tensor& widthA,
                                      // const torch::Tensor& heightA,
                                      // const torch::Tensor& widthB,
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
            [computeEncoder setBuffer:getMTLBufferStorage(result) offset:result.storage_offset() * result.element_size() atIndex:2];

            // [computeEncoder setBuffer:getMTLBufferStorage(widthA) offset:0 atIndex:2];
            // [computeEncoder setBuffer:getMTLBufferStorage(heightA) offset:0 atIndex:3];
            // [computeEncoder setBuffer:getMTLBufferStorage(widthB) offset:0 atIndex:4];

            // Set grid and thread group sizes
            //MTLSize gridSize = MTLSizeMake(widthB.item<int>(), heightA.item<int>(), 1);

            // hard set hthe gridSize and threadGroupSize
            MTLSize gridSize = MTLSizeMake(2, 2, 1); // For a 2x2 output matrix
            NSUInteger threadGroupWidth = 2;
            NSUInteger threadGroupHeight = 2;
            MTLSize threadgroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1); // Creating a 2x2 thread group


            // Dispatch the compute command
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];

            // Commit the work
            torch::mps::commit();  // or commit? or synchronize?
        });
    }
    return result;
}
