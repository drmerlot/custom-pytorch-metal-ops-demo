#include "utils.h"
#include "dispatch_matrix_multiply.h"
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

        // convert ints to torch tensors to set buffers
        auto widthATensor = torch::tensor({widthA}, torch::dtype(torch::kInt32)).to(at::kMPS);
        auto heightATensor = torch::tensor({heightA}, torch::dtype(torch::kInt32)).to(at::kMPS);
        auto widthBTensor = torch::tensor({widthB}, torch::dtype(torch::kInt32)).to(at::kMPS);

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

        // Create a compute pipeline state object for the matrix multiply
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

            // Encode the pipeline state object
            [computeEncoder setComputePipelineState:matrixMultiplyPSO];

            // Put the torch::Tensor matrix values in buffers
            [computeEncoder setBuffer:getMTLBufferStorage(A) offset:A.storage_offset() * A.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(widthATensor) offset:widthATensor.storage_offset() * widthATensor.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(heightATensor) offset:heightATensor.storage_offset() * heightATensor.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(widthBTensor) offset:widthBTensor.storage_offset() * widthBTensor.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(result) offset:result.storage_offset() * result.element_size() atIndex:5];

            // Set grid size to match the dimensions of the output matrix
            MTLSize gridSize = MTLSizeMake(widthB, heightA, 1);

            // Calculate a thread group size
            NSUInteger threadGroupSize = matrixMultiplyPSO.maxTotalThreadsPerThreadgroup;
            //NSUInteger threadGroupSize = 1;

            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1); // Adjust as needed

            // Encode the compute command
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
            //[commandBuffer waitUntilCompleted];

            // // Release the buffers by setting them to nil
            // widthABuffer = nil;
            // heightABuffer = nil;
            // widthBBuffer = nil;
        });
    }

    return result;
}

            // example of and int buffer (probably does not work?)
            // buffer for widthB
            //id<MTLBuffer> widthBBuffer = [device newBufferWithBytes:&widthB
            //                                     length:sizeof(int)
            //                                   options:MTLResourceStorageModeShared];
            //[computeEncoder setBuffer:widthBBuffer offset:0 atIndex:4];

