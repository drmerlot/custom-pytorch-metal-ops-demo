#include "utils.h"
#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


// Define a function to add tensors using Metal
torch::Tensor& dispatchAddTensors(const torch::Tensor& a,
                                  const torch::Tensor& b,
                                  torch::Tensor& output) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = a.numel();

        // get the shader source code as
        const char* customKernel = readMetalShader("./my_extension/metal/add_tensors.metal");

        // Load the custom soft shrink shader.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:customKernel]
                                                                  options:nil
                                                                    error:&error];
        // free the memory from const char
        delete[] customKernel;

        // check the library was created with CUSTOM_KERNEL
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        //
        std::string kernel_name = "addTensors";
        id<MTLFunction> customAddTensorsFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(customAddTensorsFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> addTensorsPSO = [device newComputePipelineStateWithFunction:customAddTensorsFunction error:&error];
        TORCH_CHECK(addTensorsPSO, error.localizedDescription.UTF8String);

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
            [computeEncoder setComputePipelineState:addTensorsPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(a) offset:a.storage_offset() * a.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(b) offset:b.storage_offset() * b.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:2];

            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = addTensorsPSO.maxTotalThreadsPerThreadgroup;
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
