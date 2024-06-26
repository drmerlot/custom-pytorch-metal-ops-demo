#include "utils.h"
#include <torch/extension.h>
#include <string>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


// Define a function to add tensors using Metal
torch::Tensor& dispatchElementWiseMatrixOp(const torch::Tensor& input,
                                           const torch::Tensor& width,
                                           const torch::Tensor& height,
                                           const std::string& metalOpName,
                                           torch::Tensor& output) {
    @autoreleasepool {
        // Retrieve the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // currently read-in hard hard coded var names
        id<MTLLibrary> customKernelLibrary = readCompiledMetalLibrary(error, device);

        // convernt the input string to NSString
        NSString *metalOpNameNSString = [NSString stringWithUTF8String:metalOpName.c_str()];
        id<MTLFunction> elementWiseOpFunction = [customKernelLibrary newFunctionWithName:metalOpNameNSString];
        TORCH_CHECK(elementWiseOpFunction, "Failed to create function state object for metal op");

        // Create a compute pipeline state object
        id<MTLComputePipelineState> elementWiseOpPSO = [device newComputePipelineStateWithFunction:elementWiseOpFunction error:&error];
        TORCH_CHECK(elementWiseOpPSO, error.localizedDescription.UTF8String);

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
            [computeEncoder setComputePipelineState:elementWiseOpPSO];

            // Set the tensor buffers
            [computeEncoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(width) offset:width.storage_offset() * width.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(height) offset:height.storage_offset() * height.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:3];

            // Set grid and thread group sizes
            //MTLSize gridSize = MTLSizeMake(widthB.item<int>(), heightA.item<int>(), 1);

            // hard set hthe gridSize and threadGroupSize
            // same dims as the output matrix
            int outputH = output.size(0);
            int outputW = output.size(1);
            MTLSize gridSize = MTLSizeMake(outputW, outputH, 1);

            // Query the maximum threads per thread group from the device
            NSUInteger maxThreadsPerThreadgroup = elementWiseOpPSO.maxTotalThreadsPerThreadgroup;

            // Choose a standard thread group size (like 8x8 or 16x16)
            // 32 x 32 (can be more in one dim if less in the other) seems like my max (apple M3 max binned)
            NSUInteger threadGroupWidth = 32;
            NSUInteger threadGroupHeight = 32;

            // // reduce by a factor of two until thread group fits under max threads per group
            while (threadGroupWidth * threadGroupHeight > maxThreadsPerThreadgroup) {
                threadGroupWidth /= 2;
                threadGroupHeight /= 2;
            }

            // Adjust thread group size to fit the gridSize if necessary
            if (outputW % threadGroupWidth != 0) {
                threadGroupWidth = outputW % threadGroupWidth;
            }
            if (outputH % threadGroupHeight != 0) {
                threadGroupHeight = outputH % threadGroupHeight;
            }

            MTLSize threadgroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);


            // Dispatch the compute command
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];

            // Commit the work
            torch::mps::synchronize();  // or commit? or synchronize?
        });
    }
    return output;
}