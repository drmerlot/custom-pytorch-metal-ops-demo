torch::Tensor& dispatchMatrixAdd(const torch::Tensor& A,
                                 const torch::Tensor& B,
                                 torch::Tensor& result,
                                 const int width,
                                 const int height) {
    @autoreleasepool {
        // Create and configure the Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Load the Metal shader file
        const char* shaderSource = readMetalShader("./path/to/matrix_add.metal");

        // Compile the shader source into a library
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shaderSource]
                                                      options:nil
                                                        error:&error];
        delete[] shaderSource;
        NSAssert(library, @"Failed to compile shader: %@", error);

        // Retrieve the kernel function from the library
        id<MTLFunction> function = [library newFunctionWithName:@"matrixAdd"];
        NSAssert(function, @"Failed to find kernel function in shader library");

        // Create a compute pipeline state object
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        NSAssert(pipelineState, @"Failed to create compute pipeline state: %@", error);

        // Create command buffer and encoder
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set the buffers for the kernel
        [encoder setBuffer:getMTLBufferStorage(A) offset:0 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(B) offset:0 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(result) offset:0 atIndex:2];
        [encoder setBytes:&width length:sizeof(int) atIndex:3];
        [encoder setBytes:&height length:sizeof(int) atIndex:4];

        // Configure grid and threadgroup sizes
        MTLSize gridSize = MTLSizeMake(width, height, 1);
        NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        // Encode the compute command
        [encoder setComputePipelineState:pipelineState];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Commit the command buffer
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted]; // Optional, for synchronous execution

        return result;
    }
}
