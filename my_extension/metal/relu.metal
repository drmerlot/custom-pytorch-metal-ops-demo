#include <metal_stdlib>
using namespace metal;

// The kernel function for ReLU activation
kernel void relu(
    constant float* input [[buffer(0)]],
    constant uint* numElements [[buffer(1)]], // Buffer for input data
    device float* output [[buffer(3)]],      // Buffer for output data
    uint gid [[thread_position_in_grid]]) {  // Global thread ID

    // get any of the single value of of tensor buffers
    uint outputElements = *numElements;

    // Ensure we do not access out of bounds
    if (gid < outputElements) {
        output[gid] = max(input[gid], 0.0f); // ReLU operation
    }
}
