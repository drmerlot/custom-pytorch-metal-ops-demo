#include <metal_stdlib>
using namespace metal;

kernel void relu(
    constant float* input [[buffer(0)]],       // Buffer for input data
    constant uint* numElements [[buffer(1)]],  // Buffer for number of elements
    device float* output [[buffer(2)]],        // Buffer for output data
    uint gid [[thread_position_in_grid]]) {    // Global thread ID

    if (gid < *numElements) {
        float inputValue = input[gid];
        output[gid] = (inputValue > 0.0f) ? inputValue : (0.01f * inputValue); // Correctly multiply scalar values
    }
}
