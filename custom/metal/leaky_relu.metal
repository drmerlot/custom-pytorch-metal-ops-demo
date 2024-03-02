#include <metal_stdlib>
using namespace metal;

kernel void leakyRelu(
    constant float* input [[buffer(0)]],
    constant uint* numElements [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid < *numElements) {
        float inputValue = input[gid];
        output[gid] = (inputValue > 0.0f) ? inputValue : (0.01f * inputValue);
    }
}
