#include <metal_stdlib>
using namespace metal;

// Same buffer layout as relu.metal, so both run through dispatchElementWiseMatrixOp
kernel void leakyRelu(
    device float* input [[buffer(0)]],       // Buffer for input matrix
    constant uint* width [[buffer(1)]],      // Buffer for width of the matrix
    constant uint* height [[buffer(2)]],     // Buffer for height of the matrix
    device float* output [[buffer(3)]],      // Buffer for output matrix
    uint2 gid [[thread_position_in_grid]]) { // 2D Global thread ID

    uint cols = *width;
    uint rows = *height;
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    uint index = gid.y * cols + gid.x;
    float inputValue = input[index];
    output[index] = (inputValue > 0.0f) ? inputValue : (0.01f * inputValue);  // negative slope 0.01, as torch's default
}
