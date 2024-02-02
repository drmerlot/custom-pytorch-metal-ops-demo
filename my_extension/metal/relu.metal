#include <metal_stdlib>
using namespace metal;

kernel void relu(
    device float* input [[buffer(0)]],       // Buffer for input matrix
    constant uint* width [[buffer(1)]],      // Buffer for width of the matrix
    constant uint* height [[buffer(2)]],     // Buffer for height of the matrix
    device float* output [[buffer(3)]],      // Buffer for output matrix
    uint2 gid [[thread_position_in_grid]]) { // 2D Global thread ID

    // Get matrix dimensions
    uint cols = *width;  // Number of columns in the matrix
    uint rows = *height; // Number of rows in the matrix

    // Check if gid is within the bounds of the matrix
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }

    // Calculate linear index
    uint index = gid.y * cols + gid.x;

    // Apply ReLU function element-wise
    float inputValue = input[index];
    output[index] = max(0.0f, inputValue); // Standard ReLU operation
}
