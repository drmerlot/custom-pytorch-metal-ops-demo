#include <metal_stdlib>
using namespace metal;

// Define a simple kernel function to add two tensors
kernel void matrixAdd(
    constant float* A [[buffer(0)]],
    constant float* B [[buffer(1)]],      // Buffer for input matrix
    constant uint* widthA [[buffer(2)]],      // Buffer for width of the matrix
    constant uint* heightA [[buffer(3)]],     // Buffer for height of the matrix
    device float* output [[buffer(4)]],      // Buffer for output matrix
    uint2 gid [[thread_position_in_grid]]
) {
    // Get matrix dimensions
    uint cols = *widthA;  // Number of columns in the matrix
    uint rows = *heightA; // Number of rows in the matrix

    // Check if gid is within the bounds of the matrix
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }

    // Calculate linear index
    uint index = gid.y * cols + gid.x;

    // add the matricies
    output[index] = A[index] + B[gid.x];
}