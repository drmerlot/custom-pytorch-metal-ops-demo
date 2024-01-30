#include <metal_stdlib>
using namespace metal;

// Define a simple kernel function to multiply 2 matricies
// kernel void matrixMultiply(
//     device float* A [[buffer(0)]],
//     device float* B [[buffer(1)]],
//     constant int& widthA [[buffer(2)]],
//     constant int& heightA [[buffer(3)]],
//     constant int& widthB [[buffer(4)]],
//     device float* result [[buffer(5)]],
//     uint2 gid [[thread_position_in_grid]])
// {
//     if (gid.x >= widthB || gid.y >= heightA) return;

//     float sum = 0.0;
//     for (int i = 0; i < widthA; ++i) {
//         sum += A[gid.y * widthA + i] * B[i * widthB + gid.x];
//     }
//     result[gid.y * widthB + gid.x] = sum;
// }

// Define a kernel function to create a matrix of all 1s
kernel void matrixMultiply(
    device float* A [[buffer(0)]],  // Unused in this kernel, but kept for debugging
    device float* B [[buffer(1)]],  // Unused in this kernel, but kept for debugging
    constant int& widthA [[buffer(2)]],  // Unused in this kernel, but kept for debugging
    constant int& heightA [[buffer(3)]],  // Unused in this kernel, but kept for debugging
    constant int& widthB [[buffer(4)]],  // Unused in this kernel, but kept for debugging
    device float* result [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= widthB || gid.y >= heightA) return;

    // Set each element of the result matrix to 1.0
    result[gid.y * widthB + gid.x] = 1.0;
}