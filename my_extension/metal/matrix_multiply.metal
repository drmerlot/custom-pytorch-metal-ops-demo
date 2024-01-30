#include <metal_stdlib>
using namespace metal;

// Define a simple kernel function to multiply 2 matricies
kernel void matrixMultiply(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant int& widthA [[buffer(3)]],
    constant int& heightA [[buffer(4)]],
    constant int& widthB [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= widthB || gid.y >= heightA) return;

    float sum = 0.0;
    for (int i = 0; i < widthA; ++i) {
        sum += A[gid.y * widthA + i] * B[i * widthB + gid.x];
    }
    result[gid.y * widthB + gid.x] = sum;
}
