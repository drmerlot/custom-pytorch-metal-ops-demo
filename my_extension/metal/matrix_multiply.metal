#include <metal_stdlib>
using namespace metal;

kernel void matrixMultiply(constant float *matrixA [[buffer(0)]],
                           constant float *matrixB [[buffer(1)]],
                           device float *outputMatrix [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint widthA = 2; // Number of columns in matrix A (2x3)
    uint widthB = 2; // Number of columns in matrix B (3x2)
    uint heightA = 2; // Number of rows in matrix A (2x3)

    if (gid.x >= widthB || gid.y >= heightA) {
       return;
    }

    float sum = 0;
    for (uint i = 0; i < widthA; ++i) {
        sum += matrixA[gid.y * widthA + i] * matrixB[i * widthB + gid.x];
    }
    outputMatrix[gid.y * widthB + gid.x] = sum;
}

// // Define a simple kernel function to multiply 2 matricies
// kernel void matrixMultiply(
//     device float* A [[buffer(0)]],
//     device float* B [[buffer(1)]],
//     // constant int& widthA [[buffer(2)]],
//     // constant int& heightA [[buffer(3)]],
//     // constant int& widthB [[buffer(4)]],
//     // device float* result [[buffer(5)]],
//     uint2 gid [[thread_position_in_grid]])
// {
//     result[gid]
// }

// // testing
// kernel void matrixMultiply(
//     device float* A [[buffer(0)]],  // Unused in this kernel, but kept for debugging
//     device float* B [[buffer(1)]],  // Unused in this kernel, but kept for debugging
//     device int* widthABuffer [[buffer(2)]],  // Now a pointer
//     device int* heightABuffer [[buffer(3)]],  // Now a pointer
//     device int* widthBBuffer [[buffer(4)]],  // Now a pointer
//     device float* result [[buffer(5)]],
//     uint2 gid [[thread_position_in_grid]])
// {
//     // get ints from buffers
//     int widthA = widthABuffer[0];
//     int heightA = heightABuffer[0];
//     int widthB = widthBBuffer[0];

//     if (gid.x >= widthB || gid.y >= heightA) return;

//     // Set each element of the result matrix to 1.0
//     result[gid.y * widthB + gid.x] = 1.0;
// }