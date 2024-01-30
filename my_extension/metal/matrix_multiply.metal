#include <metal_stdlib>
using namespace metal;


kernel void matrixMultiply(
    constant float* A [[buffer(0)]], // Buffer for matrix A
    constant float* B [[buffer(1)]], // Buffer for matrix B
    constant uint* widthA [[buffer(2)]], // Buffer for width of A
    constant uint* heightA [[buffer(3)]], // Buffer for height of A
    constant uint* widthB [[buffer(4)]], // Buffer for width of B
    device float* result [[buffer(5)]], // Buffer for result matrix
    uint2 gid [[thread_position_in_grid]]) {

    // Get dimensions from buffers
    uint colsA = *widthA; // Number of columns in A
    uint rowsA = *heightA; // Number of rows in A
    uint colsB = *widthB; // Number of columns in B

    // Check if gid is within the bounds of the result matrix
    if (gid.x >= colsB || gid.y >= rowsA) {
        return;
    }

    // Perform matrix multiplication for the element at gid
    float sum = 0.0;
    for (uint i = 0; i < colsA; ++i) {
        sum += A[gid.y * colsA + i] * B[i * colsB + gid.x];
    }

    // Write the result
    result[gid.y * colsB + gid.x] = sum;
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