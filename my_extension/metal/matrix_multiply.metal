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