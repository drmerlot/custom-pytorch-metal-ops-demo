#include <metal_stdlib>
using namespace metal;


constant int tileSize = 16; // Define the tile size, should be tuned based on GPU capabilities


kernel void matrixMultiply(
    constant float* A [[buffer(0)]],
    constant float* B [[buffer(1)]],
    constant uint* widthA [[buffer(2)]],
    constant uint* heightA [[buffer(3)]],
    constant uint* widthB [[buffer(4)]],
    device float* result [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {

    // Shared memory tiles
    threadgroup float Asub[tileSize][tileSize];
    threadgroup float Bsub[tileSize][tileSize];

    // Get dimensions from buffers
    uint colsA = *widthA;
    uint rowsA = *heightA;
    uint colsB = *widthB;

    // Check if gid is within the bounds of the result matrix
    if (gid.x >= colsB || gid.y >= rowsA) {
        return;
    }

    uint row = gid.y;
    uint col = gid.x;

    float Csub = 0.0;
    // Loop over the tiles
    for (uint t = 0; t < (colsA + tileSize - 1) / tileSize; ++t) {
        // Load one tile of A and B into shared memory
        if (row < rowsA && t * tileSize + tid.x < colsA) {
            Asub[tid.y][tid.x] = A[row * colsA + t * tileSize + tid.x];
        } else {
            Asub[tid.y][tid.x] = 0.0;
        }

        if (col < colsB && t * tileSize + tid.y < colsA) {
            Bsub[tid.y][tid.x] = B[(t * tileSize + tid.y) * colsB + col];
        } else {
            Bsub[tid.y][tid.x] = 0.0;
        }

        // Wait for all threads to finish loading their part of the tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Perform the multiplication for this tile
        for (uint k = 0; k < tileSize; ++k) {
            Csub += Asub[tid.y][k] * Bsub[k][tid.x];
        }

        // Wait for all threads to finish computation before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write the computed value to the output matrix
    if (row < rowsA && col < colsB) {
        result[row * colsB + col] = Csub;
    }
}
