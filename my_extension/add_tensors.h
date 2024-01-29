/*
Metal Shader code for the custom add tensor operation.
Code based on example in:
https://developer.apple.com/documentation/metal/metal_sample_code_library/customizing_a_pytorch_operation?language=objc
*/

#pragma once

// Defines the Metal soft shrink custom kernel code as char
//static char *CUSTOM_KERNEL = R"ADD_TENSORS(
char const *CUSTOM_KERNEL = R"(
#include <metal_stdlib>
using namespace metal;

// Define a simple kernel function to add two tensors
kernel void addTensors(device float *a [[buffer(0)]],
                       device float *b [[buffer(1)]],
                       device float *result [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    // Perform addition if within tensor bounds
    result[id] = a[id] + b[id];
})";