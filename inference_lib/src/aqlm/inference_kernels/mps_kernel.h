/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The shader code for the custom operation.
*/

#pragma once

// Defines the Metal soft shrink custom kernel.
static char *CUSTOM_KERNEL = R"MPS_AQLM(
#include <metal_stdlib>
using namespace metal;

template<typename T>
kernel void Code1x16MatVec(
  constant short int* A [[buffer(0)]],
  constant T*         B [[buffer(1)]],
  device   T*         C [[buffer(2)]],
  constant T*         codebook [[buffer(3)]],
  constant int&       prob_m   [[buffer(4)]],
  constant int&       prob_k   [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
    return;
}

template
[[host_name("aqlm_gemv_1x16_kernel_half")]]
kernel void Code1x16MatVec<half>(
  constant short int* A [[buffer(0)]],
  constant half*         B [[buffer(1)]],
  device   half*         C [[buffer(2)]],
  constant half*         codebook [[buffer(3)]],
  constant int&       prob_m   [[buffer(4)]],
  constant int&       prob_k   [[buffer(5)]],
  uint index [[thread_position_in_grid]]
);
)MPS_AQLM";
