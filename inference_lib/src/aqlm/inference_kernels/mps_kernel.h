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
  constant uint16_t* A [[buffer(0)]],
  constant T*         B [[buffer(1)]],
  device   T*         C [[buffer(2)]],
  constant T*         codebook [[buffer(3)]],
  constant int&       prob_m   [[buffer(4)]],
  constant int&       prob_k   [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
    uint num_codes = prob_k / 8;
    constant uint16_t* codes_row = A + index * num_codes;

    float res = 0;
    for (uint i = 0; i < num_codes; ++i) {
        constant T* encoded_vector = codebook + static_cast<size_t>(codes_row[i]) * 8;
        for (uint j = 0; j < 8; ++j) {
            res += static_cast<float>(encoded_vector[j] * B[i * 8 + j]);
        }
    }
    C[index] = res;
}

template
[[host_name("aqlm_gemv_1x16_kernel_half")]]
kernel void Code1x16MatVec<half>(
  constant uint16_t* A [[buffer(0)]],
  constant half*         B [[buffer(1)]],
  device   half*         C [[buffer(2)]],
  constant half*         codebook [[buffer(3)]],
  constant int&       prob_m   [[buffer(4)]],
  constant int&       prob_k   [[buffer(5)]],
  uint index [[thread_position_in_grid]]
);

template
[[host_name("aqlm_gemv_1x16_kernel_float")]]
kernel void Code1x16MatVec<float>(
  constant uint16_t* A [[buffer(0)]],
  constant float*         B [[buffer(1)]],
  device   float*         C [[buffer(2)]],
  constant float*         codebook [[buffer(3)]],
  constant int&       prob_m   [[buffer(4)]],
  constant int&       prob_k   [[buffer(5)]],
  uint index [[thread_position_in_grid]]
);
)MPS_AQLM";
