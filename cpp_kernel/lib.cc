#include <bits/stdc++.h>
#include <tuple>
#include "lib.h"


void sum(
    int size_A, float* A,
    int size_B, float* B,
    int size_result, float* result,
    int *n_threads
)
{
    #pragma omp parallel for num_threads(*n_threads)
    for(int i=0; i < size_A; i++){
        result[i] = A[i] + B[i];
    }
}


void triple_for(
    int num_input_groups, int num_codebooks, int codebook_size, float* lut,
    int num_input_groups_again, int num_codebooks_again, int out_features, uint8_t* b_alt,
    int output_vec_size, float* output_vec,
    int *n_threads
)
{
    if (8192 == out_features && 4 == num_codebooks && codebook_size == 256) {
        triple_for_specific(
            num_input_groups, num_codebooks, codebook_size, lut,
            num_input_groups_again, num_codebooks_again, out_features, b_alt,
            output_vec_size, output_vec,
            n_threads
        );
        return;
    }
    throw std::overflow_error("AAA");
    
    for (int j = 0; j < num_input_groups; ++j) {
        for (int c = 0; c < num_codebooks; ++c) {
            for (int i = 0; i < out_features; ++i) {
                output_vec[i] += lut[
                    j * num_codebooks * codebook_size + 
                    c * codebook_size + 
                    b_alt[
                        j * num_codebooks * out_features + 
                        c * out_features + 
                        i
                    ]
                ];
            }
        }
    }
}

void triple_for_specific(
    int num_input_groups, int num_codebooks, int codebook_size, float* lut,
    int num_input_groups_again, int num_codebooks_again, int out_features, uint8_t* b_alt,
    int output_vec_size, float* output_vec,
    int *n_threads
)
{
    for (int j = 0; j < num_input_groups; ++j) {
        for (int c = 0; c < 4; ++c) {
            auto offset_lut = j * 4 * 256 + 
                    c * 256;
            auto offset_b = j * 4 * 8192 + 
                        c * 8192;
            #pragma GCC unroll 1024
            #pragma GCC ivdep
            for (int i = 0; i < 8192; ++i) {
                output_vec[i] += lut[
                    offset_lut + 
                    b_alt[
                        offset_b + 
                        i
                    ]
                ];
            }
        }
    }
}