#include <bits/stdc++.h>
#include <tuple>
#include "lib.h"


void triple_for(
    int num_input_groups, int num_codebooks, int codebook_size, float* lut,
    int num_input_groups_again, int out_features, int num_codebooks_again, uint8_t* b_alt,
    int output_vec_size, float* output_vec,
    int *n_threads
)
{
    if (8192 == out_features && 4 == num_codebooks && codebook_size == 256 && num_input_groups == 512) {
        triple_for_specific(
            num_input_groups, num_codebooks, codebook_size, lut,
            num_input_groups_again, out_features, num_codebooks_again, b_alt,
            output_vec_size, output_vec,
            n_threads
        );
        return;
    }
    
    for (int j = 0; j < num_input_groups; ++j) {
        for (int i = 0; i < out_features; ++i) {
            for (int c = 0; c < num_codebooks; ++c) {
                output_vec[i] += lut[
                    j * num_codebooks * codebook_size + 
                    c * codebook_size + 
                    b_alt[
                        j * num_codebooks * out_features + 
                        i * num_codebooks + 
                        c
                    ]
                ];
            }
        }
    }
}

void triple_for_specific(
    int num_input_groups, int num_codebooks, int codebook_size, float* lut,
    int num_input_groups_again, int out_features, int num_codebooks_again, uint8_t* b_alt,
    int output_vec_size, float* output_vec,
    int *n_threads
)
{
    for (int j = 0; j < 512; ++j) {
        for (int i = 0; i < 8192; ++i) {
            for (int c = 0; c < 4; ++c) {
                output_vec[i] += lut[
                    j * 4 * 256 + 
                    c * 256 + 
                    b_alt[
                        j * 4 * 8192 + 
                        i * 4 + 
                        c
                    ]
                ];
            }
        }
    }
}