#include <cstdio>
#include <assert.h>
#include <omp.h>


void sum(
    int size_A, float* A,
    int size_B, float* B,
    int size_result, float* result,
    int *n_threads
);

void triple_for(
    int num_input_groups, int num_codebooks, int codebook_size, float* lut,
    int num_input_groups_again, int num_codebooks_again, int out_features, uint8_t* b_alt,
    int output_vec_size, float* output_vec,
    int *n_threads
);

void triple_for_specific(
    int num_input_groups, int num_codebooks, int codebook_size, float* lut,
    int num_input_groups_again, int num_codebooks_again, int out_features, uint8_t* b_alt,
    int output_vec_size, float* output_vec,
    int *n_threads
);