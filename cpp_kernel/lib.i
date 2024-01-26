%module bindings

%{
    #define SWIG_FILE_WITH_INIT
    #include "lib.h"
%}

%include "numpy.i"
%include "typemaps.i"

%init %{
    import_array();
%}

%apply (int DIM1, float* IN_ARRAY1) {
    (int size_A, float* A),
    (int size_B, float* B),
    (int size_result, float* result),
    (int output_vec_size, float* output_vec)
}


%apply (int DIM1, int DIM2, int DIM3, float* IN_ARRAY3) {
    (int num_input_groups, int num_codebooks, int codebook_size, float* lut)
}

%apply (int DIM1, int DIM2, int DIM3, uint8_t* IN_ARRAY3) {
    (int num_input_groups_again, int num_codebooks_again, int out_features, uint8_t* b_alt)
}

%apply int *INPUT {int *n_threads}

%include "lib.h"
