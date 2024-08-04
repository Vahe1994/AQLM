#include "lut_kernel.h"

#include <numeric>
#include <functional>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <executorch/kernels/optimized/blas/CPUBlas.h>

constexpr int GROUP_SIZE = 1024;


template<typename fp_dtype, int num_codebooks, int codebook_size>
void quadruple_for(
    int num_inputs,
    int num_input_groups, const void* lut_void,
    int out_features, const void* b_alt_void,
    void* output_vec_void
)
{
    const fp_dtype* lut = (fp_dtype*)lut_void;

    fp_dtype* output_vec = (fp_dtype*)output_vec_void;
    const uint8_t* b_alt = (uint8_t*)b_alt_void;

    for (int input = 0; input < num_inputs; ++input) {
        for (int i = 0; i < out_features; ++i) {
            output_vec[input * out_features + i] = 0;
        }
    }

    for (int input = 0; input < num_inputs; ++input) {
        for (int j = 0; j < num_input_groups; ++j) {
            for (int i = 0; i < out_features; ++i) {
                for (int c = 0; c < num_codebooks; ++c) {
                    output_vec[input * out_features + i] += lut[
                        input * num_input_groups * num_codebooks * codebook_size +
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
}

namespace torch {
  namespace executor {
    namespace native {
      Tensor& code2x8_lut_matmat_out(
        RuntimeContext& ctx,
        const Tensor& input,
        const Tensor& codes,
        const Tensor& codebooks,
        const Tensor& scales,
        const optional<Tensor>& bias,
        Tensor& out
      ) {
        auto input_sizes = input.sizes();
        auto out_features = codes.size(1) * codebooks.size(2);
        auto input_vector_size = input.size(input.dim() - 1);
        auto num_input_vectors = std::accumulate(input_sizes.begin(), input_sizes.end(), 1, std::multiplies<int64_t>()) / input_vector_size;

        // Allocate LUT
        auto lut_data = ctx.allocate_temp(
            4 * num_input_vectors * input_vector_size / 8 * codebooks.size(0) * codebooks.size(1)
        ).get();

        // A @ B.T
        ::executorch::cpublas::gemm(
            ::executorch::cpublas::TransposeType::Transpose,
            ::executorch::cpublas::TransposeType::NoTranspose,
            (int64_t)codebooks.size(0) * codebooks.size(1),      // B rows
            (int64_t)num_input_vectors * input_vector_size / 8,  // A rows
            (int64_t)8,                                          // MatMul dim size
            1.f,
            (float*)codebooks.const_data_ptr(), (int64_t)8,
            (float*)input.const_data_ptr(), (int64_t)8,
            0.f,
            (float*)lut_data, (int64_t)codebooks.size(0) * codebooks.size(1)
        );

        // Do lookup matmul
        quadruple_for<float, 2, 256>(
            num_input_vectors,
            input_vector_size / 8,
            lut_data,
            out_features,
            codes.const_data_ptr(),
            out.mutable_data_ptr()
        );
        
        // Scale and bias
        for (int j = 0; j < out_features; ++j) {
            for (int i=0; i < num_input_vectors; ++i) {
                out.mutable_data_ptr<float>()[
                    i * out_features + j
                ] *= scales.const_data_ptr<float>()[j];
                if (bias.has_value()) {
                    out.mutable_data_ptr<float>()[
                        i * out_features + j
                    ] += bias.value().const_data_ptr<float>()[j];
                }
            }
        }
        
        return out;
      }
    } // namespace native
  } // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(aqlm, "code2x8_lut_matmat.out", torch::executor::native::code2x8_lut_matmat_out);
