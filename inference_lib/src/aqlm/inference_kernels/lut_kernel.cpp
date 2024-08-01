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
        for (int j = 0; j < num_input_groups; ++j) {
            for (int i_group = 0; i_group < out_features / GROUP_SIZE; ++i_group) {
                for (int i_local = 0; i_local < GROUP_SIZE; ++i_local) {
                    auto i = i_group * GROUP_SIZE + i_local;
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
}

namespace torch {
  namespace executor {
    namespace native {
      // inline Tensor scale_bias_unflatten_output(
      //   Tensor& out,
      //   const Tensor& scales,
      //   const optional<Tensor>& bias
      // ) {
      //   out *= view_copy_out(ctx, scales, {-1}).unsqueeze(0);
      //   if (bias.has_value()) {
      //     out += bias.unsqueeze(0);
      //   }
      //   return out;
      // }


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

        std::array<exec_aten::DimOrderType, 4> lut_dim_order{
            0, 1, 2, 3};
        std::array<exec_aten::SizesType, 4> lut_sizes;
        lut_sizes[0] = num_input_vectors;      // NUM INPUTS
        lut_sizes[1] = input_vector_size / 8;  // NUM INPUT GROUPS
        lut_sizes[2] = codebooks.size(0);       // NUM CODEBOOKS
        lut_sizes[3] = codebooks.size(1);       // CODEBOOK SIZE
        std::array<exec_aten::StridesType, 4> lut_strides;
        dim_order_to_stride_nocheck(
            lut_sizes.data(),
            lut_dim_order.data(),
            4,
            lut_strides.data());
        TensorImpl k_impl = TensorImpl(
            input.scalar_type(),
            4,
            lut_sizes.data(),
            lut_data,
            lut_dim_order.data(),
            lut_strides.data(),
            TensorShapeDynamism::STATIC);
        Tensor lut(&k_impl);

        // A @ B.T
        ::executorch::cpublas::gemm(
            ::executorch::cpublas::TransposeType::NoTranspose,
            ::executorch::cpublas::TransposeType::Transpose,
            (int64_t)num_input_vectors * input_vector_size / 8,   // A rows
            (int64_t)codebooks.size(0) * codebooks.size(1),       // B rows
            (int64_t)8,                                           // MatMul dim
            1.f,
            (float*)input.const_data_ptr(), (int64_t)8,
            (float*)codebooks.const_data_ptr(), (int64_t)8,
            0.f,
            (float*)lut.mutable_data_ptr(), (int64_t)codebooks.size(0) * codebooks.size(1)
        );


        quadruple_for<float, 2, 256>(
            num_input_vectors,
            input_vector_size / 8,
            lut.const_data_ptr(),
            out_features,
            codes.const_data_ptr(),
            out.mutable_data_ptr()
        );
        
        // out = torch::executor::native::scale_bias_unflatten_output(
        //   out,
        //   scales,
        //   bias
        // );
        return out;
      }
    } // namespace native
  } // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(aqlm, "code2x8_lut_matmat.out", torch::executor::native::code2x8_lut_matmat_out);
