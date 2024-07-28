#include <torch/all.h>
#include <torch/python.h>
#include <c10/util/Exception.h>

// #include <bits/stdc++.h>
// #include <tuple>
// #include "lib.h"


inline torch::Tensor scale_bias_unflatten_output(
        torch::Tensor& flat_output,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias,
  const c10::IntArrayRef& input_sizes
) {
  flat_output *= scales.flatten().unsqueeze(0);
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(flat_output.size(-1));
  auto output = flat_output.reshape(output_sizes).clone();
  return output;
}

constexpr int GROUP_SIZE = 1024;


template<typename fp_dtype, int num_codebooks, int codebook_size>
void quadruple_for(
    int num_inputs,
    int num_input_groups, void* lut_void,
    int out_features, void* b_alt_void,
    void* output_vec_void
)
{
    fp_dtype* lut = (fp_dtype*)lut_void;

    fp_dtype* output_vec = (fp_dtype*)output_vec_void;
    uint8_t* b_alt = (uint8_t*)b_alt_void;

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

torch::Tensor code2x8_lut_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  auto input_sizes = input.sizes();
  auto out_features = codes.size(1) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::zeros({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  auto lut = torch::matmul(
    flat_input.reshape({flat_input.size(0), -1, 8}),
    codebooks.reshape({-1, 8}).transpose(0, 1).unsqueeze(0)
  );

    quadruple_for<float, 2, 256>(
        flat_input.size(0),
        input.size(-1) / 8,
        lut.data_ptr(),
        out_features,
        codes.data_ptr(),
        flat_output.data_ptr()
    );
  
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("code2x8_lut_matmat", &code2x8_lut_matmat, "2x8 (2bit) codebook matrix-matrix product.");
}
