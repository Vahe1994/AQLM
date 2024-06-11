#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

namespace F = torch::nn::functional;


inline bool check_use_bfloat16(const torch::Tensor& input) {
  auto dtype = input.dtype();
  if (dtype == at::kHalf) {
    return false;
  } else if (dtype == at::kBFloat16) {
    return true;
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support float16 and bfloat16. Got ",
        dtype.name(),
        ". Please specify the correct `torch_dtype` when loading the model."
      )
    );
  }
}


template <bool use_bfloat16, size_t group_size>
void  code1x16_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);
extern template void code1x16_matvec_cuda<false, 8>(const void*, const void*, void*, const void*, int, int);
extern template void code1x16_matvec_cuda<true, 8>(const void*, const void*, void*, const void*, int, int); 
extern template void code1x16_matvec_cuda<false, 16>(const void*, const void*, void*, const void*, int, int);
extern template void code1x16_matvec_cuda<true, 16>(const void*, const void*, void*, const void*, int, int);

template <size_t group_size>
void code1x16_dequant_cuda(
  const void* A,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);
extern template void code1x16_dequant_cuda<8>(const void*, void*, const void*, int, int);
extern template void code1x16_dequant_cuda<16>(const void*, void*, const void*, int, int);

template <bool use_bfloat16>
void code2x8_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);
extern template void code2x8_matvec_cuda<false>(const void*, const void*, void*, const void*, int, int);
extern template void code2x8_matvec_cuda<true>(const void*, const void*, void*, const void*, int, int);

void code2x8_dequant_cuda(
  const void* A,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
);

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

void code1x16_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  const bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  if (codebook.size(3) == 8) {
    if (use_bfloat16) {
      code1x16_matvec_cuda<true, 8>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    } else {
      code1x16_matvec_cuda<false, 8>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    }
  } else if (codebook.size(3) == 16) {
    if (use_bfloat16) {
      code1x16_matvec_cuda<true, 16>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    } else {
      code1x16_matvec_cuda<false, 16>(A.data_ptr(), B.data_ptr(), C.data_ptr(), codebook.data_ptr(), prob_m, prob_k);
    }
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebook.size(3),
        "."
      )
    );
  }
}

torch::Tensor code1x16_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code1x16_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      use_bfloat16
    );
  }
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code1x16_dequant(
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales
) {
  check_use_bfloat16(codebooks);
  auto in_features = codes.size(1) * codebooks.size(3);
  auto out_features = scales.size(0);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }
  weight *= scales.index({"...", 0, 0});

  return weight;
}

int4 accumulate_sizes(const torch::Tensor& codebook_partition_sizes)
{
  int4 cumulative_sizes;
  auto cumulative_size = &cumulative_sizes.x;
  int i = 0;
  int last = 0;
  assert(codebook_partition_sizes.size(0) <= 4);
  for (; i <  codebook_partition_sizes.size(0); ++i, ++cumulative_size)
  {
    *cumulative_size = codebook_partition_sizes[i].item<int>() + last;
    last = *cumulative_size;
  }
  // fill in the rest with unreachable.
  for (; i < 4; ++i, ++cumulative_size)
  {
    *cumulative_size = last*10;
  }
  return cumulative_sizes;
}

torch::Tensor code1x16_matmat_dequant(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * codebooks.size(3);
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }

  auto flat_output = F::linear(flat_input, weight);
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code1x16_matmat_dequant_transposed(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  check_use_bfloat16(codebooks);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);
  auto scaled_input = (input.reshape({-1, input.size(-1)}) * scales.flatten().unsqueeze(0)).reshape(input_sizes);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  if (codebooks.size(3) == 8) {
    code1x16_dequant_cuda<8>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else if (codebooks.size(3) == 16) {
    code1x16_dequant_cuda<16>(
      codes.data_ptr(),
      weight.data_ptr(),
      codebooks.data_ptr(),
      out_features,
      in_features
    );
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support codebooks with 8 or 16 features. Got ",
        codebooks.size(3),
        "."
      )
    );
  }

  torch::Tensor bias_2{};
  if (bias.has_value()) {
    bias_2 = bias.value();
  }

  return F::linear(scaled_input, weight.transpose(0, 1), bias_2);
}

void code2x8_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook,
  bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  if (use_bfloat16) {
    code2x8_matvec_cuda<true>(
      A.data_ptr(),
      B.data_ptr(),
      C.data_ptr(),
      codebook.data_ptr(),
      prob_m,
      prob_k
    );
  } else {
    code2x8_matvec_cuda<false>(
      A.data_ptr(),
      B.data_ptr(),
      C.data_ptr(),
      codebook.data_ptr(),
      prob_m,
      prob_k
    );
  }
}

torch::Tensor code2x8_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code2x8_matvec(
      codes.squeeze(2),
      input_vec,
      output_vec,
      codebooks,
      use_bfloat16
    );
  }
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code2x8_dequant(
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales
) {
  auto use_bfloat16 = check_use_bfloat16(codebooks);
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code2x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );
  weight *= scales.index({"...", 0, 0});

  return weight;
}

torch::Tensor code2x8_matmat_dequant(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code2x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );

  auto flat_output = F::linear(flat_input, weight);
  return scale_bias_unflatten_output(
    flat_output,
    scales,
    bias,
    input_sizes
  );
}

torch::Tensor code2x8_matmat_dequant_transposed(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  auto use_bfloat16 = check_use_bfloat16(codebooks);
  auto input_sizes = input.sizes();
  auto in_features = codes.size(1) * 8;
  auto out_features = scales.size(0);
  auto scaled_input = (input.reshape({-1, input.size(-1)}) * scales.flatten().unsqueeze(0)).reshape(input_sizes);

  auto weight = torch::empty({out_features, in_features},
    torch::TensorOptions()
      .dtype(codebooks.dtype())
      .device(codebooks.device())
  );
  code2x8_dequant_cuda(
    codes.data_ptr(),
    weight.data_ptr(),
    codebooks.data_ptr(),
    out_features,
    in_features,
    use_bfloat16
  );

  torch::Tensor bias_2{};
  if (bias.has_value()) {
    bias_2 = bias.value();
  }

  return F::linear(input, weight.transpose(0, 1), bias_2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("code1x16_matmat", &code1x16_matmat, "1x16 (2bit) codebook matrix-matrix product through matvec.");
  m.def("code1x16_dequant", &code1x16_dequant, "1x16 (2bit) codebook dequantization.");
  m.def("code1x16_matmat_dequant", &code1x16_matmat_dequant, "1x16 (2bit) codebook matrix-matrix dequantization product.");
  m.def("code1x16_matmat_dequant_transposed", &code1x16_matmat_dequant_transposed, "1x16 (2bit) codebook matrix-matrix dequantization product for backward pass.");
  m.def("code2x8_matmat", &code2x8_matmat, "2x8 (2bit) codebook matrix-matrix product.");
  m.def("code2x8_dequant", &code2x8_dequant, "2x8 (2bit) codebook dequantization.");
  m.def("code2x8_matmat_dequant", &code2x8_matmat_dequant, "2x8 (2bit) codebook matrix-matrix dequantization product.");
  m.def("code2x8_matmat_dequant_transposed", &code2x8_matmat_dequant_transposed, "2x8 (2bit) codebook matrix-matrix dequantization product for backward pass.");
}
