#include <torch/all.h>
#include <torch/python.h>

void code1x16_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);

void code2x8_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* codebook,
  int prob_m,
  int prob_k
);

void code1x16_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook
) {
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  code1x16_matvec_cuda(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    codebook.data_ptr(),
    prob_m,
    prob_k
  );
}

void code2x8_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& codebook
) {
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  code2x8_matvec_cuda(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    codebook.data_ptr(),
    prob_m,
    prob_k
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("code1x16_matvec", &code1x16_matvec, "1x16 (2bit) codebook matrix-vector product.");
  m.def("code2x8_matvec", &code2x8_matvec, "2x16 (2bit) codebook matrix-vector product.");
}
