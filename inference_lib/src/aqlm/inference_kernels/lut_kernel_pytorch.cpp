#include "lut_kernel.h"

#include <torch/library.h>
#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>

at::Tensor code2x8_lut_matmat(
  const at::Tensor& input,
  const at::Tensor& codes,
  const at::Tensor& codebooks,
  const at::Tensor& scales,
  const std::optional<at::Tensor>& bias
) {
    at::Tensor out = at::empty({input.reshape({-1, input.size(-1)}).size(0), codes.size(1) * codebooks.size(2)},
        torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device())
    );

    WRAP_TO_ATEN(code2x8_lut_matmat_out, 5)(
        input,
        codes,
        codebooks,
        scales,
        bias,
        out
    );
    return out;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("code2x8_lut_matmat", &code2x8_lut_matmat, "2x8 (2bit) codebook matrix-matrix product.");
// }

TORCH_LIBRARY(aqlm, m) {
  m.def(
      "code2x8_lut_matmat(Tensor input, Tensor codes, "
      "Tensor codebooks, Tensor scales, Tensor? bias=None) -> Tensor"
  );
  m.def(
      "code2x8_lut_matmat.out(Tensor input, Tensor codes, "
      "Tensor codebooks, Tensor scales, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)"
  );
}

TORCH_LIBRARY_IMPL(aqlm, CompositeExplicitAutograd, m) {
  m.impl(
      "code2x8_lut_matmat", code2x8_lut_matmat
  );
  m.impl(
      "code2x8_lut_matmat.out",
      WRAP_TO_ATEN(code2x8_lut_matmat_out, 5)
    );
}