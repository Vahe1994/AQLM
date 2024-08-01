#include "lut_kernel.h"

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

#include <torch/library.h>

namespace torch {
    namespace executor {
        namespace native {
            Tensor& code2x8_lut_matmat_out_no_context(
                const Tensor& input,
                const Tensor& codes,
                const Tensor& codebooks,
                const Tensor& scales,
                const optional<Tensor> bias,
                Tensor& output
            ) {
                void* memory_pool = malloc(10000000 * sizeof(uint8_t));
                MemoryAllocator allocator(10000000, (uint8_t*)memory_pool);

                exec_aten::RuntimeContext context{nullptr, &allocator};
                return torch::executor::native::code2x8_lut_matmat_out(
                    context,
                    input,
                    codes,
                    codebooks,
                    scales,
                    bias,
                    output
                );
            }

            at::Tensor code2x8_lut_matmat(
                const at::Tensor& input,
                const at::Tensor& codes,
                const at::Tensor& codebooks,
                const at::Tensor& scales,
                const c10::optional<at::Tensor> bias
            ) {
                auto sizes = input.sizes().vec();
                sizes[sizes.size() - 1] = codes.size(1) * codebooks.size(2);
                auto out = at::empty(sizes,
                    at::TensorOptions()
                    .dtype(input.dtype())
                    .device(input.device())
                );

                WRAP_TO_ATEN(code2x8_lut_matmat_out_no_context, 5)(
                    input,
                    codes,
                    codebooks,
                    scales,
                    bias,
                    out
                );
                return out;
            }
        } // namespace native
    } // namespace executor
} // namespace torch

TORCH_LIBRARY(aqlm, m) {
  m.def(
      "code2x8_lut_matmat(Tensor input, Tensor codes, "
      "Tensor codebooks, Tensor scales, Tensor? bias=None) -> Tensor"
  );
  m.def(
      "code2x8_lut_matmat.out(Tensor input, Tensor codes, "
      "Tensor codebooks, Tensor scales, Tensor? bias=None, *, Tensor(c!) out) -> Tensor(c!)"
  );
}

TORCH_LIBRARY_IMPL(aqlm, CompositeExplicitAutograd, m) {
  m.impl(
      "code2x8_lut_matmat", torch::executor::native::code2x8_lut_matmat
  );
  m.impl(
      "code2x8_lut_matmat.out",
      WRAP_TO_ATEN(torch::executor::native::code2x8_lut_matmat_out_no_context, 5)
    );
}