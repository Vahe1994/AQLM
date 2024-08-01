#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

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
);
} // namespace native
} // namespace executor
} // namespace torch
