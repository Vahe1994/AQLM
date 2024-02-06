import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import time

import numba
import numpy as np
import torch
import torch.nn.functional as F
from aqlm.utils import _dequantize_weight, pack_int_data, unpack_int_data
from torch import nn


def benchmark(f, warmup=10, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            tick = time.perf_counter()
    average_latency = (time.perf_counter() - tick) / iter
    time.sleep(1.0)
    return average_latency


MODELS = {
    "Llama 2 7B": [
        (4096, 11008),  #  gate_proj shape
    ],
    "Llama 2 13B": [
        (5120, 13824),  #  gate_proj shape
    ],
    "Llama 2 70B": [
        (8192, 28672),  #  gate_proj shape
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=1000,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--log_error",
        action="store_true",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=8,
        help="Number of bits per codebook.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=2,
        help="Number of num_codebooks.",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="Input group size.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="Num threads.",
    )

    args = parser.parse_args()

    numba.set_num_threads(args.nthreads)
    torch.set_num_threads(args.nthreads)

    for model, layers in MODELS.items():
        dense = 0
        quant = 0
        for in_features, out_features in layers:
            in_group_size, num_codebooks, nbits_per_codebook, num_input_groups = (
                args.in_group_size,
                args.num_codebooks,
                args.nbits_per_codebook,
                in_features // args.in_group_size,
            )

            @numba.njit(parallel=True)
            def aqlm_gemv_lut(x, codebooks, codes_alt, scales):
                lut = x.reshape(-1, in_group_size) @ codebooks.reshape(-1, in_group_size).T
                lut = lut.reshape(-1, num_codebooks, 2**nbits_per_codebook)

                output_vec = np.zeros(out_features, dtype=x.dtype)
                for j in numba.prange(num_input_groups):
                    for i in range(out_features):
                        for c in range(num_codebooks):
                            output_vec[i] += lut[j, c, codes_alt[j, i, c]]
                output_vec *= scales.flatten()
                return output_vec

            input = torch.randn((1, in_features), dtype=torch.float32)  #  [..., in_features]
            codes = pack_int_data(
                torch.randint(
                    2**args.nbits_per_codebook, (in_features // args.in_group_size, out_features, args.num_codebooks)
                ),  #  [num_in_groups, num_out_groups, num_codebooks]
                args.nbits_per_codebook,
            )
            codebooks = torch.randn(
                (args.num_codebooks, 2**args.nbits_per_codebook, 1, args.in_group_size), dtype=torch.float32
            )  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
            scales = torch.randn((out_features, 1, 1, 1), dtype=torch.float32)  #  [num_out_groups, 1, 1, 1]

            weight = _dequantize_weight(
                unpack_int_data(torch.permute(codes, (1, 0, 2)), args.nbits_per_codebook), codebooks, scales
            ).contiguous()

            output_ref = F.linear(input, weight)
            output = aqlm_gemv_lut(input.numpy(), codebooks.numpy(), codes.numpy(), scales.numpy())
            if args.log_error:
                print(
                    f"Relative error: {(torch.mean(torch.abs(output_ref - output)) / torch.mean(torch.abs(output_ref))).item():.2e}"
                )

            dense += benchmark(lambda: F.linear(input, weight, out=output_ref), args.warmup_iters, args.benchmark_iters)
            input, codebooks, codes, scales = (
                input.numpy(),
                codebooks.squeeze(-2).numpy(),
                codes.view(torch.uint8).numpy(),
                scales.numpy(),
            )
            quant += benchmark(
                lambda: aqlm_gemv_lut(input, codebooks, codes, scales), args.warmup_iters, args.benchmark_iters
            )

        print(f"{model}: Dense forward = {dense * 1e3:.2f} ms")
        print(f"{model}: Quant forward = {quant * 1e3:.2f} ms")
        print(f"{model}: Speedup relative to dense = {(dense / quant):.3f}")
