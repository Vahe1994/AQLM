import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"

import argparse
import time
import torch
torch.set_num_threads(1)
from torch import nn
import torch.nn.functional as F

from aqlm.inference_kernels.numba import numba_gemm_lut
from aqlm.utils import _dequantize_weight, unpack_int_data, pack_int_data

def benchmark(f, warmup=10, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            tick = time.perf_counter()
    average_latency = (time.perf_counter() - tick) / iter
    time.sleep(1.0)
    return average_latency


MODELS = {
    'fuse': {
        'Llama 2 7B': [
            (4096, 3 * 4096),
            (4096, 4096),
            (4096, 2 * 11008),
            (11008, 4096)
        ],
        'Llama 2 13B': [
            (5120, 3 * 5120),
            (5120, 5120),
            (5120, 2 * 13824),
            (13824, 5120)
        ],
        'Llama 2 70B': [
            (8192, int(1.25 * 8192)),
            (8192, 8192),
            (8192, 2 * 28672),
            (28672, 8192)
        ],
    },
    'no-fuse': {
        'Llama 2 7B': [
            # (4096, 4096),
            # (4096, 4096),
            # (4096, 4096),
            # (4096, 4096),
            # (4096, 11008),
            (4096, 11008),
            # (11008, 4096)
        ],
        'Llama 2 13B': [
            # (5120, 5120),
            # (5120, 5120),
            # (5120, 5120),
            # (5120, 5120),
            # (5120, 13824),
            (5120, 13824),
            # (13824, 5120)
        ],
        'Llama 2 70B': [
            # (8192, 8192),
            # (8192, 1024),
            # (8192, 1024),
            # (8192, 8192),
            # (8192, 28672),
            (8192, 28672),
            # (28672, 8192)
        ],
    }
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
        default=10,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--log_error",
        action="store_true",
    )
    parser.add_argument(
        "--fuse",
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

    args = parser.parse_args()

    for model, layers in MODELS['fuse' if args.fuse else 'no-fuse'].items():
        dense = 0
        quant = 0
        for in_features, out_features in layers:
            input = torch.randn((1, in_features), dtype=torch.float32)  #  [..., in_features]
            codes = pack_int_data(
                torch.randint(2 ** args.nbits_per_codebook, (in_features // args.in_group_size, out_features, args.num_codebooks)),  #  [num_in_groups, num_out_groups, num_codebooks]
                args.nbits_per_codebook,
            )
            codebooks = torch.randn((args.num_codebooks, 2 ** args.nbits_per_codebook, 1, args.in_group_size), dtype=torch.float32)  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
            scales = torch.randn((out_features, 1, 1, 1), dtype=torch.float32)  #  [num_out_groups, 1, 1, 1]

            weight = _dequantize_weight(unpack_int_data(torch.permute(codes, (1, 0, 2)), args.nbits_per_codebook), codebooks, scales).contiguous()
            
            output_ref = F.linear(input, weight)
            output = numba_gemm_lut(input, codes, codebooks, scales, None)
            if args.log_error:
                print(f"Relative error: {(torch.mean(torch.abs(output_ref - output)) / torch.mean(torch.abs(output_ref))).item():.2e}")

            dense += benchmark(lambda: F.linear(input, weight, out=output_ref), args.warmup_iters, args.benchmark_iters)
            quant += benchmark(lambda: numba_gemm_lut(input, codes, codebooks, scales, None), args.warmup_iters, args.benchmark_iters)
        print(f"{model}: Dense forward = {dense * 1e3:.2f} ms")
        print(f"{model}: Quant forward = {quant * 1e3:.2f} ms")
        print(f"{model}: Speedup relative to dense = {(dense / quant):.3f}")
