import argparse
import time

import torch
import torch.nn.functional as F
from aqlm.inference_kernels.cuda_kernel import CUDA_KERNEL
from aqlm.utils import _dequantize_weight, pack_int_data, unpack_int_data
from torch import nn


def benchmark(f, warmup=10, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.perf_counter()
    torch.cuda.synchronize()
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
    assert torch.cuda.is_available()
    device = torch.device("cuda")

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
        "--nbits_per_codebook",
        type=int,
        default=16,
        help="Number of bits per codebook.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=1,
        help="Number of num_codebooks.",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="Input group size.",
    )

    args = parser.parse_args()

    for model, layers in MODELS.items():
        dense = 0
        quant = 0
        for in_features, out_features in layers:
            input = torch.randn((1, 1, in_features), dtype=torch.half, device=device)  #  [..., in_features]
            codes = pack_int_data(
                torch.randint(
                    2**args.nbits_per_codebook,
                    (out_features, in_features // args.in_group_size, args.num_codebooks),
                    device=device,
                ),  #  [num_out_groups, num_in_groups, num_codebooks]
                args.nbits_per_codebook,
            )
            codebooks = torch.randn(
                (args.num_codebooks, 2**args.nbits_per_codebook, 1, args.in_group_size),
                dtype=torch.half,
                device=device,
            )  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
            scales = torch.ones((out_features, 1, 1, 1), dtype=torch.half, device=device)  #  [num_out_groups, 1, 1, 1]

            weight = _dequantize_weight(unpack_int_data(codes, args.nbits_per_codebook), codebooks, scales).contiguous()

            output_ref = F.linear(input, weight)

            matmul = CUDA_KERNEL.code1x16_matmat if args.nbits_per_codebook == 16 else CUDA_KERNEL.code2x8_matmat

            output = matmul(input, codes, codebooks, scales)
            if args.log_error:
                print(
                    f"Relative error: {(torch.mean(torch.abs(output_ref - output)) / torch.mean(torch.abs(output_ref))).item():.2e}"
                )

            dense += benchmark(lambda: F.linear(input, weight, out=output_ref), args.warmup_iters, args.benchmark_iters)
            quant += benchmark(lambda: matmul(input, codes, codebooks, scales), args.warmup_iters, args.benchmark_iters)

        print(f"{model}: Dense forward = {dense * 1e6:.0f} mus")
        print(f"{model}: Quant forward = {quant * 1e6:.0f} mus")
        print(f"{model}: Speedup relative to dense = {(dense / quant):.3f}")
