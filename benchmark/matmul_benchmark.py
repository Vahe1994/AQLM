import argparse
import time
import torch

from aqlm.cuda.cuda_kernel import CUDA_KERNEL


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
            (4096, 4096),
            (4096, 4096),
            (4096, 4096),
            (4096, 4096),
            (4096, 11008),
            (4096, 11008),
            (11008, 4096)
        ],
        'Llama 2 13B': [
            (5120, 5120),
            (5120, 5120),
            (5120, 5120),
            (5120, 5120),
            (5120, 13824),
            (5120, 13824),
            (13824, 5120)
        ],
        'Llama 2 70B': [
            (8192, 8192),
            (8192, 1024),
            (8192, 1024),
            (8192, 8192),
            (8192, 28672),
            (8192, 28672),
            (28672, 8192)
        ],
    }
}


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device('cuda')

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

    args = parser.parse_args()

    codebook = torch.randn((2 ** 16, 8), dtype=torch.half, device=device)

    for model, layers in MODELS['fuse' if args.fuse else 'no-fuse'].items():
        dense = 0
        quant = 0
        for K, M in layers:
            A = torch.randint(2 ** 16, (M, K // 8), dtype=torch.int, device=device)
            A_ref = torch.vstack([codebook[A[i]].flatten().unsqueeze(0) for i in range(M)])
            A = A.to(torch.int16)
            B = torch.randn((K, 1), dtype=torch.half, device=device)
            C = torch.zeros((M, 1), dtype=torch.half, device=device)

            C_ref = torch.matmul(A_ref, B)
            CUDA_KERNEL.code1x16_matvec(A, B, C, codebook)
            if args.log_error:
                print(f"Relative error: {(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref))).item():.2e}")

            dense += benchmark(lambda: torch.matmul(A_ref, B, out=C), args.warmup_iters, args.benchmark_iters)
            quant += benchmark(lambda: CUDA_KERNEL.code1x16_matvec(A, B, C, codebook), args.warmup_iters, args.benchmark_iters)
        print(f"{model}: Speedup relative to dense = {(dense / quant):.3f}")
