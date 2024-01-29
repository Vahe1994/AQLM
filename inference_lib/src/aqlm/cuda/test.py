import time
import torch

from aqlm.cuda.cuda_kernel import CUDA_KERNEL

def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res

DEV = torch.device('cuda:0')

MODELS = {
    'Llama7B': [
        (4096, 3 * 4096),
        (4096, 4096),
        (4096, 2 * 10752),
        (10752, 4096)
    ],
    'Llama13B': [
        (5120, 3 * 5120),
        (5120, 5120),
        (5120, 2 * 13568),
        (13568, 5120)
    ],
    'Llama33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    'Llama65B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ],
}

codebook = torch.randn((2 ** 16, 8), dtype=torch.half, device=DEV)

for model, layers in MODELS.items():
    dense = 0
    quant = 0
    for K, M in layers:
        A = torch.randint(2 ** 16, (M, K // 8), dtype=torch.int, device=DEV)
        # A_ref = torch.vstack([(codebook[A[i] & 0xff] + codebook[256 + (A[i] >> 8)]).flatten().unsqueeze(0) for i in range(M)])
        A_ref = torch.vstack([codebook[A[i]].flatten().unsqueeze(0) for i in range(M)])
        A = A.to(torch.int16)
        B = torch.randn((K, 1), dtype=torch.half, device=DEV)
        C = torch.zeros((M, 1), dtype=torch.half, device=DEV)

        C_ref = torch.matmul(A_ref, B)
        CUDA_KERNEL.code1x16_matvec(A, B, C, codebook)
        print(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)))

        dense += benchmark(lambda: torch.matmul(A_ref, B, out=C))
        quant += benchmark(lambda: CUDA_KERNEL.code1x16_matvec(A, B, C, codebook))
        # quant += benchmark(lambda: codebook_cuda.code2x8_matvec(A, B, C, codebook))
    print(model, dense / quant)

