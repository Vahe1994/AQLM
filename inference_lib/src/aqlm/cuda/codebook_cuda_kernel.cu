#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

const int SHM = 48 * 1024;

__global__ void Code16MatVec(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  if (a_gl_rd >= prob_m)
    return;
  // We pad shared memory to avoid bank conflicts during reads
  int b_sh_rd = 9 * (threadIdx.x % 32);
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;

  __shared__ int4 sh_b[SHM / 16];
  for (int i = threadIdx.x; i < prob_k / 8; i+= blockDim.x)
    sh_b[9 * (i / 8) + i % 8] = B[i];
  __syncthreads();

  float res = 0;

  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;
  while (a_gl_rd < a_gl_end) { 
    const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      uint32_t dec[4];
      // We bypass the L1 cache to avoid massive amounts of memory streaming that doesn't
      // actually help us; this brings > 2x speedup.
      asm volatile (
        "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
        : "l"((void*) &codebook[enc[i]])
      );
      half2* a = reinterpret_cast<half2*>(&dec);
      half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
      half2 res2 = {};
      #pragma unroll
      for (int j = 0; j < 4; j++)
        res2 = __hfma2(a[j], b[j], res2);
      res += __half2float(res2.x) + __half2float(res2.y);
      b_sh_rd++;
    }
    a_gl_rd += 32;
    b_sh_rd += 31 * 9 + 1;
  }

  #pragma unroll
  for (int i = 16; i > 0; i /= 2)
    res += __shfl_down_sync(0xffffffff, res, i);
  if (threadIdx.x % 32 == 0)
    reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
}

const int THREAD_M = 16;

void  code16_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int blocks = (prob_m + THREAD_M - 1) / THREAD_M;
  int threads = 32 * THREAD_M;
  Code16MatVec<<<blocks, threads>>>(
    (const int4*) A,
    (const int4*) B,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}
