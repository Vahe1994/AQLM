#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

template<bool use_bfloat16, size_t group_size>
__global__ void Code1x16MatVec(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / group_size;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  __shared__ int4 sh_b[32 * (group_size + 1)];
  float res = 0;

  int iters = (prob_k / group_size + group_size * 32 - 1) / (group_size * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * group_size; i += blockDim.x) {
      if (8 * (b_gl_rd + i) < prob_k)
        sh_b[(group_size + 1) * (i / group_size) + i % group_size] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * group_size;

    int b_sh_rd = (group_size + 1) * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[group_size / 2];
        // We bypass the L1 cache to avoid massive amounts of memory streaming that doesn't
        // actually help us; this brings > 2x speedup.
        asm volatile (
          "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
          : "l"((void*) &codebook[(group_size / 8) * enc[i]])
        );
        if constexpr (group_size == 16) {
          asm volatile (
            "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(dec[4]), "=r"(dec[5]), "=r"(dec[6]), "=r"(dec[7])
            : "l"((void*) &codebook[(group_size / 8) * enc[i] + 1])
          );
        }
        if constexpr (use_bfloat16) {
        #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
          nv_bfloat162* a = reinterpret_cast<nv_bfloat162*>(&dec);
          nv_bfloat162* b = reinterpret_cast<nv_bfloat162*>(&sh_b[b_sh_rd]);
          nv_bfloat162 res2 = {};
          #pragma unroll
          for (int j = 0; j < group_size / 2; j++)
            res2 = __hfma2(a[j], b[j], res2);
          res += __bfloat162float(res2.x) + __bfloat162float(res2.y);
        #endif
        } else {
          half2* a = reinterpret_cast<half2*>(&dec);
          half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
          half2 res2 = {};
          #pragma unroll
          for (int j = 0; j < group_size / 2; j++)
            res2 = __hfma2(a[j], b[j], res2);
          res += __half2float(res2.x) + __half2float(res2.y);
        }
        b_sh_rd += group_size / 8;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
    #pragma unroll
    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      if constexpr (use_bfloat16) {
        reinterpret_cast<__nv_bfloat16*>(C)[c_gl_wr] = __float2bfloat16(res);
      } else {
        reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
      }
    }
  }
}


template<size_t group_size>
__global__ void Code1x16Dequant(
  const int4* __restrict__ A,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / group_size;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  int iters = (prob_k / group_size + group_size * 32 - 1) / (group_size * 32);
  while (iters--) {
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[group_size / 2];
        // We bypass the L1 cache to avoid massive amounts of memory streaming that doesn't
        // actually help us; this brings > 2x speedup.
        asm volatile (
          "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
          : "l"((void*) &codebook[(group_size / 8) * enc[i]])
        );
        if constexpr (group_size == 16) {
          asm volatile (
            "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(dec[4]), "=r"(dec[5]), "=r"(dec[6]), "=r"(dec[7])
            : "l"((void*) &codebook[(group_size / 8) * enc[i] + 1])
          );
        }

        C[a_gl_rd * group_size + (group_size / 8) * i] = reinterpret_cast<int4*>(&dec)[0];
        if constexpr (group_size == 16) {
          C[a_gl_rd * group_size + (group_size / 8) * i + 1] = reinterpret_cast<int4*>(&dec)[1];
        }
      }
    }
    a_gl_rd += 32;
  }
}

template<bool use_bfloat16>
__global__ void Code2x8MatVec(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;
  int lane = threadIdx.x % 8;

  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_code = sh_b + 32 * 9;
  int4* sh_code0 = sh_code;
  int4* sh_code1 = sh_code + 256 * 8;

  for (int i = threadIdx.x; i < 2 * 256; i += blockDim.x) {
    int4 dec = codebook[i];
    #pragma unroll
    for (int j = 0; j < 8; j++)
      sh_code[8 * i + (j + lane) % 8] = dec;
  }
  __syncthreads();

  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8)
        sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        if constexpr (use_bfloat16) {
        #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
          nv_bfloat162* a0 = reinterpret_cast<nv_bfloat162*>(&sh_code0[8 * enc[2 * i + 0] + lane]);
          nv_bfloat162* a1 = reinterpret_cast<nv_bfloat162*>(&sh_code1[8 * enc[2 * i + 1] + lane]);
          nv_bfloat162*  b = reinterpret_cast<nv_bfloat162*>(&sh_b[b_sh_rd]);
          nv_bfloat162 res2 = {};
          #pragma unroll
          for (int j = 0; j < 4; j++)
            res2 = __hfma2(__hadd2(a0[j], a1[j]), b[j], res2);
          res += __bfloat162float(res2.x) + __bfloat162float(res2.y);
        #endif
        } else {
          half2* a0 = reinterpret_cast<half2*>(&sh_code0[8 * enc[2 * i + 0] + lane]);
          half2* a1 = reinterpret_cast<half2*>(&sh_code1[8 * enc[2 * i + 1] + lane]);
          half2*  b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
          half2 res2 = {};
          #pragma unroll
          for (int j = 0; j < 4; j++)
            res2 = __hfma2(__hadd2(a0[j], a1[j]), b[j], res2);
          res += __half2float(res2.x) + __half2float(res2.y);
        }
        b_sh_rd++;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
    #pragma unroll
    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      if constexpr (use_bfloat16) {
        reinterpret_cast<__nv_bfloat16*>(C)[c_gl_wr] = __float2bfloat16(res);
      } else {
        reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
      }
    }
  }
}

template<bool use_bfloat16>
__global__ void Code2x8Dequant(
  const int4* __restrict__ A,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = a_gl_rd < prob_m;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;
  int lane = threadIdx.x % 8;

  int c_gl_stride = prob_k / 8;
  int c_gl_wr = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  c_gl_wr = c_gl_stride * c_gl_wr + (threadIdx.x % 32) * 8;

  extern __shared__ int4 sh[];
  int4* sh_code = sh;
  int4* sh_code0 = sh_code;
  int4* sh_code1 = sh_code + 256 * 8;

  for (int i = threadIdx.x; i < 2 * 256; i += blockDim.x) {
    int4 dec = codebook[i];
    #pragma unroll
    for (int j = 0; j < 8; j++)
      sh_code[8 * i + (j + lane) % 8] = dec;
  }
  __syncthreads();

  int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
  while (iters--) {
    if (pred && a_gl_rd < a_gl_end) {
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        int4 chunk;
        if constexpr (use_bfloat16) {
          #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
          nv_bfloat162* a0 = reinterpret_cast<nv_bfloat162*>(&sh_code0[8 * enc[2 * i + 0] + lane]);
          nv_bfloat162* a1 = reinterpret_cast<nv_bfloat162*>(&sh_code1[8 * enc[2 * i + 1] + lane]);
          #pragma unroll
          for (int j = 0; j < 4; j++)
            reinterpret_cast<nv_bfloat162*>(&chunk)[j] = __hadd2(a0[j], a1[j]);
          #endif
        } else {
          half2* a0 = reinterpret_cast<half2*>(&sh_code0[8 * enc[2 * i + 0] + lane]);
          half2* a1 = reinterpret_cast<half2*>(&sh_code1[8 * enc[2 * i + 1] + lane]);
          #pragma unroll
          for (int j = 0; j < 4; j++)
            reinterpret_cast<half2*>(&chunk)[j] = __hadd2(a0[j], a1[j]);
        }
        C[a_gl_rd * 8 + i] = chunk;
      }
    }
    a_gl_rd += 32;
  }
}

template<bool use_bfloat16, int K>
__global__ void CodeKx8MatVec(
  const uint8_t* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_code = sh_b + 32 * 9;

  float res_accum = 0;
  int c_gl_wr = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = c_gl_wr < prob_m;

  for (int codebook_idx = 0; codebook_idx < K; codebook_idx++) {
    int a_gl_stride = prob_k / 8 * K;
    int a_gl_rd = c_gl_wr;
    int b_gl_rd = 0;
    a_gl_rd = a_gl_stride * a_gl_rd + (threadIdx.x % 32) * 8 * K;
    int a_gl_end = a_gl_rd + a_gl_stride - (threadIdx.x % 32) * 8 * K;
    int lane = threadIdx.x % 8;

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      int4 dec = codebook[i + 256 * codebook_idx];
      #pragma unroll
      for (int j = 0; j < 8; j++)
        sh_code[8 * i + (j + lane) % 8] = dec;
    }
    __syncthreads();
    float res = 0;

    int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
    while (iters--) {
      // We pad shared memory to avoid bank conflicts during reads
      __syncthreads();
      for (int i = threadIdx.x; i < 32 * 8; i += blockDim.x) {
        if (b_gl_rd + i < prob_k / 8)
          sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
      }
      __syncthreads();
      b_gl_rd += 32 * 8;

      int b_sh_rd = 9 * (threadIdx.x % 32);
      if (pred && a_gl_rd < a_gl_end) {
        const uint8_t* enc = &A[a_gl_rd];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
          if constexpr (use_bfloat16) {
          #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
            nv_bfloat162* a = reinterpret_cast<nv_bfloat162*>(&sh_code[8 * enc[K * i + codebook_idx] + lane]);
            nv_bfloat162* b = reinterpret_cast<nv_bfloat162*>(&sh_b[b_sh_rd]);
            nv_bfloat162 res2 = {};
            #pragma unroll
            for (int j = 0; j < 4; j++)
              res2 = __hfma2(a[j], b[j], res2);
            res += __bfloat162float(res2.x) + __bfloat162float(res2.y);
          #endif
          } else {
            half2* a = reinterpret_cast<half2*>(&sh_code[8 * enc[K * i + codebook_idx] + lane]);
            half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
            half2 res2 = {};
            #pragma unroll
            for (int j = 0; j < 4; j++)
              res2 = __hfma2(a[j], b[j], res2);
            res += __half2float(res2.x) + __half2float(res2.y);
          }
          b_sh_rd++;
        }
        a_gl_rd += 32 * 8 * K;
      }
    }

    if (pred) {
      #pragma unroll
      for (int i = 16; i > 0; i /= 2)
        res += __shfl_down_sync(0xffffffff, res, i);
      res_accum += res;
    }

    __syncthreads();
  }

  if (pred) {
    if (threadIdx.x % 32 == 0) {
      if constexpr (use_bfloat16) {
        reinterpret_cast<__nv_bfloat16*>(C)[c_gl_wr] = __float2bfloat16(res_accum);
      } else {
        reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res_accum);
      }
    }
  }
}

template<bool use_bfloat16, int K>
__global__ void CodeKx8Dequant(
  const uint8_t* __restrict__ A,
        int4* __restrict__ C,
  const int4* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  extern __shared__ int4 sh[];
  int4* sh_code = sh;

  for (int codebook_idx = 0; codebook_idx < K; codebook_idx++) {
    int a_gl_stride = prob_k / 8 * K;
    int a_gl_rd = ((blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32));
    bool pred = a_gl_rd < prob_m;
    a_gl_rd = a_gl_stride * a_gl_rd + (threadIdx.x % 32) * 8 * K;
    int a_gl_end = a_gl_rd + a_gl_stride - (threadIdx.x % 32) * 8 * K;
    int lane = threadIdx.x % 8;

    int c_gl_stride = prob_k / 8;
    int c_gl_wr = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
    c_gl_wr = c_gl_stride * c_gl_wr + (threadIdx.x % 32) * 8;

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      int4 dec = codebook[i + 256 * codebook_idx];
      #pragma unroll
      for (int j = 0; j < 8; j++)
        sh_code[8 * i + (j + lane) % 8] = dec;
    }
    __syncthreads();

    int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
    while (iters--) {
      if (pred && a_gl_rd < a_gl_end) {
        const uint8_t* enc = &A[a_gl_rd];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
          int4* c_ptr = &C[a_gl_rd / K + i];
          if constexpr (use_bfloat16) {
            #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
              nv_bfloat162* c_bf16_ptr = reinterpret_cast<nv_bfloat162*>(c_ptr);
              nv_bfloat162* a = reinterpret_cast<nv_bfloat162*>(&sh_code[8 * enc[K * i + codebook_idx] + lane]);
              if (codebook_idx != 0) {
                  #pragma unroll
                  for (int j = 0; j < 4; j++) {
                    c_bf16_ptr[j] = __hadd2(c_bf16_ptr[j], a[j]);
                  }
              } else {
                  #pragma unroll
                  for (int j = 0; j < 4; j++) {
                    c_bf16_ptr[j] = a[j];
                  }
              }
            #endif
          } else {
            half2* c_fp16_ptr = reinterpret_cast<half2*>(c_ptr);
            half2* a = reinterpret_cast<half2*>(&sh_code[8 * enc[K * i + codebook_idx] + lane]);
            if (codebook_idx != 0) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                  c_fp16_ptr[j] = __hadd2(c_fp16_ptr[j], a[j]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                  c_fp16_ptr[j] = a[j];
                }
            }
          }
        }
      }
      a_gl_rd += 32 * 8 * K;
    }

    __syncthreads();
  }
}

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

const int THREAD_M = 16;

template <bool use_bfloat16, size_t group_size>
void  code1x16_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  if (cc_major < 8 && use_bfloat16) {
    throw c10::TypeError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "You're trying to run AQLM with bfloat16 on a GPU with low compute capability. Use torch.float16 instead."
      )
    );
  }

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16MatVec<use_bfloat16, group_size><<<blocks, threads, 16*32*(group_size + 1), stream>>>(
    (const int4*) A,
    (const int4*) B,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}

template void code1x16_matvec_cuda<false, 8>(const void*, const void*, void*, const void*, int, int);
template void code1x16_matvec_cuda<true, 8>(const void*, const void*, void*, const void*, int, int);
template void code1x16_matvec_cuda<false, 16>(const void*, const void*, void*, const void*, int, int);
template void code1x16_matvec_cuda<true, 16>(const void*, const void*, void*, const void*, int, int);

template <size_t group_size>
void  code1x16_dequant_cuda(
  const void* __restrict__ A,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16Dequant<group_size><<<blocks, threads, 0, stream>>>(
    (const int4*) A,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}

template void code1x16_dequant_cuda<8>(const void*, void*, const void*, int, int);
template void code1x16_dequant_cuda<16>(const void*, void*, const void*, int, int);

template <bool use_bfloat16>
void  code2x8_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  int cc_minor;
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
  if (cc_major < 8 && use_bfloat16) {
    throw c10::TypeError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "You're trying to run AQLM with bfloat16 on a GPU with low compute capability. Use torch.float16 instead."
      )
    );
  }

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const bool is_turing = cc_major == 7 && cc_minor == 5;
  if (!is_turing) {
    int shared = 16 * (2 * 256 * 8 + 32 * 9);
    cudaFuncSetAttribute(
      Code2x8MatVec<use_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
    );
    Code2x8MatVec<use_bfloat16><<<blocks, threads, shared, stream>>>(
      (const int4*) A,
      (const int4*) B,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  } else {
    int shared = 16 * (256 * 8 + 32 * 9);
    cudaFuncSetAttribute(
      CodeKx8MatVec<use_bfloat16, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
    );
    CodeKx8MatVec<use_bfloat16, 2><<<blocks, threads, shared, stream>>>(
      (const uint8_t*) A,
      (const int4*) B,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  }
}

template void code2x8_matvec_cuda<false>(const void*, const void*, void*, const void*, int, int);
template void code2x8_matvec_cuda<true>(const void*, const void*, void*, const void*, int, int);

void  code2x8_dequant_cuda(
  const void* __restrict__ A,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
) {
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  int cc_minor;
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
  if (cc_major < 8 && use_bfloat16) {
    throw c10::TypeError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "You're trying to run AQLM with bfloat16 on a GPU with low compute capability. Use torch.float16 instead."
      )
    );
  }

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const bool is_turing = cc_major == 7 && cc_minor == 5;
  if (!is_turing) {
    int shared = 16 * (2 * 256 * 8 + 32 * 9);
    if (use_bfloat16) {
      cudaFuncSetAttribute(
        Code2x8Dequant<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
      );
      Code2x8Dequant<true><<<blocks, threads, shared, stream>>>(
        (const int4*) A,
        (int4*) C,
        (const int4*) codebook,
        prob_m,
        prob_k
      );
    } else {
      cudaFuncSetAttribute(
        Code2x8Dequant<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
      );
      Code2x8Dequant<false><<<blocks, threads, shared, stream>>>(
        (const int4*) A,
        (int4*) C,
        (const int4*) codebook,
        prob_m,
        prob_k
      );
    }
  } else {
    int shared = 16 * (256 * 8 + 32 * 9);
    if (use_bfloat16) {
      cudaFuncSetAttribute(
        CodeKx8Dequant<true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
      );
      CodeKx8Dequant<true, 2><<<blocks, threads, shared, stream>>>(
        (const uint8_t*) A,
        (int4*) C,
        (const int4*) codebook,
        prob_m,
        prob_k
      );
    } else {
      cudaFuncSetAttribute(
        CodeKx8Dequant<false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
      );
      CodeKx8Dequant<false, 2><<<blocks, threads, shared, stream>>>(
        (const uint8_t*) A,
        (int4*) C,
        (const int4*) codebook,
        prob_m,
        prob_k
      );
    }
  }
}

template <bool use_bfloat16>
void  code1x8_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k
) {
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  int cc_minor;
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
  if (cc_major < 8 && use_bfloat16) {
    throw c10::TypeError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "You're trying to run AQLM with bfloat16 on a GPU with low compute capability. Use torch.float16 instead."
      )
    );
  }

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int shared = 16 * (256 * 8 + 32 * 9);
  cudaFuncSetAttribute(
    CodeKx8MatVec<use_bfloat16, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
  );
  CodeKx8MatVec<use_bfloat16, 1><<<blocks, threads, shared, stream>>>(
    (const uint8_t*) A,
    (const int4*) B,
    (int4*) C,
    (const int4*) codebook,
    prob_m,
    prob_k
  );
}

template void code1x8_matvec_cuda<false>(const void*, const void*, void*, const void*, int, int);
template void code1x8_matvec_cuda<true>(const void*, const void*, void*, const void*, int, int);

void code1x8_dequant_cuda(
  const void* __restrict__ A,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
) {
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  int cc_minor;
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
  if (cc_major < 8 && use_bfloat16) {
    throw c10::TypeError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "You're trying to run AQLM with bfloat16 on a GPU with low compute capability. Use torch.float16 instead."
      )
    );
  }

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int shared = 16 * (256 * 8 + 32 * 9);
  if (use_bfloat16) {
    cudaFuncSetAttribute(
      CodeKx8Dequant<true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
    );
    CodeKx8Dequant<true, 1><<<blocks, threads, shared, stream>>>(
      (const uint8_t*) A,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  } else {
    cudaFuncSetAttribute(
      CodeKx8Dequant<false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
    );
    CodeKx8Dequant<false, 1><<<blocks, threads, shared, stream>>>(
      (const uint8_t*) A,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  }
}
