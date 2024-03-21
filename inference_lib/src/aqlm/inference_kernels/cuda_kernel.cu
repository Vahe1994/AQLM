#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>

template<bool use_bfloat16>
__global__ void Code1x16MatVec(
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

  __shared__ int4 sh_b[32 * 9];
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
        if constexpr (use_bfloat16) {
        #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
          nv_bfloat162* a = reinterpret_cast<nv_bfloat162*>(&dec);
          nv_bfloat162* b = reinterpret_cast<nv_bfloat162*>(&sh_b[b_sh_rd]);
          nv_bfloat162 res2 = {};
          #pragma unroll
          for (int j = 0; j < 4; j++)
            res2 = __hfma2(a[j], b[j], res2);
          res += __bfloat162float(res2.x) + __bfloat162float(res2.y);
        #endif
        } else {
          half2* a = reinterpret_cast<half2*>(&dec);
          half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
          half2 res2 = {};
          #pragma unroll
          for (int j = 0; j < 4; j++)
            res2 = __hfma2(a[j], b[j], res2);
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
__global__ void Code1x16Dequant(
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

  int c_gl_stride = prob_k / 8;
  int c_gl_wr = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  c_gl_wr = c_gl_stride * c_gl_wr + (threadIdx.x % 32) * 8;

  int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
  while (iters--) {
    int offset = 8 * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        int4 chunk;
        auto dec = reinterpret_cast<uint32_t*>(&chunk);
        // We bypass the L1 cache to avoid massive amounts of memory streaming that doesn't
        // actually help us; this brings > 2x speedup.
        asm volatile (
          "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
          : "l"((void*) &codebook[enc[i]])
        );

        C[a_gl_rd * 8 + i] = chunk;
      }
    }
    a_gl_rd += 32;
  }
}

// Dequantizes the code and codebook into weights.
// We span horizontally and do an int4 at a time in an attempt to maximize throughput.
__global__ void Code1x16Dequant(
        int4* __restrict__ weights,
  const int4* __restrict__ a,
  const int4* __restrict__ codebook,
  const int a_rows, // code rows in int4 space, so same as stride.
  const int a_cols, // code columns (matter?)
  const int4 codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at most 3 long, sums to m.
  const int codebook_stride // as int4
) {
  // Each thread decodes one int4 worth of codebook.
  int a_col = blockIdx.x * 32 + threadIdx.x;
  int a_row = blockIdx.y * 32 + threadIdx.y;

  // out of range
  if (a_row >= a_rows)
    return;

  const int weight_stride = a_rows * 8; // as int4
  weights += a_col * weight_stride + a_row * 8;

  // advance to the correct codebook, this easy because we only multiply one column of the codebook.
  auto codebook_size = &codebook_a_sizes.x;
  while (a_col >= *codebook_size)
  {
      codebook += codebook_stride;
      ++codebook_size;
  }

  // do one int4 read and write, hopefully maxing out bandwidth.
  int4 code_block = a[a_row + a_col * a_rows];
  const uint16_t* enc = reinterpret_cast<const uint16_t*>(&code_block);
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    weights[i] = codebook[enc[i]];
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

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

const int THREAD_M = 16;

void  code1x16_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
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
  if (use_bfloat16) {
    Code1x16MatVec<true><<<blocks, threads, 16*32*9, stream>>>(
      (const int4*) A,
      (const int4*) B,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  } else {
    Code1x16MatVec<false><<<blocks, threads, 16*32*9, stream>>>(
      (const int4*) A,
      (const int4*) B,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  }
}

void  code1x16_dequant_cuda(
  const void* __restrict__ A,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
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
  if (use_bfloat16) {
    Code1x16Dequant<true><<<blocks, threads, 0, stream>>>(
      (const int4*) A,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  } else {
    Code1x16Dequant<false><<<blocks, threads, 0, stream>>>(
      (const int4*) A,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  }
}

// Dequantizes the code and codebook into weights.
void code1x16_dequant_vllm(
        void* __restrict__ weights,
  const void* __restrict__ a,
  const void* __restrict__ codebook,
  const int a_rows, // code rows in element space, so k
  const int a_cols, // code columns in element space, so n
  const int4 codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at most 3 long, sums to m.
  const int codebook_stride // as int4
) {
  dim3 threads(32, 32, 1);

  assert(a_cols % 32 == 0);
  // each thread does one int4 worth.
  assert(a_rows % 8 == 0);

  const int rows = a_rows/8;

  dim3 blocks(ceildiv(a_cols, 32), ceildiv(rows, 32), 1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Code1x16Dequant<<<blocks, threads, 0, stream>>>(
    (int4*) weights,
    (const int4*) a,
    (const int4*) codebook,
    rows, // in int4 space.
    a_cols,
    codebook_a_sizes,
    codebook_stride
  );
}

void  code2x8_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ codebook,
  int prob_m,
  int prob_k,
  bool use_bfloat16
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
  int shared = 16 * (2 * 256 * 8 + 32 * 9);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (use_bfloat16) {
    cudaFuncSetAttribute(
      Code2x8MatVec<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
    );
    Code2x8MatVec<true><<<blocks, threads, shared, stream>>>(
      (const int4*) A,
      (const int4*) B,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  } else {
    cudaFuncSetAttribute(
      Code2x8MatVec<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
    );
    Code2x8MatVec<false><<<blocks, threads, shared, stream>>>(
      (const int4*) A,
      (const int4*) B,
      (int4*) C,
      (const int4*) codebook,
      prob_m,
      prob_k
    );
  }
}
