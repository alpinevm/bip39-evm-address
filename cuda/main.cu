#include "config.h"
#include "keccak.cu"
#include "sha512.cu"
#include "secp256k1.cu"

extern "C" __global__ void square_num(const int *input, const int input_count, int *output)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_idx < input_count)
    {
        output[thread_idx] = input[thread_idx] * input[thread_idx];
    }
}

extern "C" __global__ void keccak256_hash(BYTE *data, WORD data_len, BYTE *output)
{
    CUDA_KECCAK_CTX ctx;
    cuda_keccak_init(&ctx, 256);
    cuda_keccak_update(&ctx, data, data_len);
    cuda_keccak_final(&ctx, output);
}

extern "C" __global__ void sha512_hash(BYTE *data, WORD data_len, BYTE *output)
{
    CUDA_SHA512_CTX ctx;
    cuda_sha512_init(&ctx);
    cuda_sha512_update(&ctx, data, data_len);
    cuda_sha512_final(&ctx, output);
}

extern "C" __global__ void secp256k1_compute_pubkey(const unsigned int k[8], unsigned int Px[8], unsigned int Py[8])
{
    compute_public_key(k, Px, Py);
}
