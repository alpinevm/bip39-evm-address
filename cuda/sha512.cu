
/// Adapted from https://github.com/dave-andersen/cudapts/blob/305504ae28dccbf9a374bbc94cdb8c1a4f58b89f/src/gpuhash.cu

/***************************************************
 * SHA-512 GPU Implementation
 *
 * Date: 13 April 2025
 ***************************************************/

#include "config.h"

#ifndef rotl64
#define rotl64(x, n) ((x << n) | (x >> (64 - n)))
#endif
#ifndef rotr64
#define rotr64(x, n) ((x >> n) | (x << (64 - n)))
#endif

/***************************************************
 * Constants, macros, and device state
 ***************************************************/

#define SHA512_BLOCK_SIZE 128
#define SHA512_DIGEST_SIZE 64

__constant__ const LONG CUDA_SHA512_IV[8] = {
    0x6a09e667f3bcc908ULL,
    0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL,
    0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL,
    0x5be0cd19137e2179ULL};

__constant__ const LONG CUDA_SHA512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};

#define CUDA_SHA512_Ch(x, y, z) ((x & y) ^ ((~x) & z))
#define CUDA_SHA512_Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define CUDA_SHA512_Sigma0(x) (rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39))
#define CUDA_SHA512_Sigma1(x) (rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41))
#define CUDA_SHA512_sigma0(x) (rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7))
#define CUDA_SHA512_sigma1(x) (rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6))

__device__ __forceinline__ LONG cuda_sha512_swap64(LONG x)
{
    return ((x & 0x00000000000000FFULL) << 56) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x00000000FF000000ULL) << 8) |
           ((x & 0x000000FF00000000ULL) >> 8) |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0xFF00000000000000ULL) >> 56);
}

/***************************************************
 * Context structure
 ***************************************************/
typedef struct
{
    LONG state[8];
    BYTE buffer[SHA512_BLOCK_SIZE];
    LONG bitcount_high;
    LONG bitcount_low;
    int buffer_fill;
} cuda_sha512_ctx_t;
typedef cuda_sha512_ctx_t CUDA_SHA512_CTX;

/***************************************************
 * Internal helper: big-endian read from a pointer
 * (like Keccak had cuda_keccak_leuint64)
 ***************************************************/
__device__ __forceinline__ LONG cuda_sha512_load_be64(const void *src)
{
    LONG val;
    memcpy(&val, src, sizeof(val));
    return cuda_sha512_swap64(val);
}

/***************************************************
 * The core transform step: process one 1024-bit block
 ***************************************************/
__device__ void cuda_sha512_transform(CUDA_SHA512_CTX *ctx, const BYTE *block)
{
    LONG w[80];
    LONG a, b, c, d, e, f, g, h;

#pragma unroll 16
    for (int i = 0; i < 16; i++)
    {
        w[i] = cuda_sha512_load_be64(block + (i * 8));
    }
#pragma unroll
    for (int i = 16; i < 80; i++)
    {
        w[i] = CUDA_SHA512_sigma1(w[i - 2]) + w[i - 7] +
               CUDA_SHA512_sigma0(w[i - 15]) + w[i - 16];
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

#pragma unroll
    for (int i = 0; i < 80; i++)
    {
        LONG T1 = h + CUDA_SHA512_Sigma1(e) + CUDA_SHA512_Ch(e, f, g) + CUDA_SHA512_K[i] + w[i];
        LONG T2 = CUDA_SHA512_Sigma0(a) + CUDA_SHA512_Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

/***************************************************
 * Initialize the SHA-512 context
 ***************************************************/
__device__ void cuda_sha512_init(CUDA_SHA512_CTX *ctx)
{
    memset(ctx, 0, sizeof(CUDA_SHA512_CTX));

    ctx->state[0] = CUDA_SHA512_IV[0];
    ctx->state[1] = CUDA_SHA512_IV[1];
    ctx->state[2] = CUDA_SHA512_IV[2];
    ctx->state[3] = CUDA_SHA512_IV[3];
    ctx->state[4] = CUDA_SHA512_IV[4];
    ctx->state[5] = CUDA_SHA512_IV[5];
    ctx->state[6] = CUDA_SHA512_IV[6];
    ctx->state[7] = CUDA_SHA512_IV[7];

    ctx->bitcount_high = 0ULL;
    ctx->bitcount_low = 0ULL;
    ctx->buffer_fill = 0;
}

/***************************************************
 * Process input data and update the context
 ***************************************************/
__device__ void cuda_sha512_update(CUDA_SHA512_CTX *ctx, const BYTE *input, size_t inlen)
{
    LONG bits_to_add = ((LONG)inlen) << 3;
    ctx->bitcount_low += bits_to_add;
    if (ctx->bitcount_low < bits_to_add)
    {
        ctx->bitcount_high++;
    }

    int buffer_idx = ctx->buffer_fill;
    int to_fill = SHA512_BLOCK_SIZE - buffer_idx;
    size_t idx = 0;

    if (buffer_idx && inlen >= (size_t)to_fill)
    {
        memcpy(ctx->buffer + buffer_idx, input, to_fill);
        cuda_sha512_transform(ctx, ctx->buffer);
        idx += to_fill;
        buffer_idx = 0;
    }

    while ((idx + SHA512_BLOCK_SIZE) <= inlen)
    {
        cuda_sha512_transform(ctx, input + idx);
        idx += SHA512_BLOCK_SIZE;
    }

    if (idx < inlen)
    {
        memcpy(ctx->buffer + buffer_idx, input + idx, inlen - idx);
        buffer_idx += (int)(inlen - idx);
    }
    ctx->buffer_fill = buffer_idx;
}

/***************************************************
 * Final padding, produce the digest output
 ***************************************************/
__device__ void cuda_sha512_final(CUDA_SHA512_CTX *ctx, BYTE *out_digest)
{
    int buffer_idx = ctx->buffer_fill;
    ctx->buffer[buffer_idx++] = 0x80;

    if (buffer_idx > (SHA512_BLOCK_SIZE - 16))
    {
        memset(ctx->buffer + buffer_idx, 0, SHA512_BLOCK_SIZE - buffer_idx);
        cuda_sha512_transform(ctx, ctx->buffer);
        buffer_idx = 0;
    }
    memset(ctx->buffer + buffer_idx, 0, (SHA512_BLOCK_SIZE - 16) - buffer_idx);

    LONG high_be = cuda_sha512_swap64(ctx->bitcount_high);
    LONG low_be = cuda_sha512_swap64(ctx->bitcount_low);

    memcpy(ctx->buffer + (SHA512_BLOCK_SIZE - 16), &high_be, 8);
    memcpy(ctx->buffer + (SHA512_BLOCK_SIZE - 8), &low_be, 8);

    cuda_sha512_transform(ctx, ctx->buffer);

    for (int i = 0; i < 8; i++)
    {
        LONG tmp = cuda_sha512_swap64(ctx->state[i]);
        memcpy(out_digest + (i * 8), &tmp, 8);
    }
}
