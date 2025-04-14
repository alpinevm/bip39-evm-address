#include <cuda.h>
#include <cuda_runtime.h>

#ifndef _PTX_H
#define _PTX_H

#include <cuda_runtime.h>

#define madc_hi(dest, a, x, b) asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_hi_cc(dest, a, x, b) asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define mad_hi_cc(dest, a, x, b) asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))

#define mad_lo_cc(dest, a, x, b) asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo(dest, a, x, b) asm volatile("madc.lo.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo_cc(dest, a, x, b) asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))

#define addc(dest, a, b) asm volatile("addc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define add_cc(dest, a, b) asm volatile("add.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define addc_cc(dest, a, b) asm volatile("addc.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define sub_cc(dest, a, b) asm volatile("sub.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define subc_cc(dest, a, b) asm volatile("subc.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define subc(dest, a, b) asm volatile("subc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define set_eq(dest, a, b) asm volatile("set.eq.u32.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define lsbpos(x) (__ffs((x)))

__device__ __forceinline__ unsigned int endian(unsigned int x)
{
    return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

#endif

/**
 Prime modulus 2^256 - 2^32 - 977
 */
__constant__ static unsigned int _P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F};

/**
 Base point X
 */
__constant__ static unsigned int _GX[8] = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798};

/**
 Base point Y
 */
__constant__ static unsigned int _GY[8] = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8};

/**
 * Group order
 */
__constant__ static unsigned int _N[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141};

__constant__ static unsigned int _BETA[8] = {
    0x7AE96A2B, 0x657C0710, 0x6E64479E, 0xAC3434E9, 0x9CF04975, 0x12F58995, 0xC1396C28, 0x719501EE};

__constant__ static unsigned int _LAMBDA[8] = {
    0x5363AD4C, 0xC05C30E0, 0xA5261C02, 0x8812645A, 0x122E22EA, 0x20816678, 0xDF02967C, 0x1B23BD72};

__device__ __forceinline__ bool isInfinity(const unsigned int x[8])
{
    bool isf = true;

    for (int i = 0; i < 8; i++)
    {
        if (x[i] != 0xffffffff)
        {
            isf = false;
        }
    }

    return isf;
}

__device__ __forceinline__ static void copyBigInt(const unsigned int src[8], unsigned int dest[8])
{
    for (int i = 0; i < 8; i++)
    {
        dest[i] = src[i];
    }
}

__device__ static bool equal(const unsigned int *a, const unsigned int *b)
{
    bool eq = true;

    for (int i = 0; i < 8; i++)
    {
        eq &= (a[i] == b[i]);
    }

    return eq;
}

/**
 * Reads an 8-word big integer from device memory
 */
__device__ static void readInt(const unsigned int *ara, int idx, unsigned int x[8])
{
    int totalThreads = gridDim.x * blockDim.x;

    int base = idx * totalThreads * 8;

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    int index = base + threadId;

    for (int i = 0; i < 8; i++)
    {
        x[i] = ara[index];
        index += totalThreads;
    }
}

__device__ static unsigned int readIntLSW(const unsigned int *ara, int idx)
{
    int totalThreads = gridDim.x * blockDim.x;

    int base = idx * totalThreads * 8;

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    int index = base + threadId;

    return ara[index + totalThreads * 7];
}

/**
 * Writes an 8-word big integer to device memory
 */
__device__ static void writeInt(unsigned int *ara, int idx, const unsigned int x[8])
{
    int totalThreads = gridDim.x * blockDim.x;

    int base = idx * totalThreads * 8;

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    int index = base + threadId;

    for (int i = 0; i < 8; i++)
    {
        ara[index] = x[i];
        index += totalThreads;
    }
}

/**
 * Subtraction mod p
 */
__device__ static void subModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    sub_cc(c[7], a[7], b[7]);
    subc_cc(c[6], a[6], b[6]);
    subc_cc(c[5], a[5], b[5]);
    subc_cc(c[4], a[4], b[4]);
    subc_cc(c[3], a[3], b[3]);
    subc_cc(c[2], a[2], b[2]);
    subc_cc(c[1], a[1], b[1]);
    subc_cc(c[0], a[0], b[0]);

    unsigned int borrow = 0;
    subc(borrow, 0, 0);

    if (borrow)
    {
        add_cc(c[7], c[7], _P[7]);
        addc_cc(c[6], c[6], _P[6]);
        addc_cc(c[5], c[5], _P[5]);
        addc_cc(c[4], c[4], _P[4]);
        addc_cc(c[3], c[3], _P[3]);
        addc_cc(c[2], c[2], _P[2]);
        addc_cc(c[1], c[1], _P[1]);
        addc(c[0], c[0], _P[0]);
    }
}

__device__ static unsigned int add(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    add_cc(c[7], a[7], b[7]);
    addc_cc(c[6], a[6], b[6]);
    addc_cc(c[5], a[5], b[5]);
    addc_cc(c[4], a[4], b[4]);
    addc_cc(c[3], a[3], b[3]);
    addc_cc(c[2], a[2], b[2]);
    addc_cc(c[1], a[1], b[1]);
    addc_cc(c[0], a[0], b[0]);

    unsigned int carry = 0;
    addc(carry, 0, 0);

    return carry;
}

__device__ static unsigned int sub(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    sub_cc(c[7], a[7], b[7]);
    subc_cc(c[6], a[6], b[6]);
    subc_cc(c[5], a[5], b[5]);
    subc_cc(c[4], a[4], b[4]);
    subc_cc(c[3], a[3], b[3]);
    subc_cc(c[2], a[2], b[2]);
    subc_cc(c[1], a[1], b[1]);
    subc_cc(c[0], a[0], b[0]);

    unsigned int borrow = 0;
    subc(borrow, 0, 0);

    return (borrow & 0x01);
}

__device__ static void addModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    add_cc(c[7], a[7], b[7]);
    addc_cc(c[6], a[6], b[6]);
    addc_cc(c[5], a[5], b[5]);
    addc_cc(c[4], a[4], b[4]);
    addc_cc(c[3], a[3], b[3]);
    addc_cc(c[2], a[2], b[2]);
    addc_cc(c[1], a[1], b[1]);
    addc_cc(c[0], a[0], b[0]);

    unsigned int carry = 0;
    addc(carry, 0, 0);

    bool gt = false;
    for (int i = 0; i < 8; i++)
    {
        if (c[i] > _P[i])
        {
            gt = true;
            break;
        }
        else if (c[i] < _P[i])
        {
            break;
        }
    }

    if (carry || gt)
    {
        sub_cc(c[7], c[7], _P[7]);
        subc_cc(c[6], c[6], _P[6]);
        subc_cc(c[5], c[5], _P[5]);
        subc_cc(c[4], c[4], _P[4]);
        subc_cc(c[3], c[3], _P[3]);
        subc_cc(c[2], c[2], _P[2]);
        subc_cc(c[1], c[1], _P[1]);
        subc(c[0], c[0], _P[0]);
    }
}

__device__ static void mulModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    unsigned int high[8] = {0};

    unsigned int t = a[7];

    // a[7] * b (low)
    for (int i = 7; i >= 0; i--)
    {
        c[i] = t * b[i];
    }

    // a[7] * b (high)
    mad_hi_cc(c[6], t, b[7], c[6]);
    madc_hi_cc(c[5], t, b[6], c[5]);
    madc_hi_cc(c[4], t, b[5], c[4]);
    madc_hi_cc(c[3], t, b[4], c[3]);
    madc_hi_cc(c[2], t, b[3], c[2]);
    madc_hi_cc(c[1], t, b[2], c[1]);
    madc_hi_cc(c[0], t, b[1], c[0]);
    madc_hi(high[7], t, b[0], high[7]);

    // a[6] * b (low)
    t = a[6];
    mad_lo_cc(c[6], t, b[7], c[6]);
    madc_lo_cc(c[5], t, b[6], c[5]);
    madc_lo_cc(c[4], t, b[5], c[4]);
    madc_lo_cc(c[3], t, b[4], c[3]);
    madc_lo_cc(c[2], t, b[3], c[2]);
    madc_lo_cc(c[1], t, b[2], c[1]);
    madc_lo_cc(c[0], t, b[1], c[0]);
    madc_lo_cc(high[7], t, b[0], high[7]);
    addc(high[6], high[6], 0);

    // a[6] * b (high)
    mad_hi_cc(c[5], t, b[7], c[5]);
    madc_hi_cc(c[4], t, b[6], c[4]);
    madc_hi_cc(c[3], t, b[5], c[3]);
    madc_hi_cc(c[2], t, b[4], c[2]);
    madc_hi_cc(c[1], t, b[3], c[1]);
    madc_hi_cc(c[0], t, b[2], c[0]);
    madc_hi_cc(high[7], t, b[1], high[7]);
    madc_hi(high[6], t, b[0], high[6]);

    // a[5] * b (low)
    t = a[5];
    mad_lo_cc(c[5], t, b[7], c[5]);
    madc_lo_cc(c[4], t, b[6], c[4]);
    madc_lo_cc(c[3], t, b[5], c[3]);
    madc_lo_cc(c[2], t, b[4], c[2]);
    madc_lo_cc(c[1], t, b[3], c[1]);
    madc_lo_cc(c[0], t, b[2], c[0]);
    madc_lo_cc(high[7], t, b[1], high[7]);
    madc_lo_cc(high[6], t, b[0], high[6]);
    addc(high[5], high[5], 0);

    // a[5] * b (high)
    mad_hi_cc(c[4], t, b[7], c[4]);
    madc_hi_cc(c[3], t, b[6], c[3]);
    madc_hi_cc(c[2], t, b[5], c[2]);
    madc_hi_cc(c[1], t, b[4], c[1]);
    madc_hi_cc(c[0], t, b[3], c[0]);
    madc_hi_cc(high[7], t, b[2], high[7]);
    madc_hi_cc(high[6], t, b[1], high[6]);
    madc_hi(high[5], t, b[0], high[5]);

    // a[4] * b (low)
    t = a[4];
    mad_lo_cc(c[4], t, b[7], c[4]);
    madc_lo_cc(c[3], t, b[6], c[3]);
    madc_lo_cc(c[2], t, b[5], c[2]);
    madc_lo_cc(c[1], t, b[4], c[1]);
    madc_lo_cc(c[0], t, b[3], c[0]);
    madc_lo_cc(high[7], t, b[2], high[7]);
    madc_lo_cc(high[6], t, b[1], high[6]);
    madc_lo_cc(high[5], t, b[0], high[5]);
    addc(high[4], high[4], 0);

    // a[4] * b (high)
    mad_hi_cc(c[3], t, b[7], c[3]);
    madc_hi_cc(c[2], t, b[6], c[2]);
    madc_hi_cc(c[1], t, b[5], c[1]);
    madc_hi_cc(c[0], t, b[4], c[0]);
    madc_hi_cc(high[7], t, b[3], high[7]);
    madc_hi_cc(high[6], t, b[2], high[6]);
    madc_hi_cc(high[5], t, b[1], high[5]);
    madc_hi(high[4], t, b[0], high[4]);

    // a[3] * b (low)
    t = a[3];
    mad_lo_cc(c[3], t, b[7], c[3]);
    madc_lo_cc(c[2], t, b[6], c[2]);
    madc_lo_cc(c[1], t, b[5], c[1]);
    madc_lo_cc(c[0], t, b[4], c[0]);
    madc_lo_cc(high[7], t, b[3], high[7]);
    madc_lo_cc(high[6], t, b[2], high[6]);
    madc_lo_cc(high[5], t, b[1], high[5]);
    madc_lo_cc(high[4], t, b[0], high[4]);
    addc(high[3], high[3], 0);

    // a[3] * b (high)
    mad_hi_cc(c[2], t, b[7], c[2]);
    madc_hi_cc(c[1], t, b[6], c[1]);
    madc_hi_cc(c[0], t, b[5], c[0]);
    madc_hi_cc(high[7], t, b[4], high[7]);
    madc_hi_cc(high[6], t, b[3], high[6]);
    madc_hi_cc(high[5], t, b[2], high[5]);
    madc_hi_cc(high[4], t, b[1], high[4]);
    madc_hi(high[3], t, b[0], high[3]);

    // a[2] * b (low)
    t = a[2];
    mad_lo_cc(c[2], t, b[7], c[2]);
    madc_lo_cc(c[1], t, b[6], c[1]);
    madc_lo_cc(c[0], t, b[5], c[0]);
    madc_lo_cc(high[7], t, b[4], high[7]);
    madc_lo_cc(high[6], t, b[3], high[6]);
    madc_lo_cc(high[5], t, b[2], high[5]);
    madc_lo_cc(high[4], t, b[1], high[4]);
    madc_lo_cc(high[3], t, b[0], high[3]);
    addc(high[2], high[2], 0);

    // a[2] * b (high)
    mad_hi_cc(c[1], t, b[7], c[1]);
    madc_hi_cc(c[0], t, b[6], c[0]);
    madc_hi_cc(high[7], t, b[5], high[7]);
    madc_hi_cc(high[6], t, b[4], high[6]);
    madc_hi_cc(high[5], t, b[3], high[5]);
    madc_hi_cc(high[4], t, b[2], high[4]);
    madc_hi_cc(high[3], t, b[1], high[3]);
    madc_hi(high[2], t, b[0], high[2]);

    // a[1] * b (low)
    t = a[1];
    mad_lo_cc(c[1], t, b[7], c[1]);
    madc_lo_cc(c[0], t, b[6], c[0]);
    madc_lo_cc(high[7], t, b[5], high[7]);
    madc_lo_cc(high[6], t, b[4], high[6]);
    madc_lo_cc(high[5], t, b[3], high[5]);
    madc_lo_cc(high[4], t, b[2], high[4]);
    madc_lo_cc(high[3], t, b[1], high[3]);
    madc_lo_cc(high[2], t, b[0], high[2]);
    addc(high[1], high[1], 0);

    // a[1] * b (high)
    mad_hi_cc(c[0], t, b[7], c[0]);
    madc_hi_cc(high[7], t, b[6], high[7]);
    madc_hi_cc(high[6], t, b[5], high[6]);
    madc_hi_cc(high[5], t, b[4], high[5]);
    madc_hi_cc(high[4], t, b[3], high[4]);
    madc_hi_cc(high[3], t, b[2], high[3]);
    madc_hi_cc(high[2], t, b[1], high[2]);
    madc_hi(high[1], t, b[0], high[1]);

    // a[0] * b (low)
    t = a[0];
    mad_lo_cc(c[0], t, b[7], c[0]);
    madc_lo_cc(high[7], t, b[6], high[7]);
    madc_lo_cc(high[6], t, b[5], high[6]);
    madc_lo_cc(high[5], t, b[4], high[5]);
    madc_lo_cc(high[4], t, b[3], high[4]);
    madc_lo_cc(high[3], t, b[2], high[3]);
    madc_lo_cc(high[2], t, b[1], high[2]);
    madc_lo_cc(high[1], t, b[0], high[1]);
    addc(high[0], high[0], 0);

    // a[0] * b (high)
    mad_hi_cc(high[7], t, b[7], high[7]);
    madc_hi_cc(high[6], t, b[6], high[6]);
    madc_hi_cc(high[5], t, b[5], high[5]);
    madc_hi_cc(high[4], t, b[4], high[4]);
    madc_hi_cc(high[3], t, b[3], high[3]);
    madc_hi_cc(high[2], t, b[2], high[2]);
    madc_hi_cc(high[1], t, b[1], high[1]);
    madc_hi(high[0], t, b[0], high[0]);

    // At this point we have 16 32-bit words representing a 512-bit value
    // high[0 ... 7] and c[0 ... 7]
    const unsigned int s = 977;

    // Store high[6] and high[7] since they will be overwritten
    unsigned int high7 = high[7];
    unsigned int high6 = high[6];

    // Take high 256 bits, multiply by 2^32, add to low 256 bits
    // That is, take high[0 ... 7], shift it left 1 word and add it to c[0 ... 7]
    add_cc(c[6], high[7], c[6]);
    addc_cc(c[5], high[6], c[5]);
    addc_cc(c[4], high[5], c[4]);
    addc_cc(c[3], high[4], c[3]);
    addc_cc(c[2], high[3], c[2]);
    addc_cc(c[1], high[2], c[1]);
    addc_cc(c[0], high[1], c[0]);
    addc_cc(high[7], high[0], 0);
    addc(high[6], 0, 0);

    // Take high 256 bits, multiply by 977, add to low 256 bits
    // That is, take high[0 ... 5], high6, high7, multiply by 977 and add to c[0 ... 7]
    mad_lo_cc(c[7], high7, s, c[7]);
    madc_lo_cc(c[6], high6, s, c[6]);
    madc_lo_cc(c[5], high[5], s, c[5]);
    madc_lo_cc(c[4], high[4], s, c[4]);
    madc_lo_cc(c[3], high[3], s, c[3]);
    madc_lo_cc(c[2], high[2], s, c[2]);
    madc_lo_cc(c[1], high[1], s, c[1]);
    madc_lo_cc(c[0], high[0], s, c[0]);
    addc_cc(high[7], high[7], 0);
    addc(high[6], high[6], 0);

    mad_hi_cc(c[6], high7, s, c[6]);
    madc_hi_cc(c[5], high6, s, c[5]);
    madc_hi_cc(c[4], high[5], s, c[4]);
    madc_hi_cc(c[3], high[4], s, c[3]);
    madc_hi_cc(c[2], high[3], s, c[2]);
    madc_hi_cc(c[1], high[2], s, c[1]);
    madc_hi_cc(c[0], high[1], s, c[0]);
    madc_hi_cc(high[7], high[0], s, high[7]);
    addc(high[6], high[6], 0);

    // Repeat the same steps, but this time we only need to handle high[6] and high[7]
    high7 = high[7];
    high6 = high[6];

    // Take the high 64 bits, multiply by 2^32 and add to the low 256 bits
    add_cc(c[6], high[7], c[6]);
    addc_cc(c[5], high[6], c[5]);
    addc_cc(c[4], c[4], 0);
    addc_cc(c[3], c[3], 0);
    addc_cc(c[2], c[2], 0);
    addc_cc(c[1], c[1], 0);
    addc_cc(c[0], c[0], 0);
    addc(high[7], 0, 0);

    // Take the high 64 bits, multiply by 977 and add to the low 256 bits
    mad_lo_cc(c[7], high7, s, c[7]);
    madc_lo_cc(c[6], high6, s, c[6]);
    addc_cc(c[5], c[5], 0);
    addc_cc(c[4], c[4], 0);
    addc_cc(c[3], c[3], 0);
    addc_cc(c[2], c[2], 0);
    addc_cc(c[1], c[1], 0);
    addc_cc(c[0], c[0], 0);
    addc(high[7], high[7], 0);

    mad_hi_cc(c[6], high7, s, c[6]);
    madc_hi_cc(c[5], high6, s, c[5]);
    addc_cc(c[4], c[4], 0);
    addc_cc(c[3], c[3], 0);
    addc_cc(c[2], c[2], 0);
    addc_cc(c[1], c[1], 0);
    addc_cc(c[0], c[0], 0);
    addc(high[7], high[7], 0);

    bool overflow = high[7] != 0;

    unsigned int borrow = sub(c, _P, c);

    if (overflow)
    {
        if (!borrow)
        {
            sub(c, _P, c);
        }
    }
    else
    {
        if (borrow)
        {
            add(c, _P, c);
        }
    }
}

/**
 * Square mod P
 * b = a * a
 */
__device__ static void squareModP(const unsigned int a[8], unsigned int b[8])
{
    mulModP(a, a, b);
}

/**
 * Square mod P
 * x = x * x
 */
__device__ static void squareModP(unsigned int x[8])
{
    unsigned int tmp[8];
    squareModP(x, tmp);
    copyBigInt(tmp, x);
}

/**
 * Multiply mod P
 * c = a * c
 */
__device__ static void mulModP(const unsigned int a[8], unsigned int c[8])
{
    unsigned int tmp[8];
    mulModP(a, c, tmp);

    copyBigInt(tmp, c);
}

/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p and addition chains
 */
__device__ static void invModP(unsigned int value[8])
{
    unsigned int x[8];

    copyBigInt(value, x);

    unsigned int y[8] = {0, 0, 0, 0, 0, 0, 0, 1};

    // 0xd - 1101
    mulModP(x, y);
    squareModP(x);
    // mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);

    // 0x2 - 0010
    // mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);
    // mulModP(x, y);
    squareModP(x);
    // mulModP(x, y);
    squareModP(x);

    // 0xc = 0x1100
    // mulModP(x, y);
    squareModP(x);
    // mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);

    // 0xfffff
    for (int i = 0; i < 20; i++)
    {
        mulModP(x, y);
        squareModP(x);
    }

    // 0xe - 1110
    // mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);
    mulModP(x, y);
    squareModP(x);

    // 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffff
    for (int i = 0; i < 219; i++)
    {
        mulModP(x, y);
        squareModP(x);
    }
    mulModP(x, y);

    copyBigInt(y, value);
}

__device__ static void invModP(const unsigned int *value, unsigned int *inverse)
{
    copyBigInt(value, inverse);

    invModP(inverse);
}

__device__ static void negModP(const unsigned int *value, unsigned int *negative)
{
    sub_cc(negative[0], _P[0], value[0]);
    subc_cc(negative[1], _P[1], value[1]);
    subc_cc(negative[2], _P[2], value[2]);
    subc_cc(negative[3], _P[3], value[3]);
    subc_cc(negative[4], _P[4], value[4]);
    subc_cc(negative[5], _P[5], value[5]);
    subc_cc(negative[6], _P[6], value[6]);
    subc(negative[7], _P[7], value[7]);
}

__device__ __forceinline__ static void beginBatchAdd(const unsigned int *px, const unsigned int *x, unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
    // x = Gx - x
    unsigned int t[8];
    subModP(px, x, t);

    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    mulModP(t, inverse);

    writeInt(chain, batchIdx, inverse);
}

__device__ __forceinline__ static void beginBatchAddWithDouble(const unsigned int *px, const unsigned int *py, unsigned int *xPtr, unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
    unsigned int x[8];
    readInt(xPtr, i, x);

    if (equal(px, x))
    {
        addModP(py, py, x);
    }
    else
    {
        // x = Gx - x
        subModP(px, x, x);
    }

    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    mulModP(x, inverse);

    writeInt(chain, batchIdx, inverse);
}

__device__ static void completeBatchAddWithDouble(const unsigned int *px, const unsigned int *py, const unsigned int *xPtr, const unsigned int *yPtr, int i, int batchIdx, unsigned int *chain, unsigned int *inverse, unsigned int newX[8], unsigned int newY[8])
{
    unsigned int s[8];
    unsigned int x[8];
    unsigned int y[8];

    readInt(xPtr, i, x);
    readInt(yPtr, i, y);

    if (batchIdx >= 1)
    {
        unsigned int c[8];

        readInt(chain, batchIdx - 1, c);

        mulModP(inverse, c, s);

        unsigned int diff[8];
        if (equal(px, x))
        {
            addModP(py, py, diff);
        }
        else
        {
            subModP(px, x, diff);
        }

        mulModP(diff, inverse);
    }
    else
    {
        copyBigInt(inverse, s);
    }

    if (equal(px, x))
    {
        // currently s = 1 / 2y

        unsigned int x2[8];
        unsigned int tx2[8];

        // 3x^2
        mulModP(x, x, x2);
        addModP(x2, x2, tx2);
        addModP(x2, tx2, tx2);

        // s = 3x^2 * 1/2y
        mulModP(tx2, s);

        // s^2
        unsigned int s2[8];
        mulModP(s, s, s2);

        // Rx = s^2 - 2px
        subModP(s2, x, newX);
        subModP(newX, x, newX);

        // Ry = s(px - rx) - py
        unsigned int k[8];
        subModP(px, newX, k);
        mulModP(s, k, newY);
        subModP(newY, py, newY);
    }
    else
    {

        unsigned int rise[8];
        subModP(py, y, rise);

        mulModP(rise, s);

        // Rx = s^2 - Gx - Qx
        unsigned int s2[8];
        mulModP(s, s, s2);

        subModP(s2, px, newX);
        subModP(newX, x, newX);

        // Ry = s(px - rx) - py
        unsigned int k[8];
        subModP(px, newX, k);
        mulModP(s, k, newY);
        subModP(newY, py, newY);
    }
}

__device__ static void completeBatchAdd(const unsigned int *px, const unsigned int *py, unsigned int *xPtr, unsigned int *yPtr, int i, int batchIdx, unsigned int *chain, unsigned int *inverse, unsigned int newX[8], unsigned int newY[8])
{
    unsigned int s[8];
    unsigned int x[8];

    readInt(xPtr, i, x);

    if (batchIdx >= 1)
    {
        unsigned int c[8];

        readInt(chain, batchIdx - 1, c);
        mulModP(inverse, c, s);

        unsigned int diff[8];
        subModP(px, x, diff);
        mulModP(diff, inverse);
    }
    else
    {
        copyBigInt(inverse, s);
    }

    unsigned int y[8];
    readInt(yPtr, i, y);

    unsigned int rise[8];
    subModP(py, y, rise);

    mulModP(rise, s);

    // Rx = s^2 - Gx - Qx
    unsigned int s2[8];
    mulModP(s, s, s2);
    subModP(s2, px, newX);
    subModP(newX, x, newX);

    // Ry = s(px - rx) - py
    unsigned int k[8];
    subModP(px, newX, k);
    mulModP(s, k, newY);
    subModP(newY, py, newY);
}

__device__ __forceinline__ static void doBatchInverse(unsigned int inverse[8])
{
    invModP(inverse);
}

/**
 * @brief Sets a point representation to the point at infinity.
 * In this implementation, infinity is represented by all words being 0xFFFFFFFF.
 * @param P The point coordinate array (X or Y) to set to infinity.
 */
__device__ __forceinline__ void set_infinity(unsigned int P[8])
{
#pragma unroll
    for (int i = 0; i < 8; ++i)
        P[i] = 0xffffffff;
}

/**
 * @brief Copies the coordinates of one point to another.
 * @param srcX X-coordinate of the source point.
 * @param srcY Y-coordinate of the source point.
 * @param destX X-coordinate of the destination point.
 * @param destY Y-coordinate of the destination point.
 */
__device__ static void copy_point(const unsigned int srcX[8], const unsigned int srcY[8], unsigned int destX[8], unsigned int destY[8])
{
    copyBigInt(srcX, destX);
    copyBigInt(srcY, destY);
}

/**
 * @brief Doubles a point on the secp256k1 curve using affine coordinates.
 * Calculates R = 2P, where P = (Px, Py) and R = (Rx, Ry).
 * Handles the point at infinity and points where Py = 0.
 * @param Px X-coordinate of the point P.
 * @param Py Y-coordinate of the point P.
 * @param Rx Output X-coordinate of the result point R.
 * @param Ry Output Y-coordinate of the result point R.
 */
__device__ static void point_double(const unsigned int Px[8], const unsigned int Py[8], unsigned int Rx[8], unsigned int Ry[8])
{
    // If P is infinity, 2*Infinity = Infinity
    if (isInfinity(Px))
    {
        set_infinity(Rx);
        set_infinity(Ry);
        return;
    }

    // Check if Py == 0. If so, 2P = Infinity.
    bool y_is_zero = true;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        if (Py[i] != 0)
        {
            y_is_zero = false;
            break;
        }
    }
    if (y_is_zero)
    {
        set_infinity(Rx);
        set_infinity(Ry);
        return;
    }

    unsigned int s[8];
    unsigned int two_y[8];
    unsigned int inv_two_y[8];
    unsigned int three_x_sq[8];
    unsigned int x_sq[8];

    // Calculate inv_two_y = (2 * Py)^(-1) mod P
    addModP(Py, Py, two_y);    // 2 * Py
    invModP(two_y, inv_two_y); // (2 * Py)^-1

    // Calculate 3 * Px^2 mod P
    squareModP(Px, x_sq);                  // x^2
    addModP(x_sq, x_sq, three_x_sq);       // 2*x^2
    addModP(x_sq, three_x_sq, three_x_sq); // 3*x^2

    // Calculate slope s = (3 * Px^2) * inv_two_y mod P
    mulModP(three_x_sq, inv_two_y, s);

    // Calculate Rx = s^2 - 2 * Px mod P
    unsigned int s_sq[8];
    unsigned int two_px[8];
    squareModP(s, s_sq);       // s^2
    addModP(Px, Px, two_px);   // 2*Px
    subModP(s_sq, two_px, Rx); // Rx = s^2 - 2*Px

    // Calculate Ry = s * (Px - Rx) - Py mod P
    unsigned int px_minus_rx[8];
    subModP(Px, Rx, px_minus_rx); // Px - Rx
    mulModP(s, px_minus_rx, Ry);  // s * (Px - Rx)
    subModP(Ry, Py, Ry);          // Ry = s(Px - Rx) - Py
}

/**
 * @brief Adds two points on the secp256k1 curve using affine coordinates.
 * Calculates R = P1 + P2, where P1 = (P1x, P1y), P2 = (P2x, P2y), and R = (Rx, Ry).
 * Handles cases involving the point at infinity, P1 = P2, and P1 = -P2.
 * @param P1x X-coordinate of point P1.
 * @param P1y Y-coordinate of point P1.
 * @param P2x X-coordinate of point P2.
 * @param P2y Y-coordinate of point P2.
 * @param Rx Output X-coordinate of the result point R.
 * @param Ry Output Y-coordinate of the result point R.
 */
__device__ static void point_add(const unsigned int P1x[8], const unsigned int P1y[8],
                                 const unsigned int P2x[8], const unsigned int P2y[8],
                                 unsigned int Rx[8], unsigned int Ry[8])
{
    // Handle infinity cases: P + Infinity = P
    if (isInfinity(P1x))
    {
        copy_point(P2x, P2y, Rx, Ry);
        return;
    }
    if (isInfinity(P2x))
    {
        copy_point(P1x, P1y, Rx, Ry);
        return;
    }

    // Check for P1 == P2, use point doubling
    if (equal(P1x, P2x) && equal(P1y, P2y))
    {
        point_double(P1x, P1y, Rx, Ry);
        return;
    }

    // Check for P1 == -P2 (i.e., P1x == P2x and P1y + P2y == 0 mod P)
    unsigned int y1_plus_y2[8];
    addModP(P1y, P2y, y1_plus_y2);
    bool sum_y_is_zero = true;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        if (y1_plus_y2[i] != 0)
        {
            sum_y_is_zero = false;
            break;
        }
    }

    if (equal(P1x, P2x) && sum_y_is_zero)
    {
        // P1 + (-P1) = Infinity
        set_infinity(Rx);
        set_infinity(Ry);
        return;
    }

    // Calculate slope s = (P2y - P1y) / (P2x - P1x) mod P
    unsigned int s[8];
    unsigned int delta_y[8];
    unsigned int delta_x[8];
    unsigned int inv_delta_x[8];

    subModP(P2y, P1y, delta_y); // P2y - P1y
    subModP(P2x, P1x, delta_x); // P2x - P1x

    // Calculate inverse of delta_x. Should not be zero if P1 != P2 and P1 != -P2.
    invModP(delta_x, inv_delta_x);

    // Calculate slope s = delta_y * inv_delta_x mod P
    mulModP(delta_y, inv_delta_x, s);

    // Calculate Rx = s^2 - P1x - P2x mod P
    unsigned int s_sq[8];
    squareModP(s, s_sq);    // s^2
    subModP(s_sq, P1x, Rx); // s^2 - P1x
    subModP(Rx, P2x, Rx);   // Rx = s^2 - P1x - P2x

    // Calculate Ry = s * (P1x - Rx) - P1y mod P
    unsigned int p1x_minus_rx[8];
    subModP(P1x, Rx, p1x_minus_rx); // P1x - Rx
    mulModP(s, p1x_minus_rx, Ry);   // s * (P1x - Rx)
    subModP(Ry, P1y, Ry);           // Ry = s(P1x - Rx) - P1y
}

/**
 * Computes the public key (Px, Py) for a given private key k.
 * PublicKey = k * G, where G is the base point (_GX, _GY).
 * Uses the double-and-add algorithm. Assumes k is in little-endian word order.
 *
 * @param k  Input private key scalar (8 x 32-bit words, little-endian word order: k[7] is LSW).
 * @param Px Output public key X coordinate (8 x 32-bit words).
 * @param Py Output public key Y coordinate (8 x 32-bit words).
 */
__device__ static void compute_public_key(const unsigned int k[8], unsigned int Px[8], unsigned int Py[8])
{
    unsigned int tempX[8], tempY[8]; // Holds the current multiple of G (starts at G)
    copy_point(_GX, _GY, tempX, tempY);

    set_infinity(Px); // Initialize result point R to infinity
    set_infinity(Py);

// Double-and-add algorithm: iterates through bits of k from LSB to MSB.
#pragma unroll
    for (int i = 7; i >= 0; --i)
    { // Iterate through words (k[7] is LSW, k[0] is MSW)
        unsigned int word = k[i];
        for (int j = 0; j < 32; ++j)
        { // Iterate through bits within the word (LSB to MSB)
            if (word & 1)
            { // Check the least significant bit
                // If bit is 1, add the current power-of-2 multiple of G to the result
                // R = R + tempP
                point_add(Px, Py, tempX, tempY, Px, Py);
            }
            // Double the current multiple of G for the next bit position
            // tempP = 2 * tempP
            point_double(tempX, tempY, tempX, tempY);
            word >>= 1; // Move to the next bit in the word
        }
    }
    // The final result (Px, Py) holds k*G
}

/**
 * @brief Checks if a 256-bit scalar is zero.
 * @param a The scalar (8 x 32-bit words).
 * @return True if the scalar is zero, false otherwise.
 */
__device__ __forceinline__ bool isZero(const unsigned int a[8])
{
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        if (a[i] != 0)
            return false;
    }
    return true;
}

/**
 * @brief Adds two 256-bit scalars modulo the group order N.
 * c = (a + b) mod N
 * @param a First scalar operand.
 * @param b Second scalar operand.
 * @param c Result scalar.
 */
__device__ static void addModN(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    unsigned int carry = add(a, b, c);

    // Check if result >= N. This is done by trying to subtract N.
    // If subtraction underflows (borrow=1), the result was < N.
    // If subtraction does not underflow (borrow=0), the result was >= N, and we keep the subtracted value.

    unsigned int temp[8];
    unsigned int borrow = sub(c, _N, temp);

    // If no borrow occurred (carry=0 from sub) OR original addition had a carry,
    // it means (a + b) >= N. Use the result of subtraction.
    if (carry || !borrow)
    {
        copyBigInt(temp, c);
    }
    // Otherwise, (a + b) < N, and c already holds the correct value.
}

/**
 * @brief Subtracts two 256-bit scalars modulo the group order N.
 * c = (a - b) mod N
 * @param a First scalar operand.
 * @param b Second scalar operand.
 * @param c Result scalar.
 */
__device__ static void subModN(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    unsigned int borrow = sub(a, b, c);

    // If borrow occurred, it means a < b. We need to add N to the result.
    // c = (a - b + N) mod N = (a - b) + N
    if (borrow)
    {
        add(c, _N, c); // Add N, ignore carry as (a-b)+N must be < N
    }
}

/**
 * @brief Adds a tweak to a scalar modulo the group order N, in place.
 * scalar = (scalar + tweak) mod N
 * Checks if the result would be zero, which is considered an invalid tweak.
 *
 * @param scalar Input/Output scalar (8 x 32-bit words).
 * @param tweak Input tweak scalar (8 x 32-bit words).
 * @return 1 if the operation was successful, 0 if the result would be zero (tweak out of range).
 */
__device__ static int scalar_tweak_add_assign(unsigned int scalar[8], const unsigned int tweak[8])
{
    unsigned int result[8];
    addModN(scalar, tweak, result);

    if (isZero(result))
    {
        return 0; // Tweak results in zero, which is invalid.
    }

    copyBigInt(result, scalar);
    return 1; // Success
}
