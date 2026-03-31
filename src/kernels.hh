#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h> // UNTESTED on ARM
#else
#include <immintrin.h>
#endif

// Scalar stats kernel: computes running stats over all blocks.
// Returns a mini hash (no quality requirements) to prevent elision.
inline uint64_t kernel_scalar_stats(std::span<std::span<float const> const> data)
{
    struct stats
    {
        float m0 = 0, m1 = 0, m2 = 0;
        float min = +std::numeric_limits<float>::max();
        float max = -std::numeric_limits<float>::max();
    };

    stats s;
    for (auto block : data)
        for (auto d : block)
        {
            s.m0 += 1;
            s.m1 += d;
            s.m2 += d * d;
            if (d < s.min)
                s.min = d;
            if (d > s.max)
                s.max = d;
        }

    auto b = [](float f) { return std::bit_cast<uint32_t>(f); };
    return b(s.m0) ^ b(s.m1) ^ b(s.m2) ^ b(s.min) ^ b(s.max);
}

// SIMD sum kernel: uses 256-bit SIMD intrinsics (AVX2 on x86, NEON on ARM).
// Assumes each inner block size is a multiple of 8 AND 32-byte aligned.
inline uint64_t kernel_simd_sum(std::span<std::span<float const> const> data)
{
#if defined(__AVX2__) || defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
    __m256 acc = _mm256_setzero_ps();
    for (auto block : data)
    {
        auto const* ptr = block.data();
        auto const count = block.size();
        for (size_t i = 0; i < count; i += 8)
            acc = _mm256_add_ps(acc, _mm256_load_ps(ptr + i));
    }
    // reduce: bitcast to uint32 array and XOR
    auto words = std::bit_cast<std::array<uint32_t, 8>>(acc);
    uint32_t h = 0;
    for (auto w : words)
        h ^= w;
    return uint64_t(h);

#elif defined(__ARM_NEON) || defined(__ARM_NEON__) // UNTESTED
    float32x4_t acc_lo = vdupq_n_f32(0.f);
    float32x4_t acc_hi = vdupq_n_f32(0.f);
    for (auto block : data)
    {
        auto const* ptr = block.data();
        auto const count = block.size();
        for (size_t i = 0; i < count; i += 8)
        {
            acc_lo = vaddq_f32(acc_lo, vld1q_f32(ptr + i));
            acc_hi = vaddq_f32(acc_hi, vld1q_f32(ptr + i + 4));
        }
    }
    auto lo = std::bit_cast<std::array<uint32_t, 4>>(acc_lo);
    auto hi = std::bit_cast<std::array<uint32_t, 4>>(acc_hi);
    uint32_t h = 0;
    for (auto w : lo)
        h ^= w;
    for (auto w : hi)
        h ^= w;
    return uint64_t(h);
#else
#error "Need at least AVX2 or NEON support"
#endif
}

// Heavy compute kernel: accumulates v = sin(v + data[i]) over all data.
inline uint64_t kernel_heavy_sin(std::span<std::span<float const> const> data)
{
    float v = 0.f;
    for (auto block : data)
        for (auto d : block)
            v = std::sin(v + d);
    return uint64_t(std::bit_cast<uint32_t>(v));
}
