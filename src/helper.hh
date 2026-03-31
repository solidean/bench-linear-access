#pragma once

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

struct timer
{
    // according to cppreference, "[steady_clock] is most suitable for measuring intervals"
    using clock = std::chrono::steady_clock;

    clock::time_point start = clock::now();

    [[nodiscard]] double elapsed_secs() const { return std::chrono::duration<double>(clock::now() - start).count(); }
};

inline float const* align_up_32(float const* p)
{
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    addr = (addr + 31u) & ~std::uintptr_t(31u);
    return reinterpret_cast<float const*>(addr);
}

// folds a 256 MB block of memory into the seed and returns a dependent result
inline uint64_t clobber_cache(uint64_t seed)
{
    static std::vector<uint64_t> clobber_data;

    if (clobber_data.empty())
    {
        clobber_data.resize((256uLL << 20) / sizeof(uint64_t));

        std::default_random_engine rng;
        for (auto& d : clobber_data)
            d = rng();
    }

    // slightly dependency heavy but very hard to optimize away
    for (auto d : clobber_data)
        seed = seed * d + 7;

    return seed;
}
