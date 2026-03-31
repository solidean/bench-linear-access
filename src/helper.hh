#pragma once

#include <chrono>

struct timer
{
    // according to cppreference, "[steady_clock] is most suitable for measuring intervals"
    using clock = std::chrono::steady_clock;

    clock::time_point start = clock::now();

    [[nodiscard]] double elapsed_secs() const { return std::chrono::duration<double>(clock::now() - start).count(); }
};
