#include "helper.hh"
#include "kernels.hh"

#include <algorithm>
#include <bit>
#include <cassert>
#include <print>
#include <random>
#include <span>
#include <vector>

struct experiment_config
{
    using kernel_fun = uint64_t (*)(std::span<std::span<float const> const>);

    /// kernel to benchmark; called once per run with the block spans
    /// each block is aligned to 32 bytes
    kernel_fun kernel = nullptr;

    /// total bytes the kernel touches per run; must be a power of two
    size_t working_set_bytes;

    /// smallest block size in bytes to test; must be a power of two and >= 32
    size_t block_bytes_base = 32;

    /// number of runs per block size (default 9, median is reported)
    size_t runs = 9;

    /// size of the backing allocation that blocks are drawn from;
    /// must satisfy: working_set_bytes * runs <= backing_memory_bytes
    size_t backing_memory_bytes = 8uLL << 30;

    bool randomize_runs = true;
};

struct experiment_result
{
    double secs_data_prepare = -1;

    uint64_t result_hash = 0;

    struct block_result
    {
        size_t block_size_bytes;
        size_t total_size_bytes;
        double secs;
    };

    std::vector<block_result> block_results;
};

experiment_result run_experiment(experiment_config cfg)
{
    assert(cfg.kernel != nullptr);
    assert(std::has_single_bit(cfg.working_set_bytes));
    assert(std::has_single_bit(cfg.block_bytes_base));
    assert(cfg.block_bytes_base >= 32);
    assert(cfg.working_set_bytes * cfg.runs <= cfg.backing_memory_bytes);

    experiment_result res;

    auto const total_size = cfg.backing_memory_bytes / sizeof(float);
    std::println("backing_memory_bytes = {}", cfg.backing_memory_bytes);
    std::println("total_size = {}", total_size);

    std::vector<float> data;
    std::vector<std::span<float const>> blocks;

    std::default_random_engine rng;
    rng.seed(12345);

    // prepare data
    {
        timer t;

        data.resize(total_size);
        blocks.reserve(total_size);
        auto dis = std::uniform_real_distribution<float>(-1.f, 1.f);
        for (auto& d : data)
            d = dis(rng);

        res.secs_data_prepare = t.elapsed_secs();
    }

    std::vector<double> tmp_timing;

    // run each block size
    std::println("experiment per blocksize (bs):");
    for (size_t block_bytes = cfg.block_bytes_base; block_bytes <= cfg.working_set_bytes; block_bytes *= 2)
    {
        auto const block_size = block_bytes / sizeof(float);
        auto const total_block_count = total_size / block_size;
        std::println("  total_block_count = {}", total_block_count);

        // prepare randomized blocks
        blocks.resize(total_block_count);
        for (size_t i = 0; i < total_block_count; ++i)
            blocks[i] = {data.data() + i * block_size, block_size};
        std::shuffle(blocks.begin(), blocks.end(), rng);

        auto const blocks_per_run = cfg.working_set_bytes / block_bytes;
        std::println("  bs = {} x {} per run (= {} B):", block_bytes, blocks_per_run, block_bytes * blocks_per_run);

        // actually do the runs
        tmp_timing.clear();
        for (auto r = 0; r < (int)cfg.runs; ++r)
        {
            auto const block_span = std::span<std::span<float const> const>(
                blocks.data() + r * blocks_per_run * cfg.randomize_runs, blocks_per_run);
            timer t;
            auto const h = cfg.kernel(block_span);
            auto const secs = t.elapsed_secs();
            res.result_hash ^= h;

            tmp_timing.push_back(secs);

            std::println("    {} secs ({} MB/s)", secs, int(block_bytes * blocks_per_run / secs / 1024. / 1024.));
        }

        // report median time
        std::sort(tmp_timing.begin(), tmp_timing.end());
        auto const secs = tmp_timing[tmp_timing.size() / 2];

        res.block_results.push_back({
            .block_size_bytes = block_bytes,
            .total_size_bytes = block_bytes * blocks_per_run,
            .secs = secs,
        });
    }

    return res;
}

int main()
{
    auto const res = run_experiment({
        .kernel = kernel_scalar_stats,
        .working_set_bytes = 16 * 1024uLL << 10,
        .randomize_runs = false,
    });

    // ensure results are never elided
    std::println("hash = {:016x}", res.result_hash);
    std::println("");

    // results
    for (auto const& br : res.block_results)
        std::println("{:4} MB/s for {:8} B blocks", int(br.total_size_bytes / br.secs / 1024. / 1024.),
                     br.block_size_bytes);
}
