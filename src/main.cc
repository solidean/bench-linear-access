#include "helper.hh"

#include <algorithm>
#include <limits>
#include <print>
#include <random>
#include <span>
#include <vector>

struct stats
{
    float m0 = 0;
    float m1 = 0;
    float m2 = 0;
    float min = +std::numeric_limits<float>::max();
    float max = -std::numeric_limits<float>::max();

    static stats from_value(float d) { return {1, d, d * d, d, d}; }

    void add(stats const& s)
    {
        m0 += s.m0;
        m1 += s.m1;
        m2 += s.m2;
        min = std::min(min, s.min);
        max = std::max(max, s.max);
    }
};

stats compute_block_stats(std::span<float const> data)
{
    stats s;
    for (auto d : data)
        s.add(stats::from_value(d));
    return s;
}

stats compute_full_stats(std::span<std::span<float const> const> data)
{
    stats s;
    for (auto block : data)
        s.add(compute_block_stats(block));
    return s;
}

struct experiment_config
{
    size_t total_memory_size;
    size_t min_block_size = 1;
    size_t runs = 9;
    size_t min_total_memory = 1uLL << 30;
    bool randomize_runs = true;
};

struct experiment_result
{
    double secs_data_prepare = -1;

    std::vector<stats> result_stats;

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
    experiment_result res;

    auto const total_memory = std::max(cfg.min_total_memory, cfg.total_memory_size * cfg.runs);
    auto const total_size = total_memory / sizeof(float);
    std::println("total_memory = {}", total_memory);
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
    for (size_t block_size = cfg.min_block_size; block_size <= cfg.total_memory_size / sizeof(float); block_size *= 2)
    {
        auto const total_block_count = total_size / block_size;
        std::println("  total_block_count = {}", total_block_count);

        // prepare randomized blocks
        blocks.resize(total_block_count);
        for (size_t i = 0; i < total_block_count; ++i)
            blocks[i] = {data.data() + i * block_size, block_size};
        std::shuffle(blocks.begin(), blocks.end(), rng);

        auto const block_size_bytes = block_size * sizeof(float);
        auto const blocks_per_run = cfg.total_memory_size / block_size_bytes;
        std::println("  bs = {} x {} per run (= {} B):", block_size_bytes, blocks_per_run,
                     block_size_bytes * blocks_per_run);

        // actually do the runs
        tmp_timing.clear();
        for (auto r = 0; r < (int)cfg.runs; ++r)
        {
            auto const block_span = std::span<std::span<float const> const>(
                blocks.data() + r * blocks_per_run * cfg.randomize_runs, blocks_per_run);
            timer t;
            auto const s = compute_full_stats(block_span);
            auto const secs = t.elapsed_secs();
            res.result_stats.push_back(s);

            tmp_timing.push_back(secs);

            std::println("    {} secs ({} MB/s)", secs, int(block_size_bytes * blocks_per_run / secs / 1024. / 1024.));
        }

        std::sort(tmp_timing.begin(), tmp_timing.end());
        auto const secs = tmp_timing[tmp_timing.size() / 2];

        res.block_results.push_back({
            .block_size_bytes = block_size_bytes,
            .total_size_bytes = block_size_bytes * blocks_per_run,
            .secs = secs,
        });
    }

    return res;
}

int main()
{
    auto const res = run_experiment({
        .total_memory_size = 16 * 1024uLL << 10,
        .min_block_size = 4,
        .randomize_runs = false,
    });

    // ensure total stats are never elided
    stats total_stats;
    for (auto const& s : res.result_stats)
        total_stats.add(s);
    std::println("stats.m0  = {}", total_stats.m0);
    std::println("stats.m1  = {}", total_stats.m1);
    std::println("stats.m2  = {}", total_stats.m2);
    std::println("stats.min = {}", total_stats.min);
    std::println("stats.max = {}", total_stats.max);
    std::println("");

    // results
    for (auto const& br : res.block_results)
        std::println("{:4} MB/s for {:7} B blocks", int(br.total_size_bytes / br.secs / 1024. / 1024.),
                     br.block_size_bytes);
}
