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

    /// largest block size in bytes to test (will always be capped by working_set_bytes)
    size_t block_bytes_max = 2uLL << 20;

    /// number of runs per block size (default 9, median is reported)
    size_t runs = 7;

    /// size of the backing allocation that blocks are drawn from;
    /// must satisfy: working_set_bytes * runs <= backing_memory_bytes
    size_t backing_memory_bytes = 4uLL << 30;

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

// create a large amount of randomly initialized backing memory
std::vector<float> make_backing_memory(std::default_random_engine& rng, experiment_config const& cfg, experiment_result& res)
{
    std::vector<float> data;

    timer t;

    data.resize(cfg.backing_memory_bytes / sizeof(float));
    for (auto& d : data)
        d = rng() / float(rng.max()) * 2 - 1; // we don't care about quality

    res.secs_data_prepare = t.elapsed_secs();

    std::println("created {:.1} GB backing memory in {:.2} sec",   //
                 cfg.backing_memory_bytes / 1024. / 1024. / 1024., //
                 res.secs_data_prepare);
    std::fflush(stdout);

    return data;
}

// makes enough random blocks to execute cfg.runs many times
void make_random_blocks(std::default_random_engine& rng,
                        std::vector<std::span<float const>>& blocks,
                        std::span<float const> backing_data,
                        experiment_config const& cfg,
                        size_t block_bytes)
{
    auto const floats_per_block = block_bytes / sizeof(float);
    auto const blocks_per_run = cfg.working_set_bytes / block_bytes;
    auto const block_count = cfg.randomize_runs ? cfg.runs * blocks_per_run : blocks_per_run;

    blocks.resize(block_count);

    for (size_t bi = 0; bi < block_count; ++bi)
    {
        auto const p_start = align_up_32(backing_data.data() + backing_data.size() * (bi + 0) / block_count);
        auto const p_end = backing_data.data() + backing_data.size() * (bi + 1) / block_count;
        assert(p_start >= backing_data.data());
        assert(p_end <= backing_data.data() + backing_data.size());

        auto const available_floats = p_end - p_start;
        assert(available_floats >= floats_per_block);

        auto const offset_floats = available_floats - floats_per_block;
        auto const offset_floats32 = offset_floats / (32 / sizeof(float));

        auto const random_offset = std::uniform_int_distribution<size_t>(0, offset_floats32)(rng);

        blocks[bi] = {p_start + random_offset * (32 / sizeof(float)), floats_per_block};

        assert(p_start <= blocks[bi].data());
        assert(blocks[bi].data() + floats_per_block <= p_end);
        assert(blocks[bi].size() % 8 == 0);
        assert((reinterpret_cast<uintptr_t>(blocks[bi].data()) & 31) == 0);
    }

    // shuffle blocks
    std::shuffle(blocks.begin(), blocks.end(), rng);
}

experiment_result run_experiment(std::string_view kernel_name, experiment_config cfg)
{
    assert(cfg.kernel != nullptr);
    assert(std::has_single_bit(cfg.working_set_bytes));
    assert(std::has_single_bit(cfg.block_bytes_base));
    assert(cfg.block_bytes_base >= 32);
    assert(cfg.working_set_bytes * cfg.runs <= cfg.backing_memory_bytes);

    experiment_result res;

    std::default_random_engine rng;
    rng.seed(12345);

    auto const data = make_backing_memory(rng, cfg, res);

    std::vector<double> tmp_timing;
    std::vector<std::span<float const>> blocks;

    // run each block size
    std::println("experiment per blocksize (bs):");
    for (size_t block_bytes = cfg.block_bytes_base;
         block_bytes <= cfg.working_set_bytes && block_bytes <= cfg.block_bytes_max; block_bytes *= 2)
    {
        make_random_blocks(rng, blocks, data, cfg, block_bytes);

        // for non-randomized runs, we want to clobber once at the beginning
        if (!cfg.randomize_runs)
            res.result_hash = clobber_cache(res.result_hash);

        // actually do the runs
        auto const blocks_per_run = cfg.working_set_bytes / block_bytes;
        tmp_timing.clear();
        std::println("  .. {} runs for {} x {} B blocks:", cfg.runs, blocks_per_run, block_bytes);
        for (auto r = 0; r < (int)cfg.runs; ++r)
        {
            auto const block_span = std::span<std::span<float const> const>(
                blocks.data() + r * blocks_per_run * cfg.randomize_runs, blocks_per_run);

            // for randomized runs, we want to clobber each run
            if (cfg.randomize_runs)
                res.result_hash = clobber_cache(res.result_hash);

            timer t;
            auto const h = cfg.kernel(block_span);
            auto const secs = t.elapsed_secs();
            res.result_hash ^= h;

            tmp_timing.push_back(secs);

            std::println("    {} secs ({} MB/s)", secs, int(block_bytes * blocks_per_run / secs / 1024. / 1024.));
            std::fflush(stdout);
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
    auto const res = run_experiment( //
        "scalar_stats",              //
        {
            .kernel = kernel_scalar_stats,
            // .kernel = kernel_simd_sum,
            // .kernel = kernel_heavy,
            .working_set_bytes = 64 * 1024uLL << 10,
            .runs = 17,
            // .working_set_bytes = 2 * 1024uLL << 10, // DEBUG
            // .backing_memory_bytes = 64uLL << 20,    // DEBUG
            .randomize_runs = true,
        });

    // ensure results are never elided
    std::println("hash = {:016x}", res.result_hash);
    std::println("");

    // results
    for (auto const& br : res.block_results)
        std::println("{:5} MB/s for {:8} B blocks", int(br.total_size_bytes / br.secs / 1024. / 1024.),
                     br.block_size_bytes);
}
