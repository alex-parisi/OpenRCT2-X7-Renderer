/// test_thread_pool.cpp

#include "ThreadPool.hpp"
#include <algorithm>
#include <atomic>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace RCTGen;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {
    // Build a pool with a small fixed size to keep tests deterministic.
    constexpr unsigned int kPoolSize = 4;
} // namespace

// ---------------------------------------------------------------------------
// Basic correctness
// ---------------------------------------------------------------------------

TEST(ThreadPool, RunZeroCountIsNoop) {
    ThreadPool pool(kPoolSize);
    std::atomic<int> calls{0};
    pool.run(0, [&](int) { ++calls; });
    EXPECT_EQ(calls.load(), 0);
}

TEST(ThreadPool, RunNegativeCountIsNoop) {
    ThreadPool pool(kPoolSize);
    std::atomic<int> calls{0};
    pool.run(-1, [&](int) { ++calls; });
    EXPECT_EQ(calls.load(), 0);
}

TEST(ThreadPool, RunSingleItem) {
    ThreadPool pool(kPoolSize);
    std::atomic<int> seen{-1};
    pool.run(1, [&](int i) { seen.store(i); });
    EXPECT_EQ(seen.load(), 0);
}

TEST(ThreadPool, RunExactlyOnePerWorker) {
    // count == pool size: each worker picks up exactly one item.
    ThreadPool pool(kPoolSize);
    std::atomic<int> calls{0};
    pool.run(static_cast<int>(kPoolSize), [&](int) { ++calls; });
    EXPECT_EQ(calls.load(), static_cast<int>(kPoolSize));
}

TEST(ThreadPool, RunFewerThanWorkers) {
    // count < pool size: not all workers get a work unit, but all items execute.
    ThreadPool pool(kPoolSize);
    std::atomic<int> calls{0};
    pool.run(2, [&](int) { ++calls; });
    EXPECT_EQ(calls.load(), 2);
}

TEST(ThreadPool, RunManyMoreThanWorkers) {
    // Each work unit runs exactly once even when count >> pool size.
    constexpr int kCount = 1000;
    ThreadPool pool(kPoolSize);
    std::vector<std::atomic<int>> flags(kCount);
    for (auto& f : flags) f.store(0);

    pool.run(kCount, [&](int i) { flags[i].fetch_add(1); });

    for (int i = 0; i < kCount; ++i)
        EXPECT_EQ(flags[i].load(), 1) << "item " << i << " executed " << flags[i].load() << " times";
}

TEST(ThreadPool, RunCoversAllIndices) {
    // Verify exactly the indices [0, count) are visited.
    constexpr int kCount = 64;
    ThreadPool pool(kPoolSize);
    std::vector<std::atomic<bool>> seen(kCount);
    for (auto& s : seen) s.store(false);

    pool.run(kCount, [&](int i) { seen[i].store(true); });

    for (int i = 0; i < kCount; ++i)
        EXPECT_TRUE(seen[i].load()) << "index " << i << " was never visited";
}

TEST(ThreadPool, SequentialRunsProduceCorrectSum) {
    // Two successive run() calls must each complete fully before the next begins.
    ThreadPool pool(kPoolSize);
    constexpr int kN = 100;
    std::atomic<long long> sum{0};

    pool.run(kN, [&](int i) { sum.fetch_add(i, std::memory_order_relaxed); });
    const long long expected = static_cast<long long>(kN) * (kN - 1) / 2;
    EXPECT_EQ(sum.load(), expected);

    // Second run, different operation.
    std::atomic<int> count{0};
    pool.run(kN, [&](int) { count.fetch_add(1, std::memory_order_relaxed); });
    EXPECT_EQ(count.load(), kN);
}

TEST(ThreadPool, MultipleSequentialRunsAreIsolated) {
    // Each run() must see only its own accumulated results.
    ThreadPool pool(kPoolSize);
    for (int round = 0; round < 5; ++round) {
        std::atomic<int> calls{0};
        pool.run(10, [&](int) { ++calls; });
        EXPECT_EQ(calls.load(), 10) << "round " << round;
    }
}

TEST(ThreadPool, SingleThreadedPool) {
    // A pool with one worker must still execute all items correctly.
    ThreadPool pool(1);
    constexpr int kCount = 50;
    std::atomic<int> calls{0};
    pool.run(kCount, [&](int) { ++calls; });
    EXPECT_EQ(calls.load(), kCount);
}
