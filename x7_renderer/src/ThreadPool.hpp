/// ThreadPool.hpp
//
// Persistent work-stealing thread pool.  One instance is kept alive on the
// Context so thread creation/join overhead is paid only once per renderer
// lifetime rather than once per render row-batch.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace RCTGen {
    class ThreadPool {
    public:
        explicit ThreadPool(unsigned int n) {
            workers_.reserve(n);
            for (unsigned int i = 0; i < n; ++i) workers_.emplace_back([this] { worker_loop(); });
        }

        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                stop_ = true;
            }
            work_cv_.notify_all();
            for (auto& t : workers_) t.join();
        }

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        // Invoke fn(i) for i in [0, count), spread across the thread pool.
        // Blocks until all invocations complete.  Thread-safe with respect
        // to other run() calls only if called from a single thread at a time.
        void run(int count, const std::function<void(int)>& fn) {
            if (count <= 0) return;

            // Distribute work in chunks: one task per worker, each draining
            // an atomic counter.  This avoids queue churn for small counts.
            std::atomic<int> next{0};
            const int tasks = std::min(count, static_cast<int>(workers_.size()));
            {
                std::unique_lock<std::mutex> lock(mutex_);
                for (int i = 0; i < tasks; ++i) {
                    queue_.emplace_back([&next, count, &fn] {
                        for (;;) {
                            const int j = next.fetch_add(1, std::memory_order_relaxed);
                            if (j >= count) return;
                            fn(j);
                        }
                    });
                }
            }
            work_cv_.notify_all();

            std::unique_lock<std::mutex> lock(mutex_);
            idle_cv_.wait(lock, [this] { return busy_ == 0 && queue_.empty(); });
        }

    private:
        void worker_loop() {
            std::unique_lock<std::mutex> lock(mutex_);
            for (;;) {
                work_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                auto task = std::move(queue_.front());
                queue_.pop_front();
                ++busy_;
                lock.unlock();
                task();
                lock.lock();
                if (--busy_ == 0 && queue_.empty()) idle_cv_.notify_all();
            }
        }

        std::vector<std::thread> workers_;
        std::deque<std::function<void()>> queue_;
        std::mutex mutex_;
        std::condition_variable work_cv_;
        std::condition_variable idle_cv_;
        int busy_ = 0;
        bool stop_ = false;
    };
} // namespace RCTGen
