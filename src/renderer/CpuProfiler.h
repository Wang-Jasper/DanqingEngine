// CPU-side per-pass timing, API-symmetric with GpuProfiler but pure std::chrono.
// Maintains last/avg/p95/max over the last 16 frames for the EditorUI overlay.
// Stack-based nested begin/end only (like the GPU profiler); not thread-safe;
// string names as keys (O(N) lookup, fine for N <= 16).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

class CpuProfiler
{
public:
    static constexpr size_t WINDOW_SIZE = 16;

    void beginFrame()
    {
        // Don't rebuild entries; just zero lastMs so each frame recomputes
        // (same-name passes that run multiple times accumulate).
        for (auto &[k, v] : entries)
            v.lastMs = 0.0;
    }

    void beginPass(const std::string &name)
    {
        TimePoint now = Clock::now();
        startStack.push({name, now});
    }

    void endPass(const std::string &name)
    {
        if (startStack.empty())
            return;
        auto top = startStack.top();
        startStack.pop();
        if (top.name != name)
        {
            // Mismatched begin/end; ignore (assert in debug builds if needed).
            return;
        }
        TimePoint now = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - top.start).count();

        auto &entry = entries[name];
        entry.lastMs += ms; // Same-name runs in a frame accumulate
    }

    void endFrame()
    {
        for (auto &[k, v] : entries)
        {
            v.window.push_back(v.lastMs);
            if (v.window.size() > WINDOW_SIZE)
                v.window.erase(v.window.begin());

            v.avgMs = 0.0;
            v.maxMs = 0.0;
            for (double s : v.window)
            {
                v.avgMs += s;
                v.maxMs = std::max(v.maxMs, s);
            }
            v.avgMs /= static_cast<double>(v.window.size());

            std::vector<double> sorted = v.window;
            std::sort(sorted.begin(), sorted.end());
            size_t p95Idx = static_cast<size_t>(sorted.size() * 0.95);
            if (p95Idx >= sorted.size())
                p95Idx = sorted.size() - 1;
            v.p95Ms = sorted[p95Idx];
        }
    }

    struct PassEntry
    {
        double lastMs = 0.0;
        double avgMs = 0.0;
        double p95Ms = 0.0;
        double maxMs = 0.0;
        std::vector<double> window;
    };

    // Read-only access for the overlay
    const std::unordered_map<std::string, PassEntry> &getEntries() const { return entries; }

    // Fixed display order for known passes (unordered_map iteration order is arbitrary)
    static const std::vector<std::string> &getDisplayOrder()
    {
        static const std::vector<std::string> order = {
            "Frame Total",
            "Physics",
            "Update Buffers",
            "BVH Rebuild",
            "Light Gather + Cull",
            "Record Cmd",
            "EditorUI Build",
        };
        return order;
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    struct StartEntry
    {
        std::string name;
        TimePoint start;
    };

    std::stack<StartEntry> startStack;
    std::unordered_map<std::string, PassEntry> entries;
};
