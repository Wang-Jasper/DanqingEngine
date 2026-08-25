#include "renderer/GpuProfiler.h"
#include "core/VulkanContext.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

void GpuProfiler::init(VulkanContext &ctx, uint32_t framesInFlight)
{
    ctx_    = &ctx;
    device_ = ctx.getDevice();

    tsPeriodNs_ = ctx.getTimestampPeriod();
    uint32_t validBits = ctx.getTimestampValidBits();
    available_ = (validBits > 0) && (tsPeriodNs_ > 0.0f);

    // Probe pipelineStatisticsQuery support (query device features rather than cached state)
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(ctx.getPhysicalDevice(), &feats);
    psSupported_ = (feats.pipelineStatisticsQuery == VK_TRUE);

    if (!available_)
    {
        std::cerr << "[GpuProfiler] Timestamp queries not supported on this queue (validBits=0). Profiler disabled.\n";
        return;
    }

    frames_.resize(framesInFlight);

    for (auto &fr : frames_)
    {
        // Timestamp pool: MAX_PASSES * 2 slots per frame
        VkQueryPoolCreateInfo tsInfo{};
        tsInfo.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        tsInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        tsInfo.queryCount = MAX_PASSES * 2;
        if (vkCreateQueryPool(device_, &tsInfo, nullptr, &fr.tsPool) != VK_SUCCESS)
        {
            throw std::runtime_error("[GpuProfiler] Failed to create timestamp query pool");
        }
        // No vkResetQueryPool here: it requires the hostQueryReset feature.
        // Pools are reset in-command via vkCmdResetQueryPool at beginFrame.

        // Pipeline statistics pool: MAX_PASSES slots per frame (one per pass)
        if (psSupported_)
        {
            VkQueryPoolCreateInfo psInfo{};
            psInfo.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            psInfo.queryType  = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            psInfo.queryCount = MAX_PASSES;
            psInfo.pipelineStatistics =
                VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
                VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
                VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;
            if (vkCreateQueryPool(device_, &psInfo, nullptr, &fr.psPool) != VK_SUCCESS)
            {
                throw std::runtime_error("[GpuProfiler] Failed to create pipeline statistics query pool");
            }
            // Same: no host-side reset needed
        }
    }

    std::cout << "[GpuProfiler] Initialized. tsPeriod=" << tsPeriodNs_ << "ns, "
              << "validBits=" << validBits << ", "
              << "pipelineStats=" << (psSupported_ ? "yes" : "no") << "\n";
}

void GpuProfiler::cleanup()
{
    if (!device_) return;
    for (auto &fr : frames_)
    {
        if (fr.tsPool != VK_NULL_HANDLE) vkDestroyQueryPool(device_, fr.tsPool, nullptr);
        if (fr.psPool != VK_NULL_HANDLE) vkDestroyQueryPool(device_, fr.psPool, nullptr);
        fr.tsPool = VK_NULL_HANDLE;
        fr.psPool = VK_NULL_HANDLE;
    }
    frames_.clear();
    passStats_.clear();
    pipelineStats_.clear();
    passOrder_.clear();
    device_ = VK_NULL_HANDLE;
}

void GpuProfiler::beginFrame(VkCommandBuffer cmd, uint32_t frameIdx)
{
    if (!available_) return;
    currentFrameIdx_ = frameIdx;
    auto &fr = frames_[frameIdx];

    // Reset this frame's query pools before recording
    vkCmdResetQueryPool(cmd, fr.tsPool, 0, MAX_PASSES * 2);
    if (psSupported_ && fr.psPool != VK_NULL_HANDLE)
        vkCmdResetQueryPool(cmd, fr.psPool, 0, MAX_PASSES);

    fr.passes.clear();
    fr.psPassNames.clear();
    fr.nextTsSlot = 0;
    fr.nextPsSlot = 0;
    fr.activePsIdx = -1;
    fr.recorded = true;
}

void GpuProfiler::endFrame(VkCommandBuffer /*cmd*/)
{
    // No extra commands needed yet; kept for future extension (e.g. total frame time)
}

void GpuProfiler::beginPass(VkCommandBuffer cmd, const char *name)
{
    if (!available_) return;
    auto &fr = frames_[currentFrameIdx_];
    if (fr.nextTsSlot + 2 > MAX_PASSES * 2)
    {
        // Slots exhausted; silently skip
        return;
    }
    PassSlot slot;
    slot.name      = name;
    slot.beginSlot = fr.nextTsSlot++;
    slot.endSlot   = fr.nextTsSlot++;

    // Stage choice: begin writes at TOP_OF_PIPE so commands issued before the
    // marker are included; endPass writes at BOTTOM_OF_PIPE.
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, fr.tsPool, slot.beginSlot);
    fr.passes.push_back(std::move(slot));
}

void GpuProfiler::endPass(VkCommandBuffer cmd, const char *name)
{
    if (!available_) return;
    auto &fr = frames_[currentFrameIdx_];
    // Find the last unclosed slot with this name
    for (auto it = fr.passes.rbegin(); it != fr.passes.rend(); ++it)
    {
        if (!it->ended && it->name == name)
        {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, fr.tsPool, it->endSlot);
            it->ended = true;
            return;
        }
    }
}

void GpuProfiler::beginPipelineStats(VkCommandBuffer cmd, const char *name)
{
    if (!available_ || !psSupported_) return;
    auto &fr = frames_[currentFrameIdx_];
    if (fr.activePsIdx >= 0) return; // Nested stats not supported
    if (fr.nextPsSlot >= MAX_PASSES) return;
    fr.activePsIdx = static_cast<int>(fr.nextPsSlot);
    fr.psPassNames.emplace_back(name);
    vkCmdBeginQuery(cmd, fr.psPool, fr.nextPsSlot, 0);
    fr.nextPsSlot++;
}

void GpuProfiler::endPipelineStats(VkCommandBuffer cmd)
{
    if (!available_ || !psSupported_) return;
    auto &fr = frames_[currentFrameIdx_];
    if (fr.activePsIdx < 0) return;
    vkCmdEndQuery(cmd, fr.psPool, static_cast<uint32_t>(fr.activePsIdx));
    fr.activePsIdx = -1;
}

void GpuProfiler::resolve(uint32_t frameIdx)
{
    if (!available_) return;
    auto &fr = frames_[frameIdx];
    if (!fr.recorded || fr.passes.empty()) return;

    // ---- Timestamp readback ----
    std::array<uint64_t, MAX_PASSES * 2> tsResults{};
    VkResult r = vkGetQueryPoolResults(device_, fr.tsPool, 0, fr.nextTsSlot,
                                       sizeof(uint64_t) * fr.nextTsSlot,
                                       tsResults.data(), sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r == VK_SUCCESS)
    {
        for (auto &slot : fr.passes)
        {
            if (!slot.ended) continue;
            uint64_t t0 = tsResults[slot.beginSlot];
            uint64_t t1 = tsResults[slot.endSlot];
            if (t1 < t0) continue; // Skip inconsistent reads
            float ms = static_cast<float>((t1 - t0) * static_cast<double>(tsPeriodNs_) * 1e-6);

            auto &stat = passStats_[slot.name];
            if (passStats_.find(slot.name) == passStats_.end() ||
                std::find(passOrder_.begin(), passOrder_.end(), slot.name) == passOrder_.end())
            {
                passOrder_.push_back(slot.name);
            }
            stat.lastMs = ms;
            appendHistory(stat, ms);
            recomputeAggregates(stat);
        }
    }

    // ---- Pipeline statistics readback ----
    if (psSupported_ && fr.psPool != VK_NULL_HANDLE && fr.nextPsSlot > 0)
    {
        // 4 stat fields x N passes
        std::vector<uint64_t> psResults(fr.nextPsSlot * 4, 0);
        VkResult pr = vkGetQueryPoolResults(device_, fr.psPool, 0, fr.nextPsSlot,
                                            psResults.size() * sizeof(uint64_t),
                                            psResults.data(), sizeof(uint64_t) * 4,
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (pr == VK_SUCCESS)
        {
            for (uint32_t i = 0; i < fr.psPassNames.size() && i < fr.nextPsSlot; ++i)
            {
                PipelineStats ps;
                ps.vertices          = psResults[i * 4 + 0];
                ps.primitives        = psResults[i * 4 + 1];
                ps.fragInvocations   = psResults[i * 4 + 2];
                ps.clippedPrimitives = psResults[i * 4 + 3];
                pipelineStats_[fr.psPassNames[i]] = ps;
            }
        }
    }
}

void GpuProfiler::appendHistory(PassStats &s, float ms) const
{
    s.history.push_back(ms);
    while (s.history.size() > WINDOW_SIZE) s.history.pop_front();
}

void GpuProfiler::recomputeAggregates(PassStats &s) const
{
    if (s.history.empty()) return;
    float sum = 0.0f, mx = 0.0f;
    for (float v : s.history) { sum += v; if (v > mx) mx = v; }
    s.avgMs = sum / static_cast<float>(s.history.size());
    s.maxMs = mx;

    // p95: copy and sort (window is at most 120 samples; cost negligible)
    std::vector<float> sorted(s.history.begin(), s.history.end());
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(0.95f * (sorted.size() - 1));
    s.p95Ms = sorted[idx];
}
