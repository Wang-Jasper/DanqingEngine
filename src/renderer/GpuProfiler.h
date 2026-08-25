#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <cstdint>

class VulkanContext;

// GPU performance analysis: timestamp queries for per-pass timing, pipeline
// statistics for vertex/primitive/fragment counts, and a 120-frame sliding window
// (last/avg/max/p95). Call init() once, wrap passes with beginPass/endPass, and
// resolve() after the frame's fence has been waited.
class GpuProfiler
{
public:
    static constexpr uint32_t MAX_PASSES   = 16;
    static constexpr uint32_t WINDOW_SIZE  = 120;

    // 4 fields of interest; order matches the query flag bits
    struct PipelineStats
    {
        uint64_t vertices          = 0;
        uint64_t primitives        = 0;
        uint64_t fragInvocations   = 0;
        uint64_t clippedPrimitives = 0;
    };

    struct PassStats
    {
        float lastMs = 0.0f;
        float avgMs  = 0.0f;
        float maxMs  = 0.0f;
        float p95Ms  = 0.0f;
        std::deque<float> history;
    };

    void init(VulkanContext &ctx, uint32_t framesInFlight);
    void cleanup();

    // Frame-level API
    void beginFrame(VkCommandBuffer cmd, uint32_t frameIdx);
    void endFrame(VkCommandBuffer cmd);
    // Call after the queue submit's fence has been waited, so results are ready to read.
    void resolve(uint32_t frameIdx);

    // Pass-level API; timestamp slots are auto-assigned by call order
    void beginPass(VkCommandBuffer cmd, const char *name);
    void endPass  (VkCommandBuffer cmd, const char *name);

    // Pipeline statistics; wrap the Geometry/Lighting passes once each
    void beginPipelineStats(VkCommandBuffer cmd, const char *name);
    void endPipelineStats  (VkCommandBuffer cmd);

    // Query interface (read by EditorUI)
    const std::unordered_map<std::string, PassStats>     &getPassStats()     const { return passStats_; }
    const std::unordered_map<std::string, PipelineStats> &getPipelineStats() const { return pipelineStats_; }
    const std::vector<std::string>                       &getPassOrder()     const { return passOrder_; }
    bool  isAvailable() const { return available_; }

private:
    struct PassSlot
    {
        std::string name;
        uint32_t    beginSlot = 0;
        uint32_t    endSlot   = 0;
        bool        ended     = false;
    };

    struct FrameRecord
    {
        VkQueryPool tsPool   = VK_NULL_HANDLE; // timestamp pool
        VkQueryPool psPool   = VK_NULL_HANDLE; // pipeline statistics pool
        std::vector<PassSlot> passes;          // Passes recorded this frame, in order.
        std::vector<std::string> psPassNames;  // Pass names matching this frame's pipeline stats.
        uint32_t nextTsSlot  = 0;
        uint32_t nextPsSlot  = 0;
        bool     recorded    = false;          // Whether commands were actually recorded this frame.
        int      activePsIdx = -1;             // Currently open pipeline-stats query index.
    };

    void appendHistory(PassStats &s, float ms) const;
    void recomputeAggregates(PassStats &s) const;

    VulkanContext *ctx_ = nullptr;
    VkDevice       device_ = VK_NULL_HANDLE;
    float          tsPeriodNs_ = 1.0f;

    bool available_   = false;
    bool psSupported_ = false;

    std::vector<FrameRecord> frames_;
    uint32_t currentFrameIdx_ = 0;

    std::unordered_map<std::string, PassStats>     passStats_;
    std::unordered_map<std::string, PipelineStats> pipelineStats_;
    std::vector<std::string>                       passOrder_; // First-appearance order, so the UI display stays stable
};
