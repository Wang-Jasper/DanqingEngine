// Benchmark state machine: Idle -> start() -> Warmup (5s, discards data) ->
// Sampling (10s) -> Idle. tick() advances it once per frame; on sampling end it
// writes benchmarks/result_<scene>_<YYYYMMDD_HHMMSS>.csv and returns to Idle.
// Draw calls are bucketed (geometry/light volume/lighting/gizmo/imgui) so
// geometry_avg lines up with Unity URP's UnityStats.drawCalls; the other buckets
// are engine-specific commits and must not be merged into one total for
// cross-engine comparison.
// Deliberately single-threaded and non-blocking: never grabs camera control, and
// ImGui keeps running during a run (its draws count toward the imgui bucket).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

class BenchmarkRunner
{
public:
    enum class State
    {
        Idle,
        Warmup,
        Sampling,
    };

    struct Sample
    {
        double cpuMs = 0.0;
        double gpuMs = 0.0;
        double frameMs = 0.0;
        uint32_t draws = 0; // Total across all buckets
        // Per-bucket counts (mirror DrawCallStats) for cross-engine comparison.
        uint32_t drawsGeometry = 0;
        uint32_t drawsLightVolume = 0;
        uint32_t drawsLighting = 0;
        uint32_t drawsGizmo = 0;
        uint32_t drawsImGui = 0;
    };

    struct Result
    {
        std::string scene;
        double fpsAvg = 0.0, fpsP95 = 0.0, fpsMax = 0.0;
        double frameMsAvg = 0.0, frameMsP95 = 0.0;
        double gpuMsAvg = 0.0, cpuMsAvg = 0.0;
        double drawsAvg = 0.0;
        double drawsGeometryAvg = 0.0;
        double drawsLightVolumeAvg = 0.0;
        double drawsLightingAvg = 0.0;
        double drawsGizmoAvg = 0.0;
        double drawsImGuiAvg = 0.0;
        size_t samples = 0;
        std::string csvPath; // Set once the CSV has been written
    };

    // ------------------- Control -------------------

    bool isRunning() const { return state != State::Idle; }
    State getState() const { return state; }
    const std::string &getScene() const { return scene; }

    double getPhaseElapsedSec() const
    {
        if (state == State::Idle)
            return 0.0;
        auto now = Clock::now();
        return std::chrono::duration<double>(now - phaseStart).count();
    }

    double getWarmupDuration() const { return warmupSec; }
    double getSamplingDuration() const { return samplingSec; }

    void start(const std::string &sceneName)
    {
        scene = sceneName;
        samples.clear();
        state = State::Warmup;
        phaseStart = Clock::now();
        std::printf("[Bench] start: scene='%s' warmup=%.1fs sampling=%.1fs\n",
                    sceneName.c_str(), warmupSec, samplingSec);
    }

    void abort()
    {
        if (state == State::Idle)
            return;
        std::printf("[Bench] aborted at state=%s, %zu samples discarded\n",
                    stateName(state), samples.size());
        state = State::Idle;
        samples.clear();
    }

    // Call once per frame, after drawFrame() and profiler.endFrame(). Returns true
    // only when sampling just finished (CSV written), so the caller can show the result.
    //
    // Legacy overload taking only the total draw count; kept so callers don't have to change.
    bool tick(double cpuMs, double gpuMs, double frameMs, uint32_t draws)
    {
        Sample s;
        s.cpuMs = cpuMs;
        s.gpuMs = gpuMs;
        s.frameMs = frameMs;
        s.draws = draws;
        return tickSample(s);
    }

    // Overload taking per-bucket DrawCallStats; the CSV emits each bucket separately.
    bool tick(double cpuMs, double gpuMs, double frameMs,
              uint32_t drawsTotal,
              uint32_t drawsGeometry,
              uint32_t drawsLightVolume,
              uint32_t drawsLighting,
              uint32_t drawsGizmo,
              uint32_t drawsImGui)
    {
        Sample s;
        s.cpuMs = cpuMs;
        s.gpuMs = gpuMs;
        s.frameMs = frameMs;
        s.draws = drawsTotal;
        s.drawsGeometry = drawsGeometry;
        s.drawsLightVolume = drawsLightVolume;
        s.drawsLighting = drawsLighting;
        s.drawsGizmo = drawsGizmo;
        s.drawsImGui = drawsImGui;
        return tickSample(s);
    }

private:
    bool tickSample(const Sample &sIn)
    {
        if (state == State::Idle)
            return false;

        double elapsed = getPhaseElapsedSec();

        if (state == State::Warmup)
        {
            if (elapsed >= warmupSec)
            {
                state = State::Sampling;
                phaseStart = Clock::now();
                std::printf("[Bench] warmup done, sampling started\n");
            }
            return false;
        }

            // Sampling
        samples.push_back(sIn);

        if (elapsed >= samplingSec)
        {
            lastResult = computeResult();
            lastResult.csvPath = writeCsv(lastResult);
            std::printf("[Bench] done: %s   fps avg=%.1f p95=%.1f   frame=%.2fms   draws=%.0f (geom=%.0f lv=%.0f light=%.0f giz=%.0f gui=%.0f)\n",
                        scene.c_str(), lastResult.fpsAvg, lastResult.fpsP95,
                        lastResult.frameMsAvg, lastResult.drawsAvg,
                        lastResult.drawsGeometryAvg,
                        lastResult.drawsLightVolumeAvg,
                        lastResult.drawsLightingAvg,
                        lastResult.drawsGizmoAvg,
                        lastResult.drawsImGuiAvg);
            state = State::Idle;
            return true;
        }
        return false;
    }

public:
    const Result &getLastResult() const { return lastResult; }

    // ------------------- Configuration -------------------

    void setDurations(double warmup, double sampling)
    {
        warmupSec = std::max(0.0, warmup);
        samplingSec = std::max(0.1, sampling);
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    State state = State::Idle;
    std::string scene;
    TimePoint phaseStart;
    std::vector<Sample> samples;
    Result lastResult;

    double warmupSec = 5.0;
    double samplingSec = 10.0;

    static const char *stateName(State s)
    {
        switch (s)
        {
        case State::Idle:
            return "Idle";
        case State::Warmup:
            return "Warmup";
        case State::Sampling:
            return "Sampling";
        }
        return "?";
    }

    Result computeResult() const
    {
        Result r;
        r.scene = scene;
        r.samples = samples.size();
        if (samples.empty())
            return r;

        std::vector<double> fps;
        std::vector<double> frameMs;
        fps.reserve(samples.size());
        frameMs.reserve(samples.size());

        double sumCpu = 0, sumGpu = 0, sumFrame = 0, sumDraws = 0;
        double sumGeom = 0, sumLV = 0, sumLight = 0, sumGiz = 0, sumGui = 0;
        for (const auto &s : samples)
        {
            double f = s.frameMs > 0.0001 ? 1000.0 / s.frameMs : 0.0;
            fps.push_back(f);
            frameMs.push_back(s.frameMs);
            sumCpu += s.cpuMs;
            sumGpu += s.gpuMs;
            sumFrame += s.frameMs;
            sumDraws += static_cast<double>(s.draws);
            sumGeom += static_cast<double>(s.drawsGeometry);
            sumLV += static_cast<double>(s.drawsLightVolume);
            sumLight += static_cast<double>(s.drawsLighting);
            sumGiz += static_cast<double>(s.drawsGizmo);
            sumGui += static_cast<double>(s.drawsImGui);
        }
        const double n = static_cast<double>(samples.size());
        r.fpsAvg = 0.0;
        for (double f : fps)
            r.fpsAvg += f;
        r.fpsAvg /= n;
        r.cpuMsAvg = sumCpu / n;
        r.gpuMsAvg = sumGpu / n;
        r.frameMsAvg = sumFrame / n;
        r.drawsAvg = sumDraws / n;
        r.drawsGeometryAvg = sumGeom / n;
        r.drawsLightVolumeAvg = sumLV / n;
        r.drawsLightingAvg = sumLight / n;
        r.drawsGizmoAvg = sumGiz / n;
        r.drawsImGuiAvg = sumGui / n;

        // p95 / max
        std::vector<double> fpsSorted = fps;
        std::sort(fpsSorted.begin(), fpsSorted.end());
        // FPS "p95" is the 5th percentile (low end, i.e. stutter); max is the peak.
        size_t lowIdx = static_cast<size_t>(fpsSorted.size() * 0.05);
        if (lowIdx >= fpsSorted.size())
            lowIdx = fpsSorted.size() - 1;
        r.fpsP95 = fpsSorted[lowIdx];
        r.fpsMax = fpsSorted.back();

        std::vector<double> frameSorted = frameMs;
        std::sort(frameSorted.begin(), frameSorted.end());
        size_t hiIdx = static_cast<size_t>(frameSorted.size() * 0.95);
        if (hiIdx >= frameSorted.size())
            hiIdx = frameSorted.size() - 1;
        r.frameMsP95 = frameSorted[hiIdx];

        return r;
    }

    static std::string timeStamp()
    {
        std::time_t t = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                      tm.tm_hour, tm.tm_min, tm.tm_sec);
        return buf;
    }

    static std::string sanitizeScene(const std::string &s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s)
        {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
                out.push_back(c);
            else
                out.push_back('_');
        }
        return out;
    }

    // Returns the absolute path written, or an empty string on failure.
    std::string writeCsv(const Result &r)
    {
        namespace fs = std::filesystem;
        fs::path dir = fs::current_path() / "benchmarks";
        std::error_code ec;
        fs::create_directories(dir, ec);
        std::string fname = "result_" + sanitizeScene(r.scene) + "_" + timeStamp() + ".csv";
        fs::path full = dir / fname;

        std::FILE *fp = std::fopen(full.string().c_str(), "wb");
        if (!fp)
        {
            std::fprintf(stderr, "[Bench] failed to open '%s' for write\n", full.string().c_str());
            return {};
        }
        std::fprintf(fp,
                     "scene,fps_avg,fps_p95,fps_max,frame_ms_avg,frame_ms_p95,"
                     "gpu_ms_avg,cpu_ms_avg,draws_avg,samples,"
                     "geometry_avg,light_volume_avg,lighting_avg,gizmo_avg,imgui_avg\n");
        std::fprintf(fp,
                     "%s,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.2f,%zu,"
                     "%.2f,%.2f,%.2f,%.2f,%.2f\n",
                     r.scene.c_str(),
                     r.fpsAvg, r.fpsP95, r.fpsMax,
                     r.frameMsAvg, r.frameMsP95,
                     r.gpuMsAvg, r.cpuMsAvg,
                     r.drawsAvg, r.samples,
                     r.drawsGeometryAvg,
                     r.drawsLightVolumeAvg,
                     r.drawsLightingAvg,
                     r.drawsGizmoAvg,
                     r.drawsImGuiAvg);
        std::fclose(fp);
        std::printf("[Bench] csv written: %s\n", full.string().c_str());
        return full.string();
    }
};
