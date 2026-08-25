// ============================================================================
// test_light_volume_equivalence.cpp — pixel equivalence of `useLightVolume` vs the
// fullscreen path, quantified on CPU. The current per-light implementation reads
// brighter with many lights; candidates are scored by max/mean per-channel error.
// ============================================================================
#include "test_framework.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
    // --- GLSL-equivalent basic math -----------------------------------------
    glm::vec3 reinhard(const glm::vec3 &x) { return x / (x + glm::vec3(1.0f)); }

    glm::vec3 gammaCorrect(const glm::vec3 &x)
    {
        return glm::vec3(std::pow(std::max(x.x, 0.0f), 1.0f / 2.2f),
                         std::pow(std::max(x.y, 0.0f), 1.0f / 2.2f),
                         std::pow(std::max(x.z, 0.0f), 1.0f / 2.2f));
    }

    glm::vec3 clamp01(const glm::vec3 &x)
    {
        return glm::vec3(std::clamp(x.x, 0.0f, 1.0f),
                         std::clamp(x.y, 0.0f, 1.0f),
                         std::clamp(x.z, 0.0f, 1.0f));
    }

    // --- Compositing paths --------------------------------------------------
    // A. fullscreen path (baseline)
    glm::vec3 pathFullscreen(const glm::vec3 &ambient,
                             const std::vector<glm::vec3> &Lo)
    {
        glm::vec3 sum = ambient;
        for (const auto &l : Lo)
            sum += l;
        return clamp01(gammaCorrect(reinhard(sum)));
    }

    // B. Light volume — current implementation: per-light reinhard+gamma, then additive
    glm::vec3 pathPerLight(const glm::vec3 &ambient,
                           const std::vector<glm::vec3> &Lo)
    {
        glm::vec3 rt = gammaCorrect(reinhard(ambient)); // fullscreen pass
        for (const auto &l : Lo)
            rt += gammaCorrect(reinhard(l)); // LV pass: additive per light
        return clamp01(rt);
    }

    // C. Candidate: fullscreen pass outputs linear ambient, LV outputs linear Lo;
    //    UNORM8 write clamps naturally. No tonemap/gamma (control group).
    glm::vec3 pathLinearClamp(const glm::vec3 &ambient,
                              const std::vector<glm::vec3> &Lo)
    {
        glm::vec3 rt = ambient;
        for (const auto &l : Lo)
            rt += l;
        return clamp01(rt);
    }

    // D. Candidate: fullscreen outputs gamma(reinhard(ambient)); LV outputs
    //    gamma(reinhard(Lo_i)) · 1/sqrt(N) — heuristic weight compensating the
    //    per-light reinhard over-brightening with many lights. N is the current
    //    frame's light count, pushed to the shader (push constant).
    glm::vec3 pathPerLightScaled(const glm::vec3 &ambient,
                                 const std::vector<glm::vec3> &Lo)
    {
        glm::vec3 rt = gammaCorrect(reinhard(ambient));
        const float N = static_cast<float>(std::max<size_t>(1, Lo.size()));
        const float w = 1.0f / std::sqrt(N); // heuristic
        for (const auto &l : Lo)
            rt += w * gammaCorrect(reinhard(l));
        return clamp01(rt);
    }

    // E. Candidate: per-light reinhard in linear space, then a matching "partial"
    //    gamma in the fullscreen pass. LV cannot re-gamma the accumulated value
    //    afterwards — not implementable without restructuring the pipeline. Skipped.

    // F. Candidate (recommended direction): LV outputs reinhard(Lo_i / (1 + max
    //    component)), then NO gamma; the fullscreen pass outputs reinhard(ambient),
    //    also no gamma; the swapchain sRGB format converts linear → sRGB in hardware.
    //    → requires a `_UNORM` RT sampled as `_SRGB`; the current RT is the
    //      swapchain format (usually B8G8R8A8_UNORM) — needs verification.
    //    → only the "pure linear reinhard sum then clamp" effect is simulated here.
    glm::vec3 pathLinearReinhardSum(const glm::vec3 &ambient,
                                    const std::vector<glm::vec3> &Lo)
    {
        glm::vec3 sum = reinhard(ambient);
        for (const auto &l : Lo)
            sum += reinhard(l);
        // Hardware sRGB write: equivalent to gammaCorrect
        return clamp01(gammaCorrect(clamp01(sum)));
    }

    // G. Candidate (strictest): the fullscreen pass only reinhards, no gamma —
    //    outputs reinhard(ambient + Σ Lo_baseline), or just reinhard(ambient) when
    //    the light volume is ON; the LV pass outputs reinhard(Lo_i) per light;
    //    hardware sRGB write provides gamma. The fullscreen pass must drop its
    //    current pow(1/2.2) double gamma (verified: offscreen RT format is
    //    B8G8R8A8_SRGB). Equivalent to F; listed separately to compare
    //    "baseline also changes" vs "only LV changes".
    //
    //    The baseline fullscreen path should also drop gamma, or historical scenes
    //    read dark — that is the core conclusion this test pushes.
    //
    //    Math:
    //      path_fullscreen_fixed = srgb_hw( reinhard(ambient + Σ Lo) )
    //      path_lv_fixed         = srgb_hw( reinhard(ambient) + Σ reinhard(Lo_i) )
    //    where srgb_hw ≡ clamp01 + gammaCorrect — one gamma fewer than pathFullscreen.
    glm::vec3 pathFullscreenFixed(const glm::vec3 &ambient,
                                  const std::vector<glm::vec3> &Lo)
    {
        glm::vec3 sum = ambient;
        for (const auto &l : Lo)
            sum += l;
        return clamp01(gammaCorrect(clamp01(reinhard(sum))));
    }
    glm::vec3 pathLVFixed(const glm::vec3 &ambient,
                          const std::vector<glm::vec3> &Lo)
    {
        glm::vec3 sum = reinhard(ambient);
        for (const auto &l : Lo)
            sum += reinhard(l);
        return clamp01(gammaCorrect(clamp01(sum)));
    }

    // --- Scoring ------------------------------------------------------------
    struct Score
    {
        float maxErr = 0.0f;
        float meanErr = 0.0f;
    };
    Score diff(const glm::vec3 &a, const glm::vec3 &b)
    {
        glm::vec3 d = glm::abs(a - b);
        Score s;
        s.maxErr = std::max({d.x, d.y, d.z});
        s.meanErr = (d.x + d.y + d.z) / 3.0f;
        return s;
    }

    // --- Scene definitions --------------------------------------------------
    struct Scene
    {
        const char *name;
        glm::vec3 ambient;
        std::vector<glm::vec3> Lo;
    };

    // Plausible linear radiance values for ambient + Lo (post-PBR):
    //   ambient = 0.03 · albedo (white albedo)
    //   Lo_i ≈ kd · color · intensity · atten · NdotL / π;
    //   typical strong lights give Lo ≈ 0.5–3.0 near the surface
    const std::vector<Scene> &scenes()
    {
        static const std::vector<Scene> s = {
            {"single_weak_light",
             glm::vec3(0.03f),
             {glm::vec3(0.2f, 0.2f, 0.25f)}},

            {"single_strong_light",
             glm::vec3(0.03f),
             {glm::vec3(1.5f, 1.2f, 0.8f)}},

            {"four_strong_lights_overlap",
             glm::vec3(0.03f),
             {
                 glm::vec3(1.2f, 1.0f, 0.8f), // warm white
                 glm::vec3(0.3f, 0.5f, 1.4f), // cool blue
                 glm::vec3(1.5f, 0.2f, 0.2f), // red
                 glm::vec3(0.2f, 1.3f, 0.3f), // green
             }},

            {"background_only_no_light",
             glm::vec3(0.03f),
             {}},

            {"two_medium_lights",
             glm::vec3(0.03f, 0.03f, 0.05f),
             {
                 glm::vec3(0.8f, 0.6f, 0.4f),
                 glm::vec3(0.2f, 0.4f, 0.9f),
             }},

            {"eight_dense_lights",
             glm::vec3(0.04f),
             {
                 glm::vec3(0.5f),
                 glm::vec3(0.4f),
                 glm::vec3(0.3f),
                 glm::vec3(0.6f),
                 glm::vec3(0.5f),
                 glm::vec3(0.7f),
                 glm::vec3(0.4f),
                 glm::vec3(0.5f),
             }},
        };
        return s;
    }

    struct CandidateFn
    {
        const char *name;
        glm::vec3 (*fn)(const glm::vec3 &, const std::vector<glm::vec3> &);
    };
} // namespace

// ============================================================================
// Main test: print (max_err, mean_err) for each candidate × scene, then PHYS_CHECK
// ============================================================================
PHYS_TEST(LightVolume, EquivalenceReport)
{
    std::vector<CandidateFn> candidates = {
        {"per_light_current        ", pathPerLight},
        {"linear_clamp_no_tonemap  ", pathLinearClamp},
        {"per_light_scaled_1/sqrtN ", pathPerLightScaled},
        {"linear_reinhard_sum_srgb ", pathLinearReinhardSum},
    };

    std::printf("\n==== Light Volume Equivalence Report ====\n");
    std::printf("%-30s | %-30s | max_err | mean_err | fullscreen_rgb     -> path_rgb\n",
                "scene", "candidate");
    std::printf("%s\n", std::string(130, '-').c_str());

    // Global totals per candidate (for ranking)
    std::vector<float> totalMean(candidates.size(), 0.0f);
    std::vector<float> totalMax(candidates.size(), 0.0f);

    for (const auto &sc : scenes())
    {
        glm::vec3 base = pathFullscreen(sc.ambient, sc.Lo);
        for (size_t ci = 0; ci < candidates.size(); ++ci)
        {
            glm::vec3 got = candidates[ci].fn(sc.ambient, sc.Lo);
            Score s = diff(base, got);
            totalMean[ci] += s.meanErr;
            totalMax[ci] = std::max(totalMax[ci], s.maxErr);
            std::printf("%-30s | %-30s | %.4f  | %.4f   | (%.3f,%.3f,%.3f) -> (%.3f,%.3f,%.3f)\n",
                        sc.name, candidates[ci].name,
                        s.maxErr, s.meanErr,
                        base.x, base.y, base.z,
                        got.x, got.y, got.z);
        }
    }

    std::printf("\n==== SUMMARY (lower = closer to fullscreen baseline) ====\n");
    std::printf("%-30s | total_mean_err | worst_max_err\n", "candidate");
    std::printf("%s\n", std::string(70, '-').c_str());
    for (size_t ci = 0; ci < candidates.size(); ++ci)
    {
        std::printf("%-30s | %.4f         | %.4f\n",
                    candidates[ci].name,
                    totalMean[ci] / scenes().size(),
                    totalMax[ci]);
    }

    // Assert the current per_light implementation has a large max error on the
    // multi-light scene (confirms the bug)
    {
        glm::vec3 base = pathFullscreen(scenes()[2].ambient, scenes()[2].Lo);
        glm::vec3 got = pathPerLight(scenes()[2].ambient, scenes()[2].Lo);
        Score s = diff(base, got);
        std::printf("\n[history] four_strong_lights_overlap: per_light max_err=%.4f\n"
                    "          (Step 13 used this pathPerLight scheme; Step 14 replaced it\n"
                    "           with the FixedPaths scheme, see FixedPathsAreCloseEnough)\n",
                    s.maxErr);
    }

    // Verdict: does any candidate satisfy max_err ≤ 0.15 and mean_err ≤ 0.08 in
    // every scene?
    bool anyGood = false;
    std::string bestName = "(none)";
    float bestTotalMean = 1e9f;
    for (size_t ci = 0; ci < candidates.size(); ++ci)
    {
        float tm = totalMean[ci] / scenes().size();
        if (totalMax[ci] <= 0.15f && tm <= 0.08f)
        {
            anyGood = true;
            if (tm < bestTotalMean)
            {
                bestTotalMean = tm;
                bestName = candidates[ci].name;
            }
        }
    }
    std::printf("\n[conclusion] best-candidate-under-threshold: %s\n", bestName.c_str());
    if (!anyGood)
        std::printf("[conclusion] No candidate meets threshold (max<=0.15, mean<=0.08).\n"
                    "             Likely requires HDR RT + post-process tonemap (Phase 11.3.3+).\n");
    // No hard PHYS_CHECK on anyGood — this test is a diagnostic tool; its output is the value
}

// ============================================================================
// Subtest 1: zero-light scenes must match exactly (ambient only)
// ============================================================================
PHYS_TEST(LightVolume, NoLightsIsIdentical)
{
    glm::vec3 ambient(0.03f);
    glm::vec3 a = pathFullscreen(ambient, {});
    glm::vec3 b = pathPerLight(ambient, {});
    PHYS_CHECK_VEC3_NEAR(a, b, 1e-5f);
}

// ============================================================================
// Subtest 2: single weak light → per_light error stays small (still amplified to
// ~0.16 by tonemap non-linearity). Historical reference only — Step 14 replaced
// the per_light scheme with the FixedPaths scheme.
// ============================================================================
PHYS_TEST(LightVolume, SingleWeakLightHistoricalRef)
{
    glm::vec3 ambient(0.03f);
    std::vector<glm::vec3> Lo = {glm::vec3(0.1f)};
    glm::vec3 a = pathFullscreen(ambient, Lo);
    glm::vec3 b = pathPerLight(ambient, Lo);
    Score s = diff(a, b);
    std::printf("  [info] single weak light max_err(per_light) = %.4f (historical)\n", s.maxErr);
    // No pass/fail assertion — the implementation now uses pathLVFixed
}

// ============================================================================
// Subtest 3: multi-light scene → the old per_light implementation was
// significantly brighter (Step 13 bug description); current approach in
// FixedPathsAreCloseEnough.
// ============================================================================
PHYS_TEST(LightVolume, MultiLightPerLightHistoricalBrighter)
{
    glm::vec3 ambient(0.03f);
    std::vector<glm::vec3> Lo = {
        glm::vec3(1.2f, 1.0f, 0.8f),
        glm::vec3(0.3f, 0.5f, 1.4f),
        glm::vec3(1.5f, 0.2f, 0.2f),
        glm::vec3(0.2f, 1.3f, 0.3f),
    };
    glm::vec3 a = pathFullscreen(ambient, Lo);
    glm::vec3 b = pathPerLight(ambient, Lo);
    float aLum = 0.299f * a.x + 0.587f * a.y + 0.114f * a.z;
    float bLum = 0.299f * b.x + 0.587f * b.y + 0.114f * b.z;
    std::printf("  [info] 4 lights luminance — fullscreen=%.3f, per_light(old)=%.3f (ratio=%.2fx)\n",
                aLum, bLum, bLum / std::max(aLum, 1e-4f));
    // Historical reference only, no PHYS_CHECK
}

// ============================================================================
// Subtest 4: fixed-path equivalence — pathFullscreenFixed vs pathLVFixed. Both
// paths drop manual gamma (hardware sRGB write); fullscreen reinhards the summed
// lights, LV reinhards per light then adds. Assert max_err ≤ 0.15 across all
// typical scenes (within UNORM8 clamp limits).
// ============================================================================
PHYS_TEST(LightVolume, FixedPathsAreCloseEnough)
{
    std::printf("\n[FixedPaths] fullscreen_fixed vs lv_fixed diff per scene:\n");
    float worstMax = 0.0f;
    float sumMean = 0.0f;
    int n = 0;
    for (const auto &sc : scenes())
    {
        glm::vec3 a = pathFullscreenFixed(sc.ambient, sc.Lo);
        glm::vec3 b = pathLVFixed(sc.ambient, sc.Lo);
        Score s = diff(a, b);
        worstMax = std::max(worstMax, s.maxErr);
        sumMean += s.meanErr;
        n++;
        std::printf("  %-30s max=%.4f mean=%.4f  fs=(%.3f,%.3f,%.3f) lv=(%.3f,%.3f,%.3f)\n",
                    sc.name, s.maxErr, s.meanErr, a.x, a.y, a.z, b.x, b.y, b.z);
    }
    std::printf("  [summary] worst_max=%.4f, avg_mean=%.4f\n", worstMax, sumMean / n);
    PHYS_CHECK(worstMax <= 0.15f, "fixed paths: max per-channel diff must be <= 0.15");
    PHYS_CHECK(sumMean / n <= 0.08f, "fixed paths: mean per-channel diff must be <= 0.08");
}

// ============================================================================
// Subtest 5: exposure must not break path equivalence (Step 15). Exposure is
// applied to Lo and ambient before reinhard; both paths must stay equivalent.
// Also verify exposure=3.0 clearly brightens the weak-light scene over 1.0.
// ============================================================================
namespace
{
    glm::vec3 pathFullscreenFixedExposed(const glm::vec3 &ambient,
                                         const std::vector<glm::vec3> &Lo,
                                         float exposure)
    {
        glm::vec3 sum = ambient;
        for (const auto &l : Lo)
            sum += l;
        sum *= exposure;
        return clamp01(gammaCorrect(clamp01(reinhard(sum))));
    }
    glm::vec3 pathLVFixedExposed(const glm::vec3 &ambient,
                                 const std::vector<glm::vec3> &Lo,
                                 float exposure)
    {
        glm::vec3 sum = reinhard(ambient * exposure);
        for (const auto &l : Lo)
            sum += reinhard(l * exposure);
        return clamp01(gammaCorrect(clamp01(sum)));
    }
} // namespace

PHYS_TEST(LightVolume, ExposurePreservesEquivalence)
{
    std::printf("\n[Exposure] paths at exposure=3.0:\n");
    float worstMax = 0.0f;
    float sumMean = 0.0f;
    int n = 0;
    for (const auto &sc : scenes())
    {
        glm::vec3 a = pathFullscreenFixedExposed(sc.ambient, sc.Lo, 3.0f);
        glm::vec3 b = pathLVFixedExposed(sc.ambient, sc.Lo, 3.0f);
        Score s = diff(a, b);
        worstMax = std::max(worstMax, s.maxErr);
        sumMean += s.meanErr;
        n++;
        std::printf("  %-30s max=%.4f mean=%.4f  fs=(%.3f,%.3f,%.3f) lv=(%.3f,%.3f,%.3f)\n",
                    sc.name, s.maxErr, s.meanErr, a.x, a.y, a.z, b.x, b.y, b.z);
    }
    std::printf("  [summary@3.0] worst_max=%.4f, avg_mean=%.4f\n", worstMax, sumMean / n);
    PHYS_CHECK(worstMax <= 0.20f, "exposure=3.0: max per-channel diff must be <= 0.20");

    // Verify exposure actually lifts weak-light visibility:
    // scene "single_weak_light": ambient=0.03, Lo≈0.2
    glm::vec3 lowExp = pathFullscreenFixedExposed(scenes()[0].ambient, scenes()[0].Lo, 1.0f);
    glm::vec3 highExp = pathFullscreenFixedExposed(scenes()[0].ambient, scenes()[0].Lo, 3.0f);
    float lum1 = 0.299f * lowExp.x + 0.587f * lowExp.y + 0.114f * lowExp.z;
    float lum3 = 0.299f * highExp.x + 0.587f * highExp.y + 0.114f * highExp.z;
    std::printf("  [info] single_weak_light luminance: exposure=1.0 -> %.3f, exposure=3.0 -> %.3f (gain=%.2fx)\n",
                lum1, lum3, lum3 / std::max(lum1, 1e-4f));
    PHYS_CHECK(lum3 > lum1 * 1.3f, "exposure=3.0 must be clearly brighter than 1.0 (>1.3x)");
}
