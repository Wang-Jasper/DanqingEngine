// Minimal self-contained assertion framework (no GoogleTest).
// PHYS_TEST registers a case; PHYS_CHECK* record failures; RunAllTests() runs
// them all, prints [RUN]/[OK]/[FAIL] lines, and returns the failure count.

#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace phys
{
    struct TestFailure
    {
        std::string file;
        int line = 0;
        std::string cond;
        std::string msg;
    };

    struct TestCase
    {
        std::string suite;
        std::string name;
        std::function<void(std::vector<TestFailure> &)> fn;
    };

    // Global registry (Meyers singleton).
    inline std::vector<TestCase> &GetRegistry()
    {
        static std::vector<TestCase> reg;
        return reg;
    }

    struct AutoRegister
    {
        AutoRegister(const char *suite, const char *name,
                     std::function<void(std::vector<TestFailure> &)> fn)
        {
            TestCase tc;
            tc.suite = suite;
            tc.name = name;
            tc.fn = std::move(fn);
            GetRegistry().push_back(std::move(tc));
        }
    };

    // Pointer to the active failure list (a plain global for simplicity).
    inline std::vector<TestFailure> *&CurrentFailures()
    {
        static std::vector<TestFailure> *ptr = nullptr;
        return ptr;
    }

    inline int &CurrentCheckCount()
    {
        static int cnt = 0;
        return cnt;
    }

    inline void ReportFailure(const char *file, int line,
                              const char *cond, const std::string &msg)
    {
        if (!CurrentFailures())
            return;
        TestFailure f;
        f.file = file;
        f.line = line;
        f.cond = cond;
        f.msg = msg;
        CurrentFailures()->push_back(std::move(f));
    }

    // Returns the number of failed cases.
    inline int RunAllTests()
    {
        auto &reg = GetRegistry();
        int passed = 0;
        int failed = 0;
        std::printf("Running %d physics tests...\n\n", (int)reg.size());
        for (auto &tc : reg)
        {
            std::vector<TestFailure> failures;
            CurrentFailures() = &failures;
            CurrentCheckCount() = 0;
            std::printf("[RUN ] %s.%s\n", tc.suite.c_str(), tc.name.c_str());
            auto t0 = std::chrono::high_resolution_clock::now();
            try
            {
                tc.fn(failures);
            }
            catch (const std::exception &e)
            {
                TestFailure f;
                f.file = "(exception)";
                f.msg = std::string("threw std::exception: ") + e.what();
                failures.push_back(std::move(f));
            }
            catch (...)
            {
                TestFailure f;
                f.file = "(exception)";
                f.msg = "threw unknown exception";
                failures.push_back(std::move(f));
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            CurrentFailures() = nullptr;
            if (failures.empty())
            {
                std::printf("[ OK ] %s.%s  (%d checks, %.2fms)\n\n",
                            tc.suite.c_str(), tc.name.c_str(),
                            CurrentCheckCount(), ms);
                ++passed;
            }
            else
            {
                std::printf("[FAIL] %s.%s  (%d checks, %.2fms)\n",
                            tc.suite.c_str(), tc.name.c_str(),
                            CurrentCheckCount(), ms);
                for (const auto &f : failures)
                {
                    std::printf("  at %s:%d  cond=%s\n",
                                f.file.c_str(), f.line, f.cond.c_str());
                    if (!f.msg.empty())
                        std::printf("    msg=%s\n", f.msg.c_str());
                }
                std::printf("\n");
                ++failed;
            }
        }
        std::printf("== SUMMARY ==\n");
        std::printf("TOTAL: %d    PASSED: %d    FAILED: %d\n",
                    (int)reg.size(), passed, failed);
        if (failed == 0)
            std::printf("ALL PASSED\n");
        else
            std::printf("FAILED\n");
        return failed;
    }

    // ----- Float comparison helpers -----
    inline bool NearScalar(float a, float b, float eps)
    {
        return std::fabs(a - b) <= eps;
    }
    inline bool NearVec3(const glm::vec3 &a, const glm::vec3 &b, float eps)
    {
        return NearScalar(a.x, b.x, eps) &&
               NearScalar(a.y, b.y, eps) &&
               NearScalar(a.z, b.z, eps);
    }

    inline std::string Vec3ToStr(const glm::vec3 &v)
    {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "(%.5f,%.5f,%.5f)", v.x, v.y, v.z);
        return buf;
    }

} // namespace phys

// Public test macros.
#define PHYS_TEST(SUITE, NAME)                                               \
    static void phystest_##SUITE##_##NAME(std::vector<phys::TestFailure> &); \
    static phys::AutoRegister phystest_reg_##SUITE##_##NAME(                 \
        #SUITE, #NAME, &phystest_##SUITE##_##NAME);                          \
    static void phystest_##SUITE##_##NAME(std::vector<phys::TestFailure> &_phys_failures_unused)

#define PHYS_CHECK(COND, ...)                                     \
    do                                                            \
    {                                                             \
        ++phys::CurrentCheckCount();                              \
        if (!(COND))                                              \
        {                                                         \
            std::string _msg;                                     \
            const char *_argmsg = "" __VA_ARGS__;                 \
            if (_argmsg && _argmsg[0] != '\0')                    \
                _msg = _argmsg;                                   \
            phys::ReportFailure(__FILE__, __LINE__, #COND, _msg); \
        }                                                         \
    } while (0)

#define PHYS_CHECK_NEAR(A, B, EPS)                                       \
    do                                                                   \
    {                                                                    \
        ++phys::CurrentCheckCount();                                     \
        float _a = static_cast<float>(A);                                \
        float _b = static_cast<float>(B);                                \
        if (!phys::NearScalar(_a, _b, static_cast<float>(EPS)))          \
        {                                                                \
            char _buf[160];                                              \
            std::snprintf(_buf, sizeof(_buf),                            \
                          "expected %.6f, got %.6f (eps=%.6f)",          \
                          _b, _a, (double)(EPS));                        \
            phys::ReportFailure(__FILE__, __LINE__, #A " ~= " #B, _buf); \
        }                                                                \
    } while (0)

#define PHYS_CHECK_VEC3_NEAR(A, B, EPS)                                  \
    do                                                                   \
    {                                                                    \
        ++phys::CurrentCheckCount();                                     \
        glm::vec3 _a = (A);                                              \
        glm::vec3 _b = (B);                                              \
        if (!phys::NearVec3(_a, _b, static_cast<float>(EPS)))            \
        {                                                                \
            std::string _msg = "expected " + phys::Vec3ToStr(_b) +       \
                               ", got " + phys::Vec3ToStr(_a);           \
            phys::ReportFailure(__FILE__, __LINE__, #A " ~= " #B, _msg); \
        }                                                                \
    } while (0)

#define PHYS_CHECK_EQ(A, B)                                            \
    do                                                                 \
    {                                                                  \
        ++phys::CurrentCheckCount();                                   \
        if (!((A) == (B)))                                             \
        {                                                              \
            phys::ReportFailure(__FILE__, __LINE__, #A " == " #B, ""); \
        }                                                              \
    } while (0)
