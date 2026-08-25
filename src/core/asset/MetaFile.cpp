// ============================================================================
// MetaFile.cpp — .meta read/write + UUIDv4 generation
// ============================================================================

#include "core/asset/MetaFile.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

namespace MetaFile
{
    std::filesystem::path metaPathFor(const std::filesystem::path &assetPath)
    {
        // NOTE: we do NOT use path::replace_extension here because we want
        // "foo.obj" -> "foo.obj.meta" (not "foo.meta"), matching Unity's
        // convention so the .meta stays adjacent even for multi-extension files.
        std::filesystem::path p = assetPath;
        p += ".meta";
        return p;
    }

    std::optional<MetaInfo> read(const std::filesystem::path &assetPath)
    {
        const auto mp = metaPathFor(assetPath);
        std::error_code ec;
        if (!std::filesystem::exists(mp, ec))
            return std::nullopt;

        std::ifstream ifs(mp);
        if (!ifs.is_open())
            return std::nullopt;

        try
        {
            nlohmann::json j;
            ifs >> j;

            MetaInfo info;
            info.version = j.value("version", 1);
            info.guid = j.value("guid", "");
            if (info.guid.empty())
                return std::nullopt; // treat malformed meta as missing so caller regenerates
            return info;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[MetaFile] Failed to parse " << mp.string()
                      << ": " << e.what() << "\n";
            return std::nullopt;
        }
    }

    bool write(const std::filesystem::path &assetPath, const MetaInfo &info)
    {
        const auto mp = metaPathFor(assetPath);
        auto tmp = mp;
        tmp += ".tmp";

        {
            std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
            if (!ofs.is_open())
            {
                std::cerr << "[MetaFile] Failed to open for writing: " << tmp.string() << "\n";
                return false;
            }
            nlohmann::json j;
            j["version"] = info.version;
            j["guid"] = info.guid;
            ofs << j.dump(2);
        }

        std::error_code ec;
        std::filesystem::rename(tmp, mp, ec);
        if (ec)
        {
            // On Windows, rename over an existing file may fail. Fall back to
            // remove + rename.
            std::filesystem::remove(mp, ec);
            std::filesystem::rename(tmp, mp, ec);
        }
        if (ec)
        {
            std::cerr << "[MetaFile] Failed to rename " << tmp.string()
                      << " -> " << mp.string() << ": " << ec.message() << "\n";
            return false;
        }
        return true;
    }

    std::string generateGuid()
    {
        // RFC 4122 v4: 128 random bits with a few fixed bits for version/variant.
        static thread_local std::mt19937_64 rng{std::random_device{}()};

        std::uint64_t hi = rng();
        std::uint64_t lo = rng();

        // Set version to 4 (random)
        hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
        // Set variant to RFC 4122 (10xxxxxx)
        lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

        std::array<std::uint8_t, 16> b{};
        for (int i = 0; i < 8; ++i)
        {
            b[i] = static_cast<std::uint8_t>((hi >> ((7 - i) * 8)) & 0xFF);
            b[8 + i] = static_cast<std::uint8_t>((lo >> ((7 - i) * 8)) & 0xFF);
        }

        char buf[37];
        std::snprintf(buf, sizeof(buf),
                      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                      b[0], b[1], b[2], b[3],
                      b[4], b[5],
                      b[6], b[7],
                      b[8], b[9],
                      b[10], b[11], b[12], b[13], b[14], b[15]);
        return std::string(buf);
    }
}
