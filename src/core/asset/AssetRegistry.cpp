// ============================================================================
// AssetRegistry.cpp — startup scan + cache implementation
// ----------------------------------------------------------------------------
// Startup scan strategy:
//
//     loadCache(.registry.cache)                      // ok to be absent
//     for each file under assetsRoot recursively:
//         if !isAssetFile → skip
//         mt = mtime(file)
//         if cache hit && mtime matches → register (guid, relPath) directly
//         else → read .meta (generate+write if missing) → register → mark dirty
//     detect deletions: cache entries not seen on disk → mark dirty
//     if dirty → saveCache
//
// On a 200-asset tree this runs in a few tens of milliseconds on first boot
// (dominated by .meta generation for brand-new files), and 2-5ms on
// subsequent boots (one stat per file + a single JSON read/write).
// ============================================================================

#include "core/asset/AssetRegistry.h"
#include "core/asset/AssetTypes.h"
#include "core/asset/MetaFile.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_set>

namespace
{
    constexpr const char *kCacheFileName = ".registry.cache";
    constexpr int kCacheVersion = 1;
}

AssetRegistry &AssetRegistry::instance()
{
    static AssetRegistry s;
    return s;
}

std::string AssetRegistry::toForwardSlash(std::string s)
{
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
}

std::int64_t AssetRegistry::getMtimeNs(const std::filesystem::path &p)
{
    std::error_code ec;
    auto ft = std::filesystem::last_write_time(p, ec);
    if (ec)
        return 0;
    // Convert whatever clock file_time_type uses into a stable 64-bit integer.
    // We don't care about the absolute epoch, only whether two values match.
    return static_cast<std::int64_t>(ft.time_since_epoch().count());
}

std::filesystem::path AssetRegistry::cachePath() const
{
    return assetsRoot_ / kCacheFileName;
}

std::filesystem::path AssetRegistry::resolveAbsolute(const std::string &relPath) const
{
    return assetsRoot_ / std::filesystem::path(relPath);
}

std::optional<std::string> AssetRegistry::toRelativeKey(const std::filesystem::path &absOrRelPath) const
{
    std::error_code ec;
    std::filesystem::path abs = absOrRelPath;
    if (abs.is_relative())
        abs = std::filesystem::absolute(abs, ec);
    abs = std::filesystem::weakly_canonical(abs, ec);
    auto rootCanon = std::filesystem::weakly_canonical(assetsRoot_, ec);

    auto rel = std::filesystem::relative(abs, rootCanon, ec);
    if (ec || rel.empty())
        return std::nullopt;
    std::string s = toForwardSlash(rel.generic_string());
    if (s.rfind("..", 0) == 0)
        return std::nullopt; // path escapes assets root
    return s;
}

// ----------------------------------------------------------------------------
// Cache IO
// ----------------------------------------------------------------------------
bool AssetRegistry::loadCache(std::unordered_map<std::string, CacheEntry> &byPathOut) const
{
    const auto cp = cachePath();
    std::error_code ec;
    if (!std::filesystem::exists(cp, ec))
        return false;

    std::ifstream ifs(cp);
    if (!ifs.is_open())
        return false;

    try
    {
        nlohmann::json j;
        ifs >> j;
        if (j.value("version", 0) != kCacheVersion)
            return false; // future-proofing: version bump invalidates cache

        for (const auto &e : j["entries"])
        {
            CacheEntry ce;
            ce.guid = e.value("guid", "");
            ce.relPath = e.value("path", "");
            ce.mtimeNs = e.value("mtimeNs", std::int64_t(0));
            if (ce.guid.empty() || ce.relPath.empty())
                continue;
            byPathOut[ce.relPath] = std::move(ce);
        }
        return true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[AssetRegistry] Cache parse failed: " << ex.what() << "\n";
        return false;
    }
}

bool AssetRegistry::saveCache(const std::unordered_map<std::string, CacheEntry> &byPath) const
{
    const auto cp = cachePath();
    auto tmp = cp;
    tmp += ".tmp";

    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open())
            return false;
        nlohmann::json j;
        j["version"] = kCacheVersion;
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &[path, e] : byPath)
        {
            nlohmann::json item;
            item["guid"] = e.guid;
            item["path"] = e.relPath;
            item["mtimeNs"] = e.mtimeNs;
            arr.push_back(std::move(item));
        }
        j["entries"] = std::move(arr);
        ofs << j.dump(2);
    }

    std::error_code ec;
    std::filesystem::rename(tmp, cp, ec);
    if (ec)
    {
        std::filesystem::remove(cp, ec);
        std::filesystem::rename(tmp, cp, ec);
    }
    return !ec;
}

// ----------------------------------------------------------------------------
// initialize / shutdown
// ----------------------------------------------------------------------------
void AssetRegistry::initialize(const std::filesystem::path &assetsRoot)
{
    if (initialized_)
        return;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::error_code ec;
    assetsRoot_ = std::filesystem::absolute(assetsRoot, ec);
    if (ec || !std::filesystem::exists(assetsRoot_, ec) || !std::filesystem::is_directory(assetsRoot_, ec))
    {
        std::cerr << "[AssetRegistry] Invalid assets root: " << assetsRoot.string() << "\n";
        return;
    }

    std::unordered_map<std::string, CacheEntry> cacheByPath;
    const bool haveCache = loadCache(cacheByPath);

    std::unordered_map<std::string, CacheEntry> freshByPath;
    std::unordered_set<std::string> seenPaths;
    bool dirty = false;
    int hits = 0, misses = 0, generated = 0;

    for (auto it = std::filesystem::recursive_directory_iterator(assetsRoot_, ec);
         !ec && it != std::filesystem::recursive_directory_iterator();
         it.increment(ec))
    {
        const auto &p = it->path();
        if (!AssetTypes::isAssetFile(p))
            continue;

        auto relOpt = toRelativeKey(p);
        if (!relOpt)
            continue;
        const std::string &rel = *relOpt;
        seenPaths.insert(rel);

        const std::int64_t mt = getMtimeNs(p);

        auto cIt = cacheByPath.find(rel);
        if (cIt != cacheByPath.end() && cIt->second.mtimeNs == mt && !cIt->second.guid.empty())
        {
            // Cache hit — still verify .meta exists (paranoia: someone may have
            // deleted the .meta while leaving the asset). If the meta is
            // missing we fall through to the "miss" branch so a fresh one is
            // written with the SAME guid (we already have it in the cache).
            std::error_code mec;
            if (std::filesystem::exists(MetaFile::metaPathFor(p), mec))
            {
                freshByPath[rel] = cIt->second;
                guidToRel_[cIt->second.guid] = rel;
                relToGuid_[rel] = cIt->second.guid;
                ++hits;
                continue;
            }
        }

        // Cache miss or mtime mismatch — read .meta or generate.
        MetaFile::MetaInfo info;
        if (auto m = MetaFile::read(p))
        {
            info = *m;
        }
        else
        {
            // Preserve the cached guid if we had one (meta was deleted by mistake).
            info.guid = (cIt != cacheByPath.end() && !cIt->second.guid.empty())
                            ? cIt->second.guid
                            : MetaFile::generateGuid();
            info.version = 1;
            MetaFile::write(p, info);
            ++generated;
        }

        CacheEntry ce;
        ce.guid = info.guid;
        ce.relPath = rel;
        ce.mtimeNs = mt;
        freshByPath[rel] = ce;
        guidToRel_[info.guid] = rel;
        relToGuid_[rel] = info.guid;
        ++misses;
        dirty = true;
    }

    // Detect deletions (cache had entries that are no longer on disk).
    for (const auto &[rel, _] : cacheByPath)
    {
        if (!seenPaths.count(rel))
        {
            dirty = true;
            break;
        }
    }

    if (!haveCache || dirty)
        saveCache(freshByPath);

    initialized_ = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[AssetRegistry] Initialized in " << ms << " ms"
              << " (root=" << assetsRoot_.string() << ", assets=" << freshByPath.size()
              << ", cacheHits=" << hits << ", metaReads=" << misses
              << ", newMetas=" << generated << ")\n";
}

void AssetRegistry::shutdown()
{
    std::lock_guard<std::mutex> lk(mutex_);
    guidToRel_.clear();
    relToGuid_.clear();
    assetsRoot_.clear();
    initialized_ = false;
}

// ----------------------------------------------------------------------------
// Query
// ----------------------------------------------------------------------------
std::optional<std::filesystem::path> AssetRegistry::guidToPath(const std::string &guid) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = guidToRel_.find(guid);
    if (it == guidToRel_.end())
        return std::nullopt;
    return resolveAbsolute(it->second);
}

std::optional<std::string> AssetRegistry::pathToGuid(const std::filesystem::path &absOrRelPath) const
{
    auto relOpt = toRelativeKey(absOrRelPath);
    if (!relOpt)
        return std::nullopt;
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = relToGuid_.find(*relOpt);
    if (it == relToGuid_.end())
        return std::nullopt;
    return it->second;
}

// ----------------------------------------------------------------------------
// registerNewAsset — used by the import pipeline
// ----------------------------------------------------------------------------
std::string AssetRegistry::registerNewAsset(const std::filesystem::path &absPath)
{
    if (!initialized_)
    {
        std::cerr << "[AssetRegistry] registerNewAsset called before initialize()\n";
        return {};
    }

    std::error_code ec;
    if (!std::filesystem::exists(absPath, ec))
    {
        std::cerr << "[AssetRegistry] registerNewAsset: file does not exist: "
                  << absPath.string() << "\n";
        return {};
    }

    auto relOpt = toRelativeKey(absPath);
    if (!relOpt)
    {
        std::cerr << "[AssetRegistry] registerNewAsset: path is outside assets root ("
                  << assetsRoot_.string() << "): " << absPath.string() << "\n";
        // Fallback: the user imported something from outside the project — we
        // cannot turn that into a stable GUID reference. Return empty so
        // callers can decide (usually they'll fall back to the raw path and
        // log a warning).
        return {};
    }
    const std::string &rel = *relOpt;

    // Fast path: already known.
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = relToGuid_.find(rel);
        if (it != relToGuid_.end())
            return it->second;
    }

    // Load or create the .meta sidecar.
    MetaFile::MetaInfo info;
    if (auto m = MetaFile::read(absPath))
    {
        info = *m;
    }
    else
    {
        info.guid = MetaFile::generateGuid();
        info.version = 1;
        MetaFile::write(absPath, info);
    }

    {
        std::lock_guard<std::mutex> lk(mutex_);
        guidToRel_[info.guid] = rel;
        relToGuid_[rel] = info.guid;
    }

    // Persist the cache incrementally so a crash before the next startup
    // doesn't force the (slower) meta-reading path on every file.
    std::unordered_map<std::string, CacheEntry> all;
    loadCache(all); // ok if it fails, we'll just start fresh
    CacheEntry ce;
    ce.guid = info.guid;
    ce.relPath = rel;
    ce.mtimeNs = getMtimeNs(absPath);
    all[rel] = ce;
    saveCache(all);

    return info.guid;
}
