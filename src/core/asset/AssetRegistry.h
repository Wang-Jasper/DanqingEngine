#pragma once

// ============================================================================
// AssetRegistry.h — GUID <-> asset path bidirectional registry
// ----------------------------------------------------------------------------
// Design in a nutshell:
//
//   - `initialize(assetsRoot)` performs a startup scan. Paths are stored
//     relative to `assetsRoot` using forward slashes so they remain stable
//     across OS / drive moves.
//   - For every asset file we read its `<asset>.meta` sidecar (or generate
//     one on first sight). The resulting GUID uniquely identifies the asset
//     for the entire lifetime of the project — renaming / moving the pair
//     does not invalidate references.
//   - A simple JSON cache `<assetsRoot>/.registry.cache` accelerates
//     subsequent startups: we only re-read `.meta` files whose mtime no
//     longer matches the cache entry.
//
// Runtime API:
//   - `guidToPath(guid)`  -> absolute path on disk (std::nullopt if unknown)
//   - `pathToGuid(path)`  -> guid string              (std::nullopt if unknown)
//   - `registerNewAsset(absPath)` is used by the import pipeline to admit
//     a freshly-introduced asset into the registry.
//
// The class is used as a process-wide singleton via `AssetRegistry::instance()`
// because asset identity is inherently a global concept.
// ============================================================================

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

class AssetRegistry
{
public:
    static AssetRegistry &instance();

    // Called once at engine startup. Idempotent. After this returns the
    // registry contains every asset currently under `assetsRoot`.
    void initialize(const std::filesystem::path &assetsRoot);

    // Releases internal state (mainly for clean test teardown).
    void shutdown();

    bool isInitialized() const { return initialized_; }
    const std::filesystem::path &assetsRoot() const { return assetsRoot_; }

    // Query. Read-only and thread-safe once initialize() has returned.
    std::optional<std::filesystem::path> guidToPath(const std::string &guid) const;
    std::optional<std::string> pathToGuid(const std::filesystem::path &absOrRelPath) const;

    // Admit a freshly-imported asset. If the file already has a .meta we
    // reuse the stored GUID; otherwise we generate one and persist the .meta
    // next to the asset. Returns the GUID on success, empty string on failure.
    std::string registerNewAsset(const std::filesystem::path &absPath);

    // Convenience: normalise an arbitrary path to an assets-root-relative,
    // forward-slash string. Paths outside `assetsRoot` return std::nullopt.
    std::optional<std::string> toRelativeKey(const std::filesystem::path &absOrRelPath) const;

private:
    AssetRegistry() = default;
    AssetRegistry(const AssetRegistry &) = delete;
    AssetRegistry &operator=(const AssetRegistry &) = delete;

    struct CacheEntry
    {
        std::string guid;
        std::string relPath; // relative to assetsRoot_, forward-slash
        std::int64_t mtimeNs = 0;
    };

    // Cache file IO. `<assetsRoot>/.registry.cache`, JSON.
    bool loadCache(std::unordered_map<std::string, CacheEntry> &byPathOut) const;
    bool saveCache(const std::unordered_map<std::string, CacheEntry> &byPath) const;
    std::filesystem::path cachePath() const;

    // Helpers
    std::filesystem::path resolveAbsolute(const std::string &relPath) const;
    static std::string toForwardSlash(std::string s);
    static std::int64_t getMtimeNs(const std::filesystem::path &p);

private:
    std::filesystem::path assetsRoot_;
    bool initialized_ = false;

    // Both maps are populated during initialize() and then effectively read-only.
    // registerNewAsset() may mutate them; we protect those rare writes with a mutex.
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> guidToRel_; // guid -> rel path
    std::unordered_map<std::string, std::string> relToGuid_; // rel path -> guid
};
