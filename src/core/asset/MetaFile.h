#pragma once

// ============================================================================
// MetaFile.h — GUID sidecar (.meta) read / write + UUIDv4 generation
// ----------------------------------------------------------------------------
// For every tracked asset `bunny.obj` we maintain a companion `bunny.obj.meta`
// file storing the asset's stable GUID. This sidecar is what makes GUIDs
// survive renames / moves: as long as the pair travels together, the registry
// can re-bind the GUID to whatever new path the asset appears at.
//
// Format (minimal JSON, 1 line-friendly):
//   { "version": 1, "guid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890" }
// ============================================================================

#include <filesystem>
#include <optional>
#include <string>

namespace MetaFile
{
    struct MetaInfo
    {
        std::string guid;
        int version = 1;
    };

    // Returns the expected .meta path for an asset path. E.g.
    //   "assets/stanford-bunny.obj" -> "assets/stanford-bunny.obj.meta"
    std::filesystem::path metaPathFor(const std::filesystem::path &assetPath);

    // Attempt to read <assetPath>.meta. Returns nullopt if the file doesn't
    // exist or fails to parse (the caller is expected to regenerate it).
    std::optional<MetaInfo> read(const std::filesystem::path &assetPath);

    // Atomically write <assetPath>.meta. Uses a "write temp + rename" pattern
    // so a crash mid-write never leaves a truncated .meta on disk.
    bool write(const std::filesystem::path &assetPath, const MetaInfo &info);

    // Generates a fresh UUIDv4 string, e.g. "a1b2c3d4-e5f6-4890-abcd-ef1234567890".
    // Uses std::random_device + std::mt19937_64, no external uuid library needed.
    std::string generateGuid();
}
