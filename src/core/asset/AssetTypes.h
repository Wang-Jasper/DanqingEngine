#pragma once

// ============================================================================
// AssetTypes.h — asset file classification
// ----------------------------------------------------------------------------
// Centralises the list of asset file extensions that participate in the GUID
// registry. Non-asset files (e.g. scene .json, .mtl, .meta, .registry.cache)
// are explicitly excluded so they are never scanned or assigned a GUID.
// ============================================================================

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

namespace AssetTypes
{
    inline std::string lowerExt(const std::filesystem::path &p)
    {
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return ext;
    }

    // Returns true if the file is an asset we want to track (Mesh / Texture / ...).
    // Intentionally excludes: .json (scene), .mtl (material sidecar), .meta (our own sidecar),
    // .cache (our own registry cache), and any hidden dotfiles.
    inline bool isAssetFile(const std::filesystem::path &p)
    {
        if (!std::filesystem::is_regular_file(p))
            return false;

        const std::string filename = p.filename().string();
        if (filename.empty() || filename[0] == '.')
            return false; // hidden / dotfile (e.g. .registry.cache, .DS_Store)

        const std::string ext = lowerExt(p);

        // Meshes
        if (ext == ".obj" || ext == ".fbx" || ext == ".gltf" || ext == ".glb")
            return true;
        // Textures
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" ||
            ext == ".bmp" || ext == ".hdr" || ext == ".exr")
            return true;

        return false;
    }

    inline bool isMetaFile(const std::filesystem::path &p)
    {
        return lowerExt(p) == ".meta";
    }
}
