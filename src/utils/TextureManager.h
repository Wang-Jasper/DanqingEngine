#pragma once

// TextureManager — stb_image loading -> VkImage + VkImageView + VkSampler
// For PBR material maps (Albedo / Normal / Metallic-Roughness, etc.)

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "utils/DepthUtils.h" // AllocatedImage
#include "core/VulkanContext.h"
#include "core/Allocator.h"
#include "core/CommandManager.h"

#include <string>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <vector>

struct TextureResource
{
    AllocatedImage image;
    VkSampler sampler = VK_NULL_HANDLE;
};

class TextureManager
{
public:
    void init(VulkanContext *ctx, Allocator *alloc, CommandManager *cmdMgr)
    {
        context = ctx;
        allocator = alloc;
        commandManager = cmdMgr;
    }

    std::shared_ptr<TextureResource> loadTexture(const std::string &filepath);
    std::shared_ptr<TextureResource> getDefaultWhiteTexture();

    // Programmatic textures for benchmark presets / tests. pixels must point to
    // width*height RGBA8 bytes. Texture is created with VK_FORMAT_R8G8B8A8_SRGB
    // and stays alive until cleanup() (added to a private list, not the
    // path-keyed cache).
    std::shared_ptr<TextureResource> createTextureFromPixels(
        const unsigned char *pixels, int width, int height,
        const std::string &debugName);

    void cleanup();

private:
    VulkanContext *context = nullptr;
    Allocator *allocator = nullptr;
    CommandManager *commandManager = nullptr;

    std::unordered_map<std::string, std::shared_ptr<TextureResource>> cache;
    std::shared_ptr<TextureResource> defaultWhite;
    // Textures created via the public createTextureFromPixels() (e.g.
    // benchmark/programmatic textures) live here so cleanup() can free them
    // too, without needing a filesystem key.
    std::vector<std::shared_ptr<TextureResource>> programmatic;

    // Internal impl shared by loadTexture() / getDefaultWhiteTexture() and
    // the public createTextureFromPixels(); does NOT push into `programmatic`.
    std::shared_ptr<TextureResource> createTextureFromPixelsImpl(
        const unsigned char *pixels, int width, int height,
        const std::string &debugName);

    VkSampler createSampler();
};