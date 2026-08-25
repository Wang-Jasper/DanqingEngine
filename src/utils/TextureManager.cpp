#include "utils/TextureManager.h"
#include "utils/VulkanUtils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cstring>

// ============================================================================
// createSampler — linear-filtered, anisotropic texture sampler
// ============================================================================
VkSampler TextureManager::createSampler()
{
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.anisotropyEnable = VK_TRUE;
    info.maxAnisotropy = 16.0f;
    info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;
    info.compareEnable = VK_FALSE;
    info.minLod = 0.0f;
    info.maxLod = 0.0f;

    VkSampler sampler;
    vkCreateSampler(context->getDevice(), &info, nullptr, &sampler);
    return sampler;
}

// ============================================================================
// createTextureFromPixelsImpl — internal impl (not added to cache / programmatic)
// ============================================================================
std::shared_ptr<TextureResource> TextureManager::createTextureFromPixelsImpl(
    const unsigned char *pixels, int width, int height,
    const std::string &debugName)
{
    VkDevice device = context->getDevice();
    VmaAllocator vma = allocator->getVma();
    VkDeviceSize imageSize = width * height * 4; // Always RGBA

    // Staging buffer
    AllocatedBuffer staging = allocator->createBuffer(
        imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void *mapped;
    vmaMapMemory(vma, staging.allocation, &mapped);
    memcpy(mapped, pixels, imageSize);
    vmaUnmapMemory(vma, staging.allocation);

    // Create the VkImage
    auto tex = std::make_shared<TextureResource>();
    tex->image.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    tex->image.format = VK_FORMAT_R8G8B8A8_SRGB;

    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(vma, &imgInfo, &allocCI, &tex->image.image, &tex->image.allocation, nullptr);

    // Transition to TRANSFER_DST, copy data, transition to SHADER_READ_ONLY
    VkCommandBuffer cmd = commandManager->beginSingleTimeCommands(device);

    VulkanUtils::transitionImageLayout(cmd, tex->image.image,
                                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};

    vkCmdCopyBufferToImage(cmd, staging.buffer, tex->image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VulkanUtils::transitionImageLayout(cmd, tex->image.image,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    commandManager->endSingleTimeCommands(device, context->getGraphicsQueue(), cmd);

    allocator->destroyBuffer(staging);

    // Create the image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex->image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &viewInfo, nullptr, &tex->image.imageView);

    // Create the sampler
    tex->sampler = createSampler();

    std::cout << "[TextureManager] Loaded texture: " << debugName
              << " (" << width << "x" << height << ")\n";
    return tex;
}

// ============================================================================
// loadTexture — load a texture from file (cached)
// ============================================================================
std::shared_ptr<TextureResource> TextureManager::loadTexture(const std::string &filepath)
{
    // Cache lookup
    auto it = cache.find(filepath);
    if (it != cache.end())
        return it->second;

    // Load via stb_image
    int width, height, channels;
    stbi_set_flip_vertically_on_load(false); // Vulkan UVs don't need flipping
    unsigned char *pixels = stbi_load(filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels)
    {
        std::cerr << "[TextureManager] Failed to load: " << filepath
                  << " — " << stbi_failure_reason() << "\n";
        return nullptr;
    }

    auto tex = createTextureFromPixelsImpl(pixels, width, height, filepath);
    stbi_image_free(pixels);

    cache[filepath] = tex;
    return tex;
}

// ============================================================================
// getDefaultWhiteTexture — 1x1 white texture (fallback when no map is bound)
// ============================================================================
std::shared_ptr<TextureResource> TextureManager::getDefaultWhiteTexture()
{
    if (defaultWhite)
        return defaultWhite;

    unsigned char white[] = {255, 255, 255, 255};
    defaultWhite = createTextureFromPixelsImpl(white, 1, 1, "default_white");
    return defaultWhite;
}

// ============================================================================
// createTextureFromPixels — public wrapper for programmatic textures
// ============================================================================
// Caller supplies RGBA8 pixels; the texture is pushed to the `programmatic`
// list so cleanup() frees it too (no path key).
std::shared_ptr<TextureResource> TextureManager::createTextureFromPixels(
    const unsigned char *pixels, int width, int height,
    const std::string &debugName)
{
    auto tex = createTextureFromPixelsImpl(pixels, width, height, debugName);
    if (tex)
        programmatic.push_back(tex);
    return tex;
}

// ============================================================================
// cleanup — destroy all texture resources
// ============================================================================
void TextureManager::cleanup()
{
    VkDevice device = context->getDevice();
    VmaAllocator vma = allocator->getVma();

    auto destroyTex = [&](std::shared_ptr<TextureResource> &tex)
    {
        if (!tex)
            return;
        if (tex->sampler)
            vkDestroySampler(device, tex->sampler, nullptr);
        if (tex->image.imageView)
            vkDestroyImageView(device, tex->image.imageView, nullptr);
        if (tex->image.image)
            vmaDestroyImage(vma, tex->image.image, tex->image.allocation);
    };

    for (auto &[path, tex] : cache)
    {
        destroyTex(tex);
    }
    cache.clear();

    // Free programmatic textures (benchmark / tests)
    for (auto &tex : programmatic)
    {
        destroyTex(tex);
    }
    programmatic.clear();

    destroyTex(defaultWhite);
    defaultWhite = nullptr;
}
