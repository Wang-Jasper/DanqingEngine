#include "editor/EditorUI.h"
#include "renderer/Renderer.h"
#include "ecs/SceneSerializer.h"
#include "ecs/Systems.h"
#include "scene/SceneSetup.h"
#include "physics/PhysicsComponents.h"
#include "ecs/PhysicsSystem.h"
#include "core/asset/AssetRegistry.h"
#include "scripting/ScriptEngine.h" // Play/Stop callbacks
#include "tinyfiledialogs.h"
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cstring> // std::memcmp for Script Console filter

// ============================================================================
// Play / Pause / Stop icon buttons drawn with ImDrawList (zero font dependency).
// ============================================================================
namespace
{

    enum class PlayIconKind
    {
        Play,
        Pause,
        Stop,
    };

    // Draw button background (rect + theme color + rounded corners); returns icon color, dimmed when disabled
    inline ImU32 DrawIconButtonBackground(ImDrawList *dl,
                                          const ImVec2 &bbMin, const ImVec2 &bbMax,
                                          bool hovered, bool held, bool active, bool disabled)
    {
        ImU32 bg;
        if (disabled)
            bg = ImGui::GetColorU32(ImGuiCol_Button, 0.5f);
        else if (active)
            bg = ImGui::GetColorU32(ImGuiCol_ButtonActive);
        else if (held)
            bg = ImGui::GetColorU32(ImGuiCol_ButtonActive);
        else if (hovered)
            bg = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
        else
            bg = ImGui::GetColorU32(ImGuiCol_Button);
        dl->AddRectFilled(bbMin, bbMax, bg, ImGui::GetStyle().FrameRounding);

        return disabled ? ImGui::GetColorU32(ImGuiCol_TextDisabled)
                        : ImGui::GetColorU32(ImGuiCol_Text);
    }

    // Generic icon button: geometry chosen by kind
    static bool IconButton(const char *str_id, PlayIconKind kind,
                           bool active = false, bool disabled = false)
    {
        const float size = ImGui::GetFrameHeight();
        const ImVec2 btnSize(size, size);

        ImGui::PushID(str_id);

        bool pressed = false;
        if (disabled)
        {
            ImGui::BeginDisabled();
            ImGui::InvisibleButton("btn", btnSize);
            ImGui::EndDisabled();
        }
        else
        {
            pressed = ImGui::InvisibleButton("btn", btnSize);
        }

        const bool hovered = ImGui::IsItemHovered();
        const bool held = ImGui::IsItemActive();
        const ImVec2 bbMin = ImGui::GetItemRectMin();
        const ImVec2 bbMax = ImGui::GetItemRectMax();
        const ImVec2 center((bbMin.x + bbMax.x) * 0.5f, (bbMin.y + bbMax.y) * 0.5f);
        const float s = bbMax.x - bbMin.x; // assumes square

        ImDrawList *dl = ImGui::GetWindowDrawList();
        ImU32 iconCol = DrawIconButtonBackground(dl, bbMin, bbMax, hovered, held, active, disabled);

        switch (kind)
        {
        case PlayIconKind::Play:
        {
            // Isosceles triangle, apex pointing right; nudged slightly left of center
            const float halfW = s * 0.25f;
            const float halfH = s * 0.30f;
            const ImVec2 p0(center.x - halfW * 0.6f, center.y - halfH);
            const ImVec2 p1(center.x - halfW * 0.6f, center.y + halfH);
            const ImVec2 p2(center.x + halfW * 1.1f, center.y);
            dl->AddTriangleFilled(p0, p1, p2, iconCol);
            break;
        }
        case PlayIconKind::Pause:
        {
            // Two vertical bars: width 0.18*s, height 0.55*s, gap 0.14*s
            const float barW = s * 0.16f;
            const float barH = s * 0.55f;
            const float gap = s * 0.12f;
            ImVec2 lMin(center.x - gap * 0.5f - barW, center.y - barH * 0.5f);
            ImVec2 lMax(center.x - gap * 0.5f, center.y + barH * 0.5f);
            ImVec2 rMin(center.x + gap * 0.5f, center.y - barH * 0.5f);
            ImVec2 rMax(center.x + gap * 0.5f + barW, center.y + barH * 0.5f);
            dl->AddRectFilled(lMin, lMax, iconCol, 1.0f);
            dl->AddRectFilled(rMin, rMax, iconCol, 1.0f);
            break;
        }
        case PlayIconKind::Stop:
        {
            const float half = s * 0.28f;
            dl->AddRectFilled(ImVec2(center.x - half, center.y - half),
                              ImVec2(center.x + half, center.y + half),
                              iconCol, 1.0f);
            break;
        }
        }

        ImGui::PopID();
        return pressed;
    }

} // namespace
// ============================================================================

void EditorUI::init(Renderer &r)
{
    renderer = &r;
}

void EditorUI::initImGui()
{
    VkDevice device = renderer->vulkanContext.getDevice();

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100},
    };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.maxSets = 100;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &imguiDescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create ImGui descriptor pool!");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // --- Consolas as main UI font (Windows system font, monospaced) ---
    //   ImGui 1.92 dynamic font system: loaded once, scaled via FontScaleDpi
    //   15px base ≈ original ProggyClean (13px) at 100% DPI but sharper
    //   Falls back to the built-in font if loading fails (non-Windows / font file removed)
    {
        const char *kConsolasPath = "C:\\Windows\\Fonts\\consola.ttf";
        ImFont *font = io.Fonts->AddFontFromFileTTF(kConsolasPath, 15.0f);
        if (!font)
        {
            std::cerr << "[EditorUI] Failed to load Consolas from " << kConsolasPath
                      << ", fall back to ImGui default font.\n";
            io.Fonts->AddFontDefault();
        }
    }

    // --- Read the content scale of the monitor hosting the main window ---
    // Windows 100% → 1.0, 125% → 1.25, 150% → 1.5, 200% → 2.0 ...
    float xScale = 1.0f, yScale = 1.0f;
    glfwGetWindowContentScale(renderer->window, &xScale, &yScale);
    monitorDpi = (xScale > 0.0f) ? xScale : 1.0f;

    ImGui::StyleColorsDark();

    // --- Apply the custom "modern dark" theme ---
    //     Must run before applyUiScale so the theme is captured as baselineStyle
    //     on first snapshot; later User Scale changes must not reset colors
    applyModernDarkTheme();

    // --- Apply scaling once (monitorDpi * userUiScale; initial userUiScale = 1.0) ---
    applyUiScale();

    ImGui_ImplGlfw_InitForVulkan(renderer->window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = renderer->vulkanContext.getInstance();
    initInfo.PhysicalDevice = renderer->vulkanContext.getPhysicalDevice();
    initInfo.Device = device;
    initInfo.QueueFamily = renderer->vulkanContext.getQueueFamilies().graphicsFamily.value();
    initInfo.Queue = renderer->vulkanContext.getGraphicsQueue();
    initInfo.DescriptorPool = imguiDescriptorPool;
    initInfo.MinImageCount = renderer->swapchain.getImageCount();
    initInfo.ImageCount = renderer->swapchain.getImageCount();
    initInfo.UseDynamicRendering = true;

    VkFormat swapFormat = renderer->swapchain.getImageFormat();
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapFormat;

    ImGui_ImplVulkan_Init(&initInfo);

    std::cout << "[EditorUI] ImGui initialized (Dynamic Rendering).\n";
}

void EditorUI::cleanupImGui()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(renderer->vulkanContext.getDevice(), imguiDescriptorPool, nullptr);
    imguiDescriptorPool = VK_NULL_HANDLE;
    std::cout << "[EditorUI] ImGui cleaned up.\n";
}

// ============================================================================
// applyModernDarkTheme — dark theme; pixel-sized fields scaled by applyUiScale().
// ============================================================================
void EditorUI::applyModernDarkTheme()
{
    ImGuiStyle &style = ImGui::GetStyle();

    // --- Non-color: rounding / borders / padding (1.0x baseline values) ---
    style.WindowRounding = 6.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 4.0f;

    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;

    style.WindowPadding = ImVec2(10, 10);
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);
    style.ItemInnerSpacing = ImVec2(6, 4);
    style.IndentSpacing = 18.0f;
    style.ScrollbarSize = 14.0f;
    style.GrabMinSize = 10.0f;

    style.WindowTitleAlign = ImVec2(0.02f, 0.5f); // title text nudged left
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);

    // --- Color table (RGBA, 0~1) based on VSCode Dark+ / Unity Editor Dark, darkened one step ---
    // References:
    //   VSCode defaults/dark_vs.json: editor.bg=#1E1E1E, sideBar.bg=#252526,
    //     activityBar.bg=#333333, editor.fg=#D4D4D4, focusBorder=#007FD4
    //   Unity EditorStyles.skin:     headerBar=#2B2B2B, inspector=#3C3C3C,
    //     selection=#2C5D87, text=#B4B4B4
    //
    // sRGB layering (hex in per-value comments):
    //   bg0=#181818  bg1=#1E1E1E  bg2=#232324  bg3=#2A2A2B  bg4=#333334
    const ImVec4 bg0{0.094f, 0.094f, 0.094f, 1.00f};    // #181818 darkest (DockSpace empty space / inactive tab)
    const ImVec4 bg1{0.118f, 0.118f, 0.118f, 1.00f};    // #1E1E1E window body (= VSCode editor.background)
    const ImVec4 bg2{0.137f, 0.137f, 0.141f, 1.00f};    // #232324 title bar / header / menubar
    const ImVec4 bg3{0.165f, 0.165f, 0.169f, 1.00f};    // #2A2A2B frame / input / button (darkened to avoid large grey areas)
    const ImVec4 bg4{0.200f, 0.200f, 0.204f, 1.00f};    // #333334 hover
    const ImVec4 border{0.235f, 0.235f, 0.243f, 1.00f}; // #3C3C3E solid border
    const ImVec4 borderShadow{0.00f, 0.00f, 0.00f, 0.00f};
    const ImVec4 text{0.800f, 0.800f, 0.800f, 1.00f};         // #CCCCCC VSCode foreground
    const ImVec4 textDisabled{0.459f, 0.459f, 0.471f, 1.00f}; // #757578
    // Unity's official selection blue #2C5D87, deeper than before; hover uses Unity bright blue #3A72B0
    const ImVec4 accent{0.173f, 0.365f, 0.529f, 1.00f};      // #2C5D87
    const ImVec4 accentHover{0.227f, 0.447f, 0.690f, 1.00f}; // #3A72B0
    const ImVec4 accentDim{0.173f, 0.365f, 0.529f, 0.50f};
    const ImVec4 separator{0.220f, 0.220f, 0.224f, 0.80f};
    const ImVec4 scrollBg{0.078f, 0.078f, 0.078f, 0.60f};
    const ImVec4 scrollGrab{0.263f, 0.263f, 0.271f, 1.00f};
    const ImVec4 scrollGrabHover{0.333f, 0.333f, 0.345f, 1.00f};
    const ImVec4 scrollGrabActive{0.404f, 0.404f, 0.420f, 1.00f};
    const ImVec4 dimOverlay{0.094f, 0.094f, 0.094f, 0.60f}; // modal dim background

    ImVec4 *c = style.Colors;
    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = textDisabled;
    c[ImGuiCol_WindowBg] = bg1;
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = bg1;
    c[ImGuiCol_Border] = border;
    c[ImGuiCol_BorderShadow] = borderShadow;

    c[ImGuiCol_FrameBg] = bg3;
    c[ImGuiCol_FrameBgHovered] = bg4;
    c[ImGuiCol_FrameBgActive] = accentDim;

    c[ImGuiCol_TitleBg] = bg0;
    c[ImGuiCol_TitleBgActive] = bg2;
    c[ImGuiCol_TitleBgCollapsed] = bg0;

    c[ImGuiCol_MenuBarBg] = bg2;

    c[ImGuiCol_ScrollbarBg] = scrollBg;
    c[ImGuiCol_ScrollbarGrab] = scrollGrab;
    c[ImGuiCol_ScrollbarGrabHovered] = scrollGrabHover;
    c[ImGuiCol_ScrollbarGrabActive] = scrollGrabActive;

    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = accentHover;

    c[ImGuiCol_Button] = bg3;
    c[ImGuiCol_ButtonHovered] = bg4;
    c[ImGuiCol_ButtonActive] = accent;

    c[ImGuiCol_Header] = bg2;
    c[ImGuiCol_HeaderHovered] = bg4;
    c[ImGuiCol_HeaderActive] = accentDim;

    c[ImGuiCol_Separator] = separator;
    c[ImGuiCol_SeparatorHovered] = accent;
    c[ImGuiCol_SeparatorActive] = accentHover;

    c[ImGuiCol_ResizeGrip] = ImVec4(accent.x, accent.y, accent.z, 0.25f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    c[ImGuiCol_ResizeGripActive] = accentHover;

    c[ImGuiCol_Tab] = bg0;
    c[ImGuiCol_TabHovered] = bg4;
    c[ImGuiCol_TabSelected] = bg1;
    c[ImGuiCol_TabSelectedOverline] = accent;
    c[ImGuiCol_TabDimmed] = bg0;
    c[ImGuiCol_TabDimmedSelected] = bg2;
    c[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(accent.x, accent.y, accent.z, 0.40f);

    c[ImGuiCol_DockingPreview] = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    c[ImGuiCol_DockingEmptyBg] = bg0;

    c[ImGuiCol_PlotLines] = accentHover;
    c[ImGuiCol_PlotLinesHovered] = text;
    c[ImGuiCol_PlotHistogram] = accent;
    c[ImGuiCol_PlotHistogramHovered] = accentHover;

    c[ImGuiCol_TableHeaderBg] = bg2;
    c[ImGuiCol_TableBorderStrong] = border;
    c[ImGuiCol_TableBorderLight] = separator;
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1, 1, 1, 0.03f);

    c[ImGuiCol_TextLink] = accentHover;
    c[ImGuiCol_TextSelectedBg] = ImVec4(accent.x, accent.y, accent.z, 0.45f);

    c[ImGuiCol_DragDropTarget] = accentHover;
    c[ImGuiCol_NavCursor] = accent;
    c[ImGuiCol_NavWindowingHighlight] = ImVec4(1, 1, 1, 0.70f);
    c[ImGuiCol_NavWindowingDimBg] = dimOverlay;
    c[ImGuiCol_ModalWindowDimBg] = dimOverlay;
}

// ============================================================================
// applyUiScale — reapply fonts/style at monitorDpi * userUiScale from a 1.0x baseline snapshot.
// ============================================================================
void EditorUI::applyUiScale()
{
    const float finalScale = monitorDpi * userUiScale;

    ImGuiStyle &style = ImGui::GetStyle();

    // --- First entry: snapshot the current style (StyleColorsDark + custom constants) as the 1.0x baseline ---
    if (!baselineStyleCaptured)
    {
        baselineStyle = style;
        baselineStyleCaptured = true;
    }

    // --- Restore the baseline wholesale (colors + all size fields) so ScaleAllSizes never compounds ---
    style = baselineStyle;

    // --- Font: ImGui 1.92 dynamic fonts; FontSizeBase * FontScaleDpi = final pixel size ---
    ImGuiIO &io = ImGui::GetIO();
    io.FontGlobalScale = 1.0f; // legacy API, no longer used
    style.FontSizeBase = baseFontSize;
    style.FontScaleDpi = finalScale;

    // --- Scale size fields in one pass ---
    style.ScaleAllSizes(finalScale);
}

void EditorUI::createOffscreenResources()
{
    VkDevice device = renderer->vulkanContext.getDevice();
    VmaAllocator vma = renderer->allocator.getVma();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = renderer->swapchain.getImageFormat();
    imageInfo.extent.width = renderer->viewportExtent.width;
    imageInfo.extent.height = renderer->viewportExtent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(vma, &imageInfo, &allocCI,
                       &offscreenImage.image, &offscreenImage.allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create offscreen image!");
    }
    offscreenImage.format = renderer->swapchain.getImageFormat();
    offscreenImage.extent = renderer->viewportExtent;

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = offscreenImage.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = offscreenImage.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &offscreenImage.imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create offscreen image view!");
    }

    // ------------------------------------------------------------------
    // HDR lighting RT (FP16 R16G16B16A16_SFLOAT). The Lighting Pass and
    // Light Volume render here in linear HDR; the Composite Pass (owned by
    // Renderer) samples this and writes offscreenImage (sRGB) after
    // exposure + tone mapping + gamma.
    // ------------------------------------------------------------------
    {
        VkImageCreateInfo hdrInfo{};
        hdrInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        hdrInfo.imageType = VK_IMAGE_TYPE_2D;
        hdrInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        hdrInfo.extent.width = renderer->viewportExtent.width;
        hdrInfo.extent.height = renderer->viewportExtent.height;
        hdrInfo.extent.depth = 1;
        hdrInfo.mipLevels = 1;
        hdrInfo.arrayLayers = 1;
        hdrInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        hdrInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        hdrInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        if (vmaCreateImage(vma, &hdrInfo, &allocCI,
                           &hdrImage.image, &hdrImage.allocation, nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HDR image!");
        }
        hdrImage.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        hdrImage.extent = renderer->viewportExtent;

        VkImageViewCreateInfo hdrViewInfo{};
        hdrViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        hdrViewInfo.image = hdrImage.image;
        hdrViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        hdrViewInfo.format = hdrImage.format;
        hdrViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        hdrViewInfo.subresourceRange.baseMipLevel = 0;
        hdrViewInfo.subresourceRange.levelCount = 1;
        hdrViewInfo.subresourceRange.baseArrayLayer = 0;
        hdrViewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &hdrViewInfo, nullptr, &hdrImage.imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HDR image view!");
        }
    }

    // ------------------------------------------------------------------
    // Pre-FXAA LDR target. Same spec as offscreenImage so the spec block is
    // reused; written by the Composite Pass, sampled by the FXAA Pass.
    // ------------------------------------------------------------------
    {
        VkImageCreateInfo preInfo = imageInfo; // same B8G8R8A8_SRGB / extent / usage
        if (vmaCreateImage(vma, &preInfo, &allocCI,
                           &ldrPreFxaaImage.image, &ldrPreFxaaImage.allocation, nullptr) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pre-FXAA LDR image!");
        }
        ldrPreFxaaImage.format = preInfo.format;
        ldrPreFxaaImage.extent = renderer->viewportExtent;

        VkImageViewCreateInfo preViewInfo = viewInfo;
        preViewInfo.image = ldrPreFxaaImage.image;
        if (vkCreateImageView(device, &preViewInfo, nullptr, &ldrPreFxaaImage.imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pre-FXAA LDR image view!");
        }
    }

    if (offscreenSampler == VK_NULL_HANDLE)
    {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &samplerInfo, nullptr, &offscreenSampler);
    }

    offscreenImGuiDS = ImGui_ImplVulkan_AddTexture(
        offscreenSampler, offscreenImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    std::cout << "[EditorUI] Offscreen RT created ("
              << renderer->viewportExtent.width << "x" << renderer->viewportExtent.height << ").\n";
}

void EditorUI::cleanupOffscreenResources()
{
    VkDevice device = renderer->vulkanContext.getDevice();
    VmaAllocator vma = renderer->allocator.getVma();

    if (offscreenImGuiDS != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(offscreenImGuiDS);
        offscreenImGuiDS = VK_NULL_HANDLE;
    }
    if (offscreenImage.imageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, offscreenImage.imageView, nullptr);
        offscreenImage.imageView = VK_NULL_HANDLE;
    }
    if (offscreenImage.image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(vma, offscreenImage.image, offscreenImage.allocation);
        offscreenImage.image = VK_NULL_HANDLE;
        offscreenImage.allocation = VK_NULL_HANDLE;
    }

    // HDR lighting RT
    if (hdrImage.imageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, hdrImage.imageView, nullptr);
        hdrImage.imageView = VK_NULL_HANDLE;
    }
    if (hdrImage.image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(vma, hdrImage.image, hdrImage.allocation);
        hdrImage.image = VK_NULL_HANDLE;
        hdrImage.allocation = VK_NULL_HANDLE;
    }

    // Pre-FXAA LDR
    if (ldrPreFxaaImage.imageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, ldrPreFxaaImage.imageView, nullptr);
        ldrPreFxaaImage.imageView = VK_NULL_HANDLE;
    }
    if (ldrPreFxaaImage.image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(vma, ldrPreFxaaImage.image, ldrPreFxaaImage.allocation);
        ldrPreFxaaImage.image = VK_NULL_HANDLE;
        ldrPreFxaaImage.allocation = VK_NULL_HANDLE;
    }
}

void EditorUI::cleanup()
{
    // Offscreen must be cleaned before cleanupImGui(): cleanupOffscreenResources
    // calls ImGui_ImplVulkan_RemoveTexture, which needs the ImGui Vulkan backend
    // alive. The reversed order hits ACCESS_VIOLATION in RemoveTexture.
    cleanupOffscreenResources();
    cleanupImGui();
    if (offscreenSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(renderer->vulkanContext.getDevice(), offscreenSampler, nullptr);
        offscreenSampler = VK_NULL_HANDLE;
    }
}

void EditorUI::drawEditorUI()
{
    auto &reg = renderer->scene.registry;
    ImGuizmo::BeginFrame();

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

    // --- Main menu bar ---
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Save Scene", "Ctrl+S"))
            {
                const char *filters[] = {"*.json"};
                const char *path = tinyfd_saveFileDialog("Save Scene", "scene.json", 1, filters, "Scene Files (*.json)");
                if (path)
                {
                    SceneSerializer::saveScene(renderer->scene, path);
                }
            }
            if (ImGui::MenuItem("Load Scene", "Ctrl+O"))
            {
                const char *filters[] = {"*.json"};
                const char *path = tinyfd_openFileDialog("Load Scene", "", 1, filters, "Scene Files (*.json)", 0);
                if (path)
                {
                    vkDeviceWaitIdle(renderer->vulkanContext.getDevice());
                    // If the user loads a scene while a simulation is in
                    // flight, give scripts a chance to run on_stop on the OLD
                    // scene before its entities are destroyed. After this
                    // point the registry is wiped, so anything held by a
                    // script becomes stale. Per-entity callOnStopAll covers
                    // every ScriptComponent in the registry.
                    if (PhysicsSystem::simulationRunning)
                    {
                        if (auto *se = renderer->getScriptEngine())
                        {
                            se->callOnStopAll(renderer->scene);
                        }
                    }
                    // Go through exitPlay(restoreSnapshots=false) to clear
                    // simulationRunning / snapshots / PhysicsWorld / physicsIndex
                    // in one place, then run the scene-level destroyAll and clear
                    // selectedEntity
                    PhysicsSystem::exitPlay(renderer->scene.registry, renderer->physicsWorld,
                                            renderer->transformSnapshots, /*restoreSnapshots=*/false);
                    renderer->scene.selectedEntity = entt::null;
                    // Scene load is a big state change; reset local UI state too
                    colliderEditMode = false;
                    // destroyAll() fires the destruction hooks so every old rigid body unregisters from PhysicsWorld
                    renderer->scene.destroyAll();
                    SceneSerializer::loadScene(renderer->scene, path,
                                               [this](const std::string &source) -> std::shared_ptr<MeshReference>
                                               {
                                                   if (source == "builtin:cube")
                                                       return renderer->cubeMeshRef;
                                                   if (source == "builtin:plane")
                                                       return renderer->planeMeshRef;

                                                   // Resolve `guid:<uuid>[#sub]` via AssetRegistry.
                                                   // Anything else is treated as a raw filesystem path (legacy /
                                                   // outside-assets-root fallback).
                                                   std::string resolvedPath = source;
                                                   std::string subSelector;
                                                   if (source.rfind("guid:", 0) == 0)
                                                   {
                                                       std::string rest = source.substr(5);
                                                       auto hashPos = rest.find('#');
                                                       std::string guid = (hashPos == std::string::npos) ? rest : rest.substr(0, hashPos);
                                                       if (hashPos != std::string::npos)
                                                           subSelector = rest.substr(hashPos + 1);

                                                       auto absOpt = AssetRegistry::instance().guidToPath(guid);
                                                       if (!absOpt)
                                                       {
                                                           std::cerr << "[LoadScene] Unknown GUID, cannot resolve: " << guid << "\n";
                                                           return nullptr;
                                                       }
                                                       resolvedPath = absOpt->string();
                                                   }

                                                   Mesh mesh;
                                                   try
                                                   {
                                                       mesh.loadFromOBJ(resolvedPath);
                                                       auto meshRef = std::make_shared<MeshReference>();
                                                       VkDeviceSize vSize = sizeof(Vertex) * mesh.getVertices().size();
                                                       VkDeviceSize iSize = sizeof(uint32_t) * mesh.getIndices().size();
                                                       meshRef->vertexBuffer = renderer->allocator.createBufferWithStaging(
                                                           renderer->vulkanContext, mesh.getVertices().data(), vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
                                                       meshRef->indexBuffer = renderer->allocator.createBufferWithStaging(
                                                           renderer->vulkanContext, mesh.getIndices().data(), iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
                                                       meshRef->indexCount = mesh.getIndexCount();
                                                       meshRef->vertexCount = static_cast<uint32_t>(mesh.getVertices().size());
                                                       SceneSetup::computeMeshBounds(mesh.getVertices(), meshRef);
                                                       renderer->importedMeshes.push_back(meshRef);
                                                       return meshRef;
                                                   }
                                                   catch (const std::exception &e)
                                                   {
                                                       std::cerr << "[LoadScene] Failed to load mesh: " << resolvedPath << " - " << e.what() << "\n";
                                                       return nullptr;
                                                   }
                                               });
                    renderer->lightsDirty = true;
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Import Model (.obj)"))
            {
                const char *filters[] = {"*.obj"};
                const char *path = tinyfd_openFileDialog("Import OBJ Model", "", 1, filters, "OBJ Files (*.obj)", 0);
                if (path)
                    renderer->doImportModel(path);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "Q"))
                glfwSetWindowShouldClose(renderer->window, GLFW_TRUE);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Add"))
        {
            if (ImGui::MenuItem("Empty Node"))
                renderer->scene.createEntity("New Node");
            ImGui::Separator();
            if (ImGui::MenuItem("Cube"))
            {
                auto e = renderer->scene.createEntity("Cube");
                reg.emplace<MeshComponent>(e, MeshComponent{renderer->cubeMeshRef, "builtin:cube"});
                reg.emplace<MaterialComponent>(e);
            }
            if (ImGui::MenuItem("Plane"))
            {
                auto e = renderer->scene.createEntity("Plane");
                reg.emplace<MeshComponent>(e, MeshComponent{renderer->planeMeshRef, "builtin:plane"});
                reg.emplace<MaterialComponent>(e);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Point Light"))
            {
                auto e = renderer->scene.createEntity("New Light");
                reg.emplace<PointLightComponent>(e);
                renderer->lightsDirty = true;
            }
            if (ImGui::MenuItem("Directional Light"))
            {
                auto e = renderer->scene.createEntity("Dir Light");
                reg.emplace<DirectionalLightComponent>(e);
                renderer->lightsDirty = true;
            }
            if (ImGui::MenuItem("Spot Light"))
            {
                auto e = renderer->scene.createEntity("Spot Light");
                reg.emplace<SpotLightComponent>(e);
                renderer->lightsDirty = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View"))
        {
            // --- Debug View (moved from Render Settings' Debug View collapsible) ---
            if (ImGui::MenuItem("Final", "F5", renderer->debugMode == 0))
                renderer->debugMode = 0;
            if (ImGui::MenuItem("Position", "F1", renderer->debugMode == 1))
                renderer->debugMode = 1;
            if (ImGui::MenuItem("Normal", "F2", renderer->debugMode == 2))
                renderer->debugMode = 2;
            if (ImGui::MenuItem("Albedo", "F3", renderer->debugMode == 3))
                renderer->debugMode = 3;
            if (ImGui::MenuItem("Depth", "F4", renderer->debugMode == 4))
                renderer->debugMode = 4;
            if (ImGui::MenuItem("Hi-Z (mip 0)", "F6", renderer->debugMode == 5))
                renderer->debugMode = 5;

            ImGui::Separator();

            // --- UI Scale (moved from Render Settings' UI collapsible) ---
            if (ImGui::BeginMenu("UI Scale"))
            {
                ImGui::Text("Monitor DPI: %.2fx", monitorDpi);

                // Update the value while dragging; applyUiScale only on release, to avoid rebuilding fonts every frame (visual jitter)
                ImGui::SetNextItemWidth(160.0f);
                ImGui::SliderFloat("User Scale", &userUiScale, 0.5f, 3.0f, "%.2fx");
                if (ImGui::IsItemDeactivatedAfterEdit())
                    applyUiScale();

                ImGui::SameLine();
                if (ImGui::SmallButton("Reset"))
                {
                    userUiScale = 1.0f;
                    applyUiScale();
                }
                ImGui::TextDisabled("Final: %.2fx", monitorDpi * userUiScale);
                ImGui::EndMenu();
            }

            ImGui::Separator();
            // Performance overlay toggle
            ImGui::MenuItem("Performance Overlay", nullptr, &showPerformanceOverlay);

            ImGui::EndMenu();
        }
        // === Tools menu — Benchmark scene presets ===
        if (ImGui::BeginMenu("Tools"))
        {
            if (ImGui::BeginMenu("Benchmark Scenes"))
            {
                using BP = SceneSetup::BenchmarkPreset;
                if (ImGui::MenuItem("Empty (camera baseline)"))
                    SceneSetup::loadBenchmarkScene(BP::Empty, renderer->scene,
                                                   renderer->cubeMeshRef, renderer->planeMeshRef);
                if (ImGui::MenuItem("Stress100 (10x10 cubes + 4 lights)"))
                    SceneSetup::loadBenchmarkScene(BP::Stress100, renderer->scene,
                                                   renderer->cubeMeshRef, renderer->planeMeshRef);
                if (ImGui::MenuItem("Stress1000 (10x10x10 cubes + 16 lights)"))
                    SceneSetup::loadBenchmarkScene(BP::Stress1000, renderer->scene,
                                                   renderer->cubeMeshRef, renderer->planeMeshRef);
                if (ImGui::MenuItem("LightStress (100 cubes + 256 lights)"))
                    SceneSetup::loadBenchmarkScene(BP::LightStress, renderer->scene,
                                                   renderer->cubeMeshRef, renderer->planeMeshRef);
                if (ImGui::MenuItem("TextureStress (100 cubes / 4 textures / 4 lights)"))
                    renderer->loadTextureStressScene();
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // --- 3D viewport panel ---
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("Viewport");
    {
        ImVec2 size = ImGui::GetContentRegionAvail();
        // SSAA: the panel size is the ImGui display size; rendering happens at
        // panel × renderScale, then ImGui::Image's sampler box-filters the downsample.
        const float scale = renderer->getRenderScale();
        const uint32_t panelW = std::max(1u, (uint32_t)size.x);
        const uint32_t panelH = std::max(1u, (uint32_t)size.y);
        // (B) Quantize internal RT to 16-pixel steps so panel jitter / SSAA
        //     rounding doesn't push the target extent across an integer
        //     threshold every frame. With this alone, hairline drags no longer
        //     touch onViewportResize at all.
        auto quantize = [](uint32_t v) -> uint32_t
        {
            constexpr uint32_t q = 16u;
            return std::max(q, (v + q / 2) / q * q);
        };
        const uint32_t w = quantize((uint32_t)(panelW * scale + 0.5f));
        const uint32_t h = quantize((uint32_t)(panelH * scale + 0.5f));

        // (A) Throttle / debounce: onViewportResize() is extremely heavy
        //     (vkDeviceWaitIdle + tear-down & rebuild of GBuffer / depth /
        //     HiZ / Composite / FXAA / SSAO / Bloom / Geometry+Lighting
        //     pipelines / per-frame UBO+SSBO / Gizmo). Triggering it every
        //     frame while the user drags a dock split or window edge stalls
        //     the render thread and looks like a hang. Defer the actual
        //     commit until the mouse is released AND the requested extent
        //     has been stable for a short window.
        static uint32_t s_viewportPendingW = 0;
        static uint32_t s_viewportPendingH = 0;
        static double s_viewportPendingChangeTime = 0.0;
        static bool s_viewportEverCommitted = false;
        if (w != renderer->viewportExtent.width || h != renderer->viewportExtent.height)
        {
            if (w != s_viewportPendingW || h != s_viewportPendingH)
            {
                s_viewportPendingW = w;
                s_viewportPendingH = h;
                s_viewportPendingChangeTime = ImGui::GetTime();
            }
            const bool mouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Left);
            const double stableSec = ImGui::GetTime() - s_viewportPendingChangeTime;
            // First time we ever match the panel size: commit immediately so
            // the first rendered frame uses the correct viewport extent
            // (otherwise we'd render the default 1280x720 for ~150ms on
            // startup). After that, debounce to coalesce drag events.
            const bool firstCommit = !s_viewportEverCommitted;
            if (firstCommit || (!mouseDown && stableSec > 0.15))
            {
                // (D): write the pending flag so drawFrame() performs the
                // fast-path rebuild after vkWaitForFences. Don't modify
                // viewportExtent or call onViewportResize_RT() directly — we're
                // mid-ImGui-frame, and an immediate rebuild would race in-flight
                // command buffers for the same resources.
                renderer->requestViewportResize({s_viewportPendingW, s_viewportPendingH});
                s_viewportPendingW = 0;
                s_viewportPendingH = 0;
                s_viewportEverCommitted = true;
            }
        }
        else
        {
            // Target reached — clear pending so a future change starts a
            // fresh debounce window.
            s_viewportPendingW = 0;
            s_viewportPendingH = 0;
            s_viewportEverCommitted = true;
        }
        ImVec2 viewportPos = ImGui::GetCursorScreenPos();
        if (offscreenImGuiDS)
            ImGui::Image((ImTextureID)offscreenImGuiDS, size);

        // --- Shortcuts (only respond while the viewport is hovered and the mouse isn't captured) ---
        bool viewportHovered = ImGui::IsWindowHovered();
        bool viewportFocused = ImGui::IsWindowFocused();
        if ((viewportHovered || viewportFocused) && !renderer->cursorCaptured && !ImGuizmo::IsUsing())
        {
            if (ImGui::IsKeyPressed(ImGuiKey_W))
            {
                gizmoOperation = ImGuizmo::TRANSLATE;
                colliderEditMode = false;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_E))
            {
                gizmoOperation = ImGuizmo::ROTATE;
                colliderEditMode = false;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_R))
            {
                gizmoOperation = ImGuizmo::SCALE;
                colliderEditMode = false;
            }
        }

        // --- Left-click enters FPS camera (when ImGuizmo isn't in use) ---
        if (viewportHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !renderer->cursorCaptured && !ImGuizmo::IsOver())
        {
            renderer->cursorCaptured = true;
            glfwSetInputMode(renderer->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            double x, y;
            glfwGetCursorPos(renderer->window, &x, &y);
            renderer->camera.resetMouse(x, y);
        }

        // --- ImGuizmo gizmo rendering ---
        // In Play mode, drawing the gizmo is gated by allowPlayModeGizmo:
        //   - non-Play: draw whenever the mouse isn't captured (old behavior)
        //   - Play: draw only if allowPlayModeGizmo == true
        // After a drag, syncTransformToPhysics clears velocity and wakes the body.
        auto sel = renderer->scene.selectedEntity;
        const bool gizmoAllowed = !PhysicsSystem::simulationRunning || allowPlayModeGizmo;
        if (sel != entt::null && reg.valid(sel) && reg.all_of<TransformComponent>(sel) && !renderer->cursorCaptured && gizmoAllowed)
        {

            auto &tc = reg.get<TransformComponent>(sel);
            float aspect = (h > 0) ? (float)w / (float)h : 1.0f;

            glm::mat4 view = renderer->camera.getViewMatrix();
            glm::mat4 proj = renderer->camera.getProjectionMatrix(aspect);
            proj[1][1] *= -1.0f; // flip back to OpenGL convention (ImGuizmo expects it)

            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist();
            ImGuizmo::SetRect(viewportPos.x, viewportPos.y, size.x, size.y);

            if (colliderEditMode && reg.all_of<BoxColliderComponent>(sel))
            {
                // --- Collider edit mode: drag bc.center ---
                auto &bc = reg.get<BoxColliderComponent>(sel);
                glm::mat4 colliderWorld = tc.worldMatrix * glm::translate(glm::mat4(1.0f), bc.center);
                glm::mat4 delta;

                if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                                         ImGuizmo::TRANSLATE, ImGuizmo::LOCAL,
                                         glm::value_ptr(colliderWorld), glm::value_ptr(delta)))
                {
                    // Extract the world position from the edited colliderWorld and invert back to a local center
                    glm::mat4 invWorld = glm::inverse(tc.worldMatrix);
                    glm::vec3 newCenterWorld = glm::vec3(colliderWorld[3]);
                    bc.center = glm::vec3(invWorld * glm::vec4(newCenterWorld, 1.0f));
                }
            }
            else
            {
                // --- Standard Transform gizmo ---
                glm::mat4 modelMatrix = tc.worldMatrix;

                if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                                         gizmoOperation, gizmoMode,
                                         glm::value_ptr(modelMatrix)))
                {

                    // With a parent, convert into local space
                    glm::mat4 localMatrix = modelMatrix;
                    if (reg.all_of<HierarchyComponent>(sel))
                    {
                        auto &hc = reg.get<HierarchyComponent>(sel);
                        if (hc.parent != entt::null && reg.valid(hc.parent) && reg.all_of<TransformComponent>(hc.parent))
                        {
                            auto &parentTC = reg.get<TransformComponent>(hc.parent);
                            localMatrix = glm::inverse(parentTC.worldMatrix) * modelMatrix;
                        }
                    }

                    // Decompose local matrix → position / rotation / scale
                    float matrixTranslation[3], matrixRotation[3], matrixScale[3];
                    ImGuizmo::DecomposeMatrixToComponents(
                        glm::value_ptr(localMatrix),
                        matrixTranslation, matrixRotation, matrixScale);

                    tc.position = glm::vec3(matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]);
                    tc.setRotationMatrix(glm::mat3(localMatrix));
                    tc.scale = glm::vec3(matrixScale[0], matrixScale[1], matrixScale[2]);

                    // Fix: refresh the selected entity subtree's worldMatrix in
                    // place. Only TRS was written above, so tc.worldMatrix still holds
                    // last frame's updateTransforms result, while the
                    // syncTransformToPhysics call below and many downstream consumers
                    // read world pose from tc.worldMatrix. Without recomputing, physics
                    // gets the stale pose and writes it back into the TC on the next
                    // substep — the drag is swallowed (TRANSLATE / ROTATE appear dead,
                    // while SCALE looks fine because physics never writes scale back).
                    //
                    // Generic path: compute the entity's parentWorld, then
                    // updateTransformRecursive to refresh the whole subtree (self
                    // included). No full-scene recompute, no per-component-type cases.
                    {
                        glm::mat4 parentWorld(1.0f);
                        if (reg.all_of<HierarchyComponent>(sel))
                        {
                            auto &hc2 = reg.get<HierarchyComponent>(sel);
                            if (hc2.parent != entt::null && reg.valid(hc2.parent) && reg.all_of<TransformComponent>(hc2.parent))
                                parentWorld = reg.get<TransformComponent>(hc2.parent).worldMatrix;
                        }
                        Systems::updateTransformRecursive(reg, sel, parentWorld);
                    }

                    // After a Play-mode drag, write the new pose back to the physics
                    // world so the next substep can't overwrite Transform with the stale
                    // pose; no-ops for missing / unregistered bodies, harmless outside Play.
                    PhysicsSystem::syncTransformToPhysics(reg, sel, renderer->physicsWorld);

                    // Mark lights dirty after dragging a light
                    bool isLight = reg.all_of<PointLightComponent>(sel) || reg.all_of<DirectionalLightComponent>(sel) || reg.all_of<SpotLightComponent>(sel);
                    if (isLight)
                        renderer->lightsDirty = true;
                }
            }
        }

        // Leave collider edit mode when the selection changes
        {
            static entt::entity lastSel = entt::null;
            if (sel != lastSel)
            {
                colliderEditMode = false;
                lastSel = sel;
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();

    // --- Hierarchy panel ---
    ImGui::Begin("Hierarchy");
    {
        // Delete-key shortcut: when the Hierarchy window is hovered/focused (and the
        // mouse isn't captured) with a valid selection, Delete follows the exact same
        // destroy path as the context menu's "Delete" (destroyEntity recursively clears
        // the subtree, fires onBeforeDestroyEntity to unregister physics bodies, and
        // clears selectedEntity). Requires explicit focus on this panel to avoid accidents.
        {
            bool hierHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
            bool hierFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
            if ((hierHovered || hierFocused) && !renderer->cursorCaptured &&
                ImGui::IsKeyPressed(ImGuiKey_Delete, /*repeat=*/false))
            {
                auto sel = renderer->scene.selectedEntity;
                if (sel != entt::null && reg.valid(sel))
                {
                    // Same as the node context-menu Delete: mark lights dirty so the SSBO refreshes next frame
                    bool isLight = reg.all_of<PointLightComponent>(sel) ||
                                   reg.all_of<DirectionalLightComponent>(sel) ||
                                   reg.all_of<SpotLightComponent>(sel);
                    if (isLight)
                        renderer->lightsDirty = true;
                    renderer->scene.destroyEntity(sel);
                }
            }
        }

        std::function<void(entt::entity)> drawEntity = [&](entt::entity e)
        {
            if (!reg.valid(e))
                return;
            auto &nc = reg.get<NameComponent>(e);
            auto &hc = reg.get<HierarchyComponent>(e);
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (hc.children.empty())
                flags |= ImGuiTreeNodeFlags_Leaf;
            if (renderer->scene.selectedEntity == e)
                flags |= ImGuiTreeNodeFlags_Selected;
            const char *icon = reg.all_of<MeshComponent>(e) ? "[M] " : (reg.all_of<PointLightComponent>(e) ? "[PL] " : (reg.all_of<DirectionalLightComponent>(e) ? "[DL] " : (reg.all_of<SpotLightComponent>(e) ? "[SL] " : "")));
            char label[256];
            snprintf(label, sizeof(label), "%s%s", icon, nc.name.c_str());
            bool opened = ImGui::TreeNodeEx((void *)(intptr_t)(uint32_t)e, flags, "%s", label);
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
                renderer->scene.selectedEntity = e;
            if (ImGui::BeginDragDropSource())
            {
                ImGui::SetDragDropPayload("ECS_ENTITY", &e, sizeof(entt::entity));
                ImGui::Text("%s", nc.name.c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginDragDropTarget())
            {
                if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("ECS_ENTITY"))
                {
                    entt::entity dragged = *(entt::entity *)payload->Data;
                    if (dragged != e)
                        renderer->scene.setParent(dragged, e);
                }
                ImGui::EndDragDropTarget();
            }
            if (ImGui::BeginPopupContextItem())
            {
                if (ImGui::MenuItem("Add Empty Node"))
                {
                    auto c = renderer->scene.createEntity("New Node");
                    renderer->scene.setParent(c, e);
                }
                if (ImGui::MenuItem("Add Point Light"))
                {
                    auto c = renderer->scene.createEntity("New Light");
                    reg.emplace<PointLightComponent>(c);
                    renderer->scene.setParent(c, e);
                    renderer->lightsDirty = true;
                }
                if (ImGui::MenuItem("Add Directional Light"))
                {
                    auto c = renderer->scene.createEntity("Dir Light");
                    reg.emplace<DirectionalLightComponent>(c);
                    renderer->scene.setParent(c, e);
                    renderer->lightsDirty = true;
                }
                if (ImGui::MenuItem("Add Spot Light"))
                {
                    auto c = renderer->scene.createEntity("Spot Light");
                    reg.emplace<SpotLightComponent>(c);
                    renderer->scene.setParent(c, e);
                    renderer->lightsDirty = true;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Delete"))
                {
                    renderer->lightsDirty = true;
                    renderer->scene.destroyEntity(e);
                    ImGui::EndPopup();
                    if (opened)
                        ImGui::TreePop();
                    return;
                }
                ImGui::EndPopup();
            }
            if (opened)
            {
                auto children = hc.children;
                for (auto c : children)
                    drawEntity(c);
                ImGui::TreePop();
            }
        };
        if (ImGui::BeginPopupContextWindow("HierarchyCtx", ImGuiPopupFlags_NoOpenOverItems))
        {
            if (ImGui::MenuItem("Add Empty Node"))
                renderer->scene.createEntity("New Node");
            if (ImGui::MenuItem("Add Point Light"))
            {
                auto e = renderer->scene.createEntity("New Light");
                reg.emplace<PointLightComponent>(e);
                renderer->lightsDirty = true;
            }
            if (ImGui::MenuItem("Add Directional Light"))
            {
                auto e = renderer->scene.createEntity("Dir Light");
                reg.emplace<DirectionalLightComponent>(e);
                renderer->lightsDirty = true;
            }
            if (ImGui::MenuItem("Add Spot Light"))
            {
                auto e = renderer->scene.createEntity("Spot Light");
                reg.emplace<SpotLightComponent>(e);
                renderer->lightsDirty = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Import Model (.obj)"))
            {
                const char *filters[] = {"*.obj"};
                const char *path = tinyfd_openFileDialog("Import OBJ Model", "", 1, filters, "OBJ Files (*.obj)", 0);
                if (path)
                    renderer->doImportModel(path);
            }
            ImGui::EndPopup();
        }
        for (auto e : renderer->scene.getRootEntities())
            drawEntity(e);
    }
    ImGui::End();

    // --- Inspector panel ---
    ImGui::Begin("Inspector");
    {
        auto sel = renderer->scene.selectedEntity;
        if (sel != entt::null && reg.valid(sel))
        {
            auto &nc = reg.get<NameComponent>(sel);
            char buf[128];
            strncpy(buf, nc.name.c_str(), 127);
            buf[127] = '\0';
            if (ImGui::InputText("##Name", buf, 128))
                nc.name = buf;
            ImGui::Separator();
            if (reg.all_of<TransformComponent>(sel))
            {
                auto &tc = reg.get<TransformComponent>(sel);
                bool isLight = reg.all_of<PointLightComponent>(sel) || reg.all_of<DirectionalLightComponent>(sel) || reg.all_of<SpotLightComponent>(sel);
                if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::DragFloat3("Position", &tc.position.x, 0.1f) && isLight)
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat3("Rotation", &tc.rotation.x, 1.0f))
                    {
                        tc.syncOrientationFromEuler();
                        if (isLight)
                            renderer->lightsDirty = true;
                    }
                    ImGui::DragFloat3("Scale", &tc.scale.x, 0.05f, 0.01f, 100.0f);
                }
            }
            if (reg.all_of<MeshComponent>(sel))
            {
                auto &mc = reg.get<MeshComponent>(sel);
                if (mc.mesh && ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Text("Indices: %u", mc.mesh->indexCount);
                    ImGui::Text("Triangles: %u", mc.mesh->indexCount / 3);
                }
            }
            if (reg.all_of<PointLightComponent>(sel))
            {
                auto &plc = reg.get<PointLightComponent>(sel);
                if (ImGui::CollapsingHeader("Point Light", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::ColorEdit3("Color", &plc.color.x))
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat("Intensity", &plc.intensity, 0.1f, 0.0f, 100.0f))
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat("Radius", &plc.radius, 0.1f, 0.1f, 100.0f))
                        renderer->lightsDirty = true;
                    // Point light shadow toggle (bounded by the top-N sort, N=4)
                    if (ImGui::Checkbox("Cast Shadows##PL", &plc.castShadows))
                        renderer->lightsDirty = true;

                    // Unity-style shadow bias (shown only when Cast Shadows is enabled)
                    if (plc.castShadows)
                    {
                        ImGui::Indent();
                        ImGui::TextDisabled("Shadow Bias (Unity-style multipliers, base=1.0)");
                        ImGui::DragFloat("Slope Bias##PL", &plc.shadowSlopeBias, 0.05f, 0.0f, 8.0f);
                        ImGui::DragFloat("Constant Bias##PL", &plc.shadowConstantBias, 0.05f, 0.0f, 8.0f);
                        ImGui::DragFloat("Normal Bias##PL", &plc.shadowNormalBias, 0.05f, 0.0f, 8.0f);
                        ImGui::DragFloat("Depth Bias##PL", &plc.shadowDepthBias, 0.05f, 0.0f, 8.0f);
                        ImGui::Unindent();
                    }
                }
            }
            if (reg.all_of<DirectionalLightComponent>(sel))
            {
                auto &dlc = reg.get<DirectionalLightComponent>(sel);
                if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::ColorEdit3("Color##DL", &dlc.color.x))
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat("Intensity##DL", &dlc.intensity, 0.1f, 0.0f, 100.0f))
                        renderer->lightsDirty = true;
                    // Cast Shadows toggle (aligned with Unity)
                    ImGui::Checkbox("Cast Shadows##DL", &dlc.castShadows);
                    ImGui::TextWrapped("Direction derived from Transform Rotation");
                }
            }
            if (reg.all_of<SpotLightComponent>(sel))
            {
                auto &slc = reg.get<SpotLightComponent>(sel);
                if (ImGui::CollapsingHeader("Spot Light", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::ColorEdit3("Color##SL", &slc.color.x))
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat("Intensity##SL", &slc.intensity, 0.1f, 0.0f, 100.0f))
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat("Radius##SL", &slc.radius, 0.1f, 0.1f, 200.0f))
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat("Inner Angle", &slc.innerAngle, 0.5f, 1.0f, 89.0f))
                        renderer->lightsDirty = true;
                    if (ImGui::DragFloat("Outer Angle", &slc.outerAngle, 0.5f, 1.0f, 90.0f))
                        renderer->lightsDirty = true;
                    if (slc.innerAngle > slc.outerAngle)
                        slc.innerAngle = slc.outerAngle;
                    // Spot light shadow toggle (bounded by the top-N sort, N=4)
                    if (ImGui::Checkbox("Cast Shadows##SL", &slc.castShadows))
                        renderer->lightsDirty = true;
                }
            }
            if (reg.all_of<MeshComponent>(sel))
            {
                if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (!reg.all_of<MaterialComponent>(sel))
                    {
                        if (ImGui::Button("Add Material Component"))
                        {
                            reg.emplace<MaterialComponent>(sel);
                        }
                    }
                    else
                    {
                        auto &mat = reg.get<MaterialComponent>(sel);
                        ImGui::ColorEdit3("Albedo", &mat.albedo.x);
                        ImGui::SliderFloat("Metallic", &mat.metallic, 0.0f, 1.0f);
                        ImGui::SliderFloat("Roughness", &mat.roughness, 0.05f, 1.0f);
                        ImGui::Separator();
                        if (ImGui::Button("Load Material (.mtl)"))
                        {
                            const char *filters[] = {"*.mtl"};
                            const char *path = tinyfd_openFileDialog(
                                "Load Material", "", 1, filters, "MTL Material Files (*.mtl)", 0);
                            if (path)
                            {
                                auto materials = SceneSetup::parseMTL(path);
                                if (materials.size() == 1)
                                {
                                    mat.albedo = materials[0].albedo;
                                    mat.metallic = materials[0].metallic;
                                    mat.roughness = materials[0].roughness;
                                }
                                else if (materials.size() > 1)
                                {
                                    // Multiple materials: open the picker popup
                                    loadedMaterials = materials;
                                    loadMaterialTarget = sel;
                                    ImGui::OpenPopup("Select Material");
                                }
                            }
                        }
                        // Multi-material picker popup
                        if (ImGui::BeginPopup("Select Material"))
                        {
                            ImGui::Text("Select a material:");
                            ImGui::Separator();
                            for (auto &m : loadedMaterials)
                            {
                                // Material name + color preview
                                ImGui::ColorButton(("##clr" + m.name).c_str(),
                                                   ImVec4(m.albedo.r, m.albedo.g, m.albedo.b, 1.0f),
                                                   0, ImVec2(16, 16));
                                ImGui::SameLine();
                                if (ImGui::Selectable(m.name.c_str()))
                                {
                                    if (loadMaterialTarget != entt::null && reg.valid(loadMaterialTarget) && reg.all_of<MaterialComponent>(loadMaterialTarget))
                                    {
                                        auto &targetMat = reg.get<MaterialComponent>(loadMaterialTarget);
                                        targetMat.albedo = m.albedo;
                                        targetMat.metallic = m.metallic;
                                        targetMat.roughness = m.roughness;
                                    }
                                }
                            }
                            ImGui::EndPopup();
                        }
                    }
                }
            }
            // --- RigidBody component ---
            if (reg.all_of<RigidBodyComponent>(sel))
            {
                auto &rb = reg.get<RigidBodyComponent>(sel);
                if (ImGui::CollapsingHeader("Rigid Body", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    // ---- Config items (always editable; bodyType / mass changes are
                    // synced to PhysicsWorld::updateBody next frame by
                    // PhysicsSystem::update, which triggers inertia refresh and transient
                    // reset) ----
                    const char *bodyTypes[] = {"Static", "Dynamic", "Kinematic"};
                    int currentType = static_cast<int>(rb.bodyType);
                    if (ImGui::Combo("Body Type", &currentType, bodyTypes, 3))
                    {
                        rb.bodyType = static_cast<BodyType>(currentType);
                    }
                    if (rb.bodyType == BodyType::Dynamic)
                    {
                        ImGui::DragFloat("Mass##RB", &rb.mass, 0.1f, 0.01f, 1000.0f);
                    }
                    // CP-3.2: RigidBody-level Friction / Restitution controls were
                    // removed; these live only on the collider's PhysicsMaterial (the
                    // Material subsection of the BoxCollider / Colliders panels below).
                    // Reasons: 1) the solver only reads shape.material, so a body-level
                    // fallback is dead unless updateBody writes it back into the material
                    // — a "looks editable" UI that isn't; 2) with multiple colliders a
                    // single body-level value can't express per-shape materials, so it
                    // can only live on the collider layer.

                    // ---- Initial velocity: "current velocity" (read-only runtime
                    //      state) in Play, "start velocity" (config) otherwise; one field
                    //      carries both meanings, distinguished by the UI so runtime
                    //      velocity isn't edited as config ----
                    if (!PhysicsSystem::simulationRunning)
                    {
                        ImGui::DragFloat3("Initial Velocity", &rb.velocity.x, 0.1f);
                    }

                    // ---- Runtime state (Play only; all read-only) ----
                    if (PhysicsSystem::simulationRunning && rb.physicsIndex >= 0)
                    {
                        ImGui::SeparatorText("Runtime State (read-only)");
                        const auto &body = renderer->physicsWorld.getBody(rb.physicsIndex);

                        ImGui::Text("Velocity: %.2f, %.2f, %.2f",
                                    body.velocity.x, body.velocity.y, body.velocity.z);
                        ImGui::Text("Speed: %.2f m/s", glm::length(body.velocity));

                        ImGui::Text("Angular Vel: %.3f, %.3f, %.3f",
                                    body.angularVelocity.x, body.angularVelocity.y, body.angularVelocity.z);
                        ImGui::Text("Angular Speed: %.3f rad/s", glm::length(body.angularVelocity));

                        ImGui::Text("Sleeping: %s (frames=%d)",
                                    body.sleeping ? "yes" : "no", body.sleepFrames);

                        // Per-entity contact count: count getContacts() entries involving this body
                        int involved = 0;
                        for (const auto &c : renderer->physicsWorld.getContacts())
                        {
                            if (c.indexA == rb.physicsIndex || c.indexB == rb.physicsIndex)
                                ++involved;
                        }
                        ImGui::Text("Contacts (this body): %d", involved);

                        ImGui::Text("Physics Index: %d", rb.physicsIndex);
                    }
                }
            }
            // --- BoxCollider component ---
            if (reg.all_of<BoxColliderComponent>(sel))
            {
                auto &bc = reg.get<BoxColliderComponent>(sel);
                if (ImGui::CollapsingHeader("Box Collider", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::DragFloat3("Center##BC", &bc.center.x, 0.05f);
                    ImGui::DragFloat3("Half Extents", &bc.halfExtents.x, 0.05f, 0.01f, 100.0f);
                    // CP-3.2c: Box Collider physics material — friction / restitution
                    // used to live on the RigidBody panel (removed); now on the collider.
                    ImGui::Separator();
                    ImGui::DragFloat("Restitution##BC", &bc.material.restitution, 0.01f, 0.0f, 1.0f);
                    ImGui::DragFloat("Friction##BC", &bc.material.friction, 0.01f, 0.0f, 2.0f);
                    ImGui::DragFloat("Rolling Friction##BC", &bc.material.rollingFriction, 0.001f, 0.0f, 1.0f);
                    ImGui::Separator();
                    if (colliderEditMode)
                    {
                        if (ImGui::Button("Stop Editing Collider"))
                            colliderEditMode = false;
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "Editing...");
                    }
                    else
                    {
                        if (ImGui::Button("Edit Collider"))
                            colliderEditMode = true;
                    }
                }
            }
            // --- Colliders component (generic multi-collider editing entry) ---
            if (reg.all_of<CollidersComponent>(sel))
            {
                auto &cc = reg.get<CollidersComponent>(sel);
                char header[64];
                std::snprintf(header, sizeof(header), "Colliders (%d)", (int)cc.list.size());
                if (ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen))
                {
                    int removeIdx = -1;
                    for (int i = 0; i < (int)cc.list.size(); ++i)
                    {
                        auto &cd = cc.list[i];
                        ImGui::PushID(i);
                        char nodeLabel[64];
                        const char *typeName = "?";
                        switch (cd.shape.type)
                        {
                        case ShapeType::Box:
                            typeName = "Box";
                            break;
                        case ShapeType::Sphere:
                            typeName = "Sphere";
                            break;
                        case ShapeType::Capsule:
                            typeName = "Capsule";
                            break;
                        case ShapeType::ConvexHull:
                            typeName = "ConvexHull";
                            break;
                        case ShapeType::Compound:
                            typeName = "Compound";
                            break;
                        }
                        std::snprintf(nodeLabel, sizeof(nodeLabel), "#%d %s", i, typeName);
                        if (ImGui::TreeNodeEx(nodeLabel, ImGuiTreeNodeFlags_DefaultOpen))
                        {
                            // Shape type dropdown
                            const char *items[] = {"Box", "Sphere", "Capsule", "ConvexHull"};
                            int typeIdx = (int)cd.shape.type;
                            if (typeIdx > 3)
                                typeIdx = 0; // Compound not allowed in the UI; clamp to Box
                            if (ImGui::Combo("Shape", &typeIdx, items, IM_ARRAYSIZE(items)))
                            {
                                // Shape switch: reset to the new type's defaults (no stale parameters)
                                ShapeType newType = (ShapeType)typeIdx;
                                cd.shape = Shape{};
                                cd.shape.type = newType;
                                if (newType == ShapeType::Box)
                                    cd.shape.halfExtents = glm::vec3(0.5f);
                                else if (newType == ShapeType::Sphere)
                                    cd.shape.radius = 0.5f;
                                else if (newType == ShapeType::Capsule)
                                {
                                    cd.shape.radius = 0.3f;
                                    cd.shape.halfHeight = 0.5f;
                                }
                                else if (newType == ShapeType::ConvexHull)
                                    cd.shape.hullIndex = -1;
                            }
                            // Shape parameters
                            switch (cd.shape.type)
                            {
                            case ShapeType::Box:
                                ImGui::DragFloat3("Half Extents", &cd.shape.halfExtents.x, 0.05f, 0.01f, 100.0f);
                                break;
                            case ShapeType::Sphere:
                                ImGui::DragFloat("Radius", &cd.shape.radius, 0.05f, 0.01f, 100.0f);
                                break;
                            case ShapeType::Capsule:
                                ImGui::DragFloat("Radius##cap", &cd.shape.radius, 0.05f, 0.01f, 100.0f);
                                ImGui::DragFloat("Half Height", &cd.shape.halfHeight, 0.05f, 0.01f, 100.0f);
                                break;
                            case ShapeType::ConvexHull:
                                ImGui::Text("Hull Index: %d", cd.shape.hullIndex);
                                ImGui::TextWrapped("ConvexHull vertices editing UI is not implemented in Phase 6.2.C. "
                                                   "Register a hull via code/script and set hullIndex.");
                                break;
                            default:
                                break;
                            }
                            // Local pose
                            ImGui::Separator();
                            ImGui::DragFloat3("Local Position", &cd.localPosition.x, 0.05f);
                            glm::vec3 eul = glm::degrees(glm::eulerAngles(cd.localRotation));
                            if (ImGui::DragFloat3("Local Rotation (deg)", &eul.x, 1.0f))
                            {
                                cd.localRotation = glm::quat(glm::radians(eul));
                            }
                            // Material
                            ImGui::Separator();
                            ImGui::DragFloat("Restitution", &cd.material.restitution, 0.01f, 0.0f, 1.0f);
                            ImGui::DragFloat("Friction", &cd.material.friction, 0.01f, 0.0f, 2.0f);
                            ImGui::DragFloat("Rolling Friction", &cd.material.rollingFriction, 0.001f, 0.0f, 1.0f);
                            // Layer / mask / trigger
                            ImGui::Separator();
                            int layerTmp = (int)cd.layer;
                            if (ImGui::DragInt("Layer (0=inherit)", &layerTmp, 1.0f, 0, 31))
                                cd.layer = (uint32_t)layerTmp;
                            unsigned int maskTmp = cd.mask;
                            ImGui::InputScalar("Mask (0=inherit)", ImGuiDataType_U32, &maskTmp, nullptr, nullptr, "%08X");
                            cd.mask = maskTmp;
                            ImGui::Checkbox("Is Trigger", &cd.isTrigger);
                            // Remove
                            ImGui::Separator();
                            if (ImGui::Button("Remove Collider"))
                                removeIdx = i;
                            ImGui::TreePop();
                        }
                        ImGui::PopID();
                    }
                    if (removeIdx >= 0)
                        cc.list.erase(cc.list.begin() + removeIdx);
                    ImGui::Separator();
                    if (ImGui::Button("Add Collider"))
                    {
                        ColliderDesc cd;
                        cd.shape.type = ShapeType::Box;
                        cd.shape.halfExtents = glm::vec3(0.5f);
                        cc.list.push_back(cd);
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(changes apply on next Play)");
                }
            }
            // --- Script component (per-entity Python scripts) ---
            // Mirrors the Material panel: "Load Script (.py)" opens the file dialog, the
            // current script is shown read-only as a filename (no hand-typed paths),
            // symmetric with Load Material. Reload / Clear / Remove cover hot-reload and
            // unload.
            // Key constraint: after the Inspector edits scriptPath / enabled, the next
            // frame's ScriptEngine::syncFromScene reconciles automatically (attaches the
            // new script / detaches the old) — no manual sync needed.
            if (reg.all_of<ScriptComponent>(sel))
            {
                auto &sc = reg.get<ScriptComponent>(sel);
                if (ImGui::CollapsingHeader("Script", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    // Status badge reads the ScriptComponent mirror fields (ScriptEngine
                    // writes sc.loaded / sc.faulted / sc.lastError on attach / callback
                    // errors). The panel therefore doesn't depend on Play mode and shows
                    // "Loaded / Faulted / Empty" while stopped.
                    if (sc.scriptPath.empty())
                    {
                        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Status: Empty (no script bound)");
                    }
                    else if (sc.faulted)
                    {
                        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Status: Faulted");
                        if (!sc.lastError.empty())
                        {
                            // Wrap lastError so a long traceback summary can't overflow the panel
                            ImGui::TextWrapped("%s", sc.lastError.c_str());
                        }
                    }
                    else if (sc.loaded)
                    {
                        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Status: Loaded");
                    }
                    else
                    {
                        // scriptPath set but loaded=false: ScriptEngine hasn't run
                        // syncFromScene yet (brief window before the first frame), or the
                        // engine isn't initialized.
                        ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f), "Status: Pending attach...");
                    }

                    // Current script shown read-only: filename() shortens the display and
                    // a hover tooltip shows the full path — same "resource name, not full
                    // path" convention as the Mesh / Material panels.
                    if (sc.scriptPath.empty())
                    {
                        ImGui::TextDisabled("Script: <none>");
                    }
                    else
                    {
                        std::filesystem::path p(sc.scriptPath);
                        ImGui::Text("Script: %s", p.filename().generic_string().c_str());
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("%s", sc.scriptPath.c_str());
                    }

                    ImGui::Separator();

                    // Main action: Load Script (.py) — fully symmetric with the Material
                    // panel's "Load Material (.mtl)". The dialog result goes straight into
                    // sc.scriptPath; next frame's syncFromScene sees the change and
                    // attaches the new script / detaches the old one.
                    if (ImGui::Button("Load Script (.py)"))
                    {
                        const char *filters[] = {"*.py"};
                        const char *picked = tinyfd_openFileDialog(
                            "Load Script", "", 1, filters, "Python Script (*.py)", 0);
                        if (picked && *picked)
                        {
                            // Store the absolute path; ScriptEngine::attachScript resolves
                            // it via resolveScriptAbsPath (relative or absolute), so the
                            // .py can be picked from anywhere.
                            sc.scriptPath = picked;
                            sc.loaded = false;
                            sc.faulted = false;
                            sc.lastError.clear();
                        }
                    }

                    // Clear: keep the ScriptComponent but drop the .py (status back to
                    // Empty). For temporarily unmounting while keeping enabled / defaults.
                    bool canClear = !sc.scriptPath.empty();
                    if (!canClear)
                        ImGui::BeginDisabled();
                    ImGui::SameLine();
                    if (ImGui::Button("Clear##Script"))
                    {
                        // Detach the Python side first (frees globals), then clear the
                        // path so syncFromScene also erases the EntityScript entry.
                        if (auto *se = renderer->getScriptEngine())
                            se->detachScript(sel);
                        sc.scriptPath.clear();
                        sc.loaded = false;
                        sc.faulted = false;
                        sc.lastError.clear();
                    }
                    if (!canClear)
                        ImGui::EndDisabled();

                    // Reload: force a re-exec of the current scriptPath; recovers even
                    // from faulted — reloadScript routes through attachScript, which
                    // resets the EntityScript and the mirrored sc.faulted flag.
                    bool canReload = !sc.scriptPath.empty();
                    if (!canReload)
                        ImGui::BeginDisabled();
                    ImGui::SameLine();
                    if (ImGui::Button("Reload##Script"))
                    {
                        if (auto *se = renderer->getScriptEngine())
                        {
                            // Clear the fault so reloadScript takes the attachScript path (no stale state)
                            se->clearEntityFault(sel);
                            se->reloadScript(renderer->scene, sel);
                        }
                    }
                    if (!canReload)
                        ImGui::EndDisabled();

                    ImGui::Separator();

                    ImGui::Checkbox("Enabled##Script", &sc.enabled);
                    ImGui::SameLine();
                    ImGui::TextDisabled("(false = skip on_start/on_update/on_stop)");

                    if (ImGui::Button("Remove Script Component"))
                    {
                        // Detach the Python side before removing the ECS component —
                        // otherwise syncFromScene would see the missing ScriptComponent
                        // next frame and run the detach branch again; harmless, but this
                        // order is cleaner.
                        if (auto *se = renderer->getScriptEngine())
                            se->detachScript(sel);
                        reg.remove<ScriptComponent>(sel);
                    }
                }
            }
            // --- Add Component buttons ---
            if (!reg.all_of<RigidBodyComponent>(sel))
            {
                if (ImGui::Button("Add RigidBody"))
                {
                    reg.emplace<RigidBodyComponent>(sel);
                    if (!reg.all_of<BoxColliderComponent>(sel))
                    {
                        BoxColliderComponent bc;
                        if (reg.all_of<MeshComponent>(sel))
                        {
                            auto &mc = reg.get<MeshComponent>(sel);
                            if (mc.mesh)
                            {
                                bc.center = mc.mesh->boundsCenter;
                                bc.halfExtents = mc.mesh->boundsExtents;
                            }
                        }
                        reg.emplace<BoxColliderComponent>(sel, bc);
                    }
                    PhysicsSystem::registerEntity(reg, sel, renderer->physicsWorld);
                }
            }
            if (!reg.all_of<CollidersComponent>(sel))
            {
                if (ImGui::Button("Add Colliders Component"))
                {
                    CollidersComponent cc;
                    ColliderDesc cd;
                    cd.shape.type = ShapeType::Box;
                    cd.shape.halfExtents = glm::vec3(0.5f);
                    cc.list.push_back(cd);
                    reg.emplace<CollidersComponent>(sel, cc);
                }
            }
            // Allow attaching a ScriptComponent straight from the Inspector. It starts
            // with an empty path; the user fills in a .py via Load Script, and the next
            // frame's syncFromScene attaches it automatically.
            if (!reg.all_of<ScriptComponent>(sel))
            {
                if (ImGui::Button("Add Script Component"))
                {
                    ScriptComponent sc;
                    sc.scriptPath = ""; // empty path = "Empty" placeholder; no Python exception
                    sc.enabled = true;
                    reg.emplace<ScriptComponent>(sel, sc);
                }
            }
        }
        else
        {
            ImGui::TextWrapped("Select an entity in the Hierarchy panel.");
        }
    }
    ImGui::End();

    // --- Render Settings panel ---
    ImGui::Begin("Render Settings");
    {
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
        {
            float fov = renderer->camera.getFov();
            if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f))
                renderer->camera.setFov(fov);
            float np = renderer->camera.getNearPlane();
            if (ImGui::DragFloat("Near Plane", &np, 0.01f, 0.001f, 10.0f))
                renderer->camera.setNearPlane(np);
            float fp = renderer->camera.getFarPlane();
            if (ImGui::DragFloat("Far Plane", &fp, 1.0f, 1.0f, 10000.0f))
                renderer->camera.setFarPlane(fp);
        }
        if (ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("Frame Time: %.2f ms", 1000.0f / ImGui::GetIO().Framerate);
            ImGui::Text("Meshes: %d", (int)reg.view<MeshComponent>().size());
            ImGui::Text("Lights: %d", (int)reg.view<PointLightComponent>().size());
            ImGui::Text("Viewport: %ux%u", renderer->viewportExtent.width, renderer->viewportExtent.height);
            ImGui::Text("RigidBodies: %d", (int)reg.view<RigidBodyComponent>().size());
        }
        if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // Play / Pause / Stop icon buttons (replacing the original text buttons)
            //   - not running: Play enabled, Pause / Stop disabled
            //   - running, not paused: Play disabled, Pause enabled (unhighlighted), Stop enabled
            //   - running, paused: Play disabled, Pause enabled (highlighted = "paused"), Stop enabled
            const bool running = PhysicsSystem::simulationRunning;
            const bool paused = PhysicsSystem::simulationPaused;

            if (IconButton("##play", PlayIconKind::Play,
                           /*active=*/false, /*disabled=*/running))
            {
                // Leave collider edit mode before entering Play so the gizmo isn't still
                // on the collider-editing track when the simulation starts
                colliderEditMode = false;
                PhysicsSystem::enterPlay(reg, renderer->physicsWorld, renderer->transformSnapshots);
                // Fire on_start AFTER enterPlay so the script sees the
                // snapshot-locked TRS state the simulation will run from.
                // Errors in on_start mark scriptFaulted_ inside ScriptEngine;
                // subsequent on_update calls become no-ops until Reload
                // clears the flag. Also fire callOnStartAll for every Entity
                // carrying an enabled ScriptComponent.
                if (auto *se = renderer->getScriptEngine())
                {
                    se->syncFromScene(renderer->scene);
                    se->callOnStartAll(renderer->scene);
                }
            }
            ImGui::SameLine(0, 4);
            if (IconButton("##pause", PlayIconKind::Pause,
                           /*active=*/(running && paused),
                           /*disabled=*/!running))
            {
                PhysicsSystem::setPaused(!paused);
            }
            ImGui::SameLine(0, 4);
            if (IconButton("##stop", PlayIconKind::Stop,
                           /*active=*/false, /*disabled=*/!running))
            {
                // Same reset on exiting Play; symmetric with Play / Load Scene
                colliderEditMode = false;
                // Fire on_stop BEFORE exitPlay so the user script observes
                // the live post-simulation world state. After exitPlay() runs
                // the registry is back to the pre-Play snapshot — anything the
                // script wanted to capture (final velocities, positions) would
                // already be gone. Per-entity callOnStopAll runs for every
                // ScriptComponent in the registry.
                if (auto *se = renderer->getScriptEngine())
                {
                    se->callOnStopAll(renderer->scene);
                }
                PhysicsSystem::exitPlay(reg, renderer->physicsWorld, renderer->transformSnapshots,
                                        /*restoreSnapshots=*/true);
            }
            ImGui::SameLine();
            const char *stateText = !running ? "Stopped"
                                             : (paused ? "Paused" : "Simulating...");
            ImGui::Text("%s", stateText);

            // Explicit toggle for Gizmo interaction while running.
            //   - OFF (default): no Gizmo drawn in Play mode — prevents accidental drags
            //     from disrupting the physics simulation
            //   - ON: Gizmo visible in Play; dragging a Dynamic body writes back to the
            //     physics world, clears velocity, and wakes it
            // Outside Play the toggle doesn't affect Gizmo availability; UI display only.
            {
                const char *gizmoLabel = allowPlayModeGizmo ? "Play Gizmo: ON" : "Play Gizmo: OFF";
                // Snapshot the flag locally to decide whether to Pop, so a Button
                // callback flipping allowPlayModeGizmo can't mismatch Push/Pop counts
                // (ImGui asserts / crashes).
                const bool pushed = allowPlayModeGizmo;
                if (pushed)
                    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                if (ImGui::Button(gizmoLabel))
                    allowPlayModeGizmo = !allowPlayModeGizmo;
                if (pushed)
                    ImGui::PopStyleColor();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Allow Gizmo drag/rotate/scale while simulation is running.\nOFF: Gizmo hidden in Play mode.\nON: dragging a Dynamic body stops its velocity and re-enters simulation.");
            }
            ImGui::DragFloat3("Gravity", &renderer->physicsWorld.gravity.x, 0.1f);

            // Physics stats + debug log toggle
            ImGui::SeparatorText("Stats");
            const PhysicsStats &st = renderer->physicsWorld.stats();
            ImGui::Text("Body: %d total / %d active / %d sleeping",
                        st.bodyCount, st.activeBodyCount, st.sleepingBodyCount);
            ImGui::Text("Shape: %d   Broadphase Nodes: %d",
                        st.shapeCount, st.broadphaseNodeCount);
            ImGui::Text("Broadphase Pairs: %d   Contacts: %d",
                        st.broadphasePairs, st.contactCount);
            ImGui::Text("Substeps/frame: %d   Last step: %.3f ms",
                        st.substepsLastFrame, st.lastStepMs);
            ImGui::Text("Events (last frame): %d",
                        (int)renderer->physicsWorld.events().size());

            ImGui::SeparatorText("Debug");
            bool dbg = renderer->physicsWorld.debugLogEnabled;
            if (ImGui::Checkbox("Enable physics_debug.log (first 500 steps)", &dbg))
                renderer->physicsWorld.debugLogEnabled = dbg;
        }
    }
    ImGui::End();

    // --- Script Console panel ---
    // Dock panel on par with Inspector / Hierarchy / Render Settings. Content is
    // ScriptEngine::getLogSnapshot() (polled per frame; script output is small, no
    // perf concern). The redirector funnels both stdout and stderr here, each line
    // carrying [stdout]/[stderr] plus an optional [on_xxx #id Name] prefix so
    // multiple entities can be told apart.
    //
    // Conventions (same as Render Settings / Inspector):
    //   - showScriptConsole is toggled from the View menu (future); shown directly now
    //   - filter matches case-sensitively (Unity Console is too; simpler to implement)
    //   - autoScroll: shown top-right, snaps ScrollY to GetScrollMaxY() to follow the tail
    if (showScriptConsole)
    {
        ImGui::Begin("Script Console", &showScriptConsole);
        if (auto *se = renderer->getScriptEngine())
        {
            // Toolbar: Clear / Auto-scroll / Reload All / Filter
            if (ImGui::Button("Clear##ScriptConsole"))
                se->clearLog();
            ImGui::SameLine();
            ImGui::Checkbox("Auto-scroll", &scriptConsoleAutoScroll);
            ImGui::SameLine();
            // Reload All Scripts: iterate every entity with a ScriptComponent and call
            // reloadScript on each (clears the fault flag + re-execs). Only available
            // outside Play so a half hot-swap can't break the running state.
            const bool playRunning = PhysicsSystem::simulationRunning;
            if (playRunning)
                ImGui::BeginDisabled();
            if (ImGui::Button("Reload All Scripts"))
            {
                auto &reg = renderer->scene.registry;
                auto view = reg.view<ScriptComponent>();
                for (auto e : view)
                {
                    se->clearEntityFault(e);
                    se->reloadScript(renderer->scene, e);
                }
            }
            if (playRunning)
            {
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Stop simulation first to reload scripts safely.");
            }

            ImGui::SameLine();
            ImGui::SetNextItemWidth(160.0f);
            ImGui::InputTextWithHint("##ScriptConsoleFilter", "Filter (substring)",
                                     scriptConsoleFilter, sizeof(scriptConsoleFilter));

            ImGui::Separator();

            // Scrolling text area: one child window around the whole log with
            // HorizontalScrollbar so long lines scroll sideways instead of wrapping
            // (traceback file paths are often too wide).
            const ImGuiWindowFlags childFlags = ImGuiWindowFlags_HorizontalScrollbar;
            ImGui::BeginChild("ScriptConsoleScroll", ImVec2(0, 0), false, childFlags);

            std::string snapshot = se->getLogSnapshot();
            const std::string filter = scriptConsoleFilter;

            if (filter.empty())
            {
                // No filter: render the whole block unformatted — cheapest path.
                // TextUnformatted splits on '\n' and clips to the viewport internally,
                // so thousands of lines stay smooth.
                ImGui::TextUnformatted(snapshot.c_str(),
                                       snapshot.c_str() + snapshot.size());
            }
            else
            {
                // Filtered: scan line by line, render only hits. The O(N) scan is cheap
                // for <1k-line script logs; consider a pre-split line cache beyond that.
                size_t pos = 0;
                while (pos < snapshot.size())
                {
                    size_t lineEnd = snapshot.find('\n', pos);
                    if (lineEnd == std::string::npos)
                        lineEnd = snapshot.size();
                    const char *lineBeg = snapshot.c_str() + pos;
                    const size_t lineLen = lineEnd - pos;
                    // std::string::npos == not found; string_view would cost the same
                    // but the project doesn't use it yet, so search manually (memmem-style):
                    bool match = false;
                    if (lineLen >= filter.size())
                    {
                        for (size_t i = 0; i + filter.size() <= lineLen; ++i)
                        {
                            if (std::memcmp(lineBeg + i, filter.data(), filter.size()) == 0)
                            {
                                match = true;
                                break;
                            }
                        }
                    }
                    if (match)
                        ImGui::TextUnformatted(lineBeg, lineBeg + lineLen);
                    pos = lineEnd + 1;
                }
            }

            // Auto-scroll to the tail only when already near the bottom, so the user's
            // manual scroll up through history isn't interrupted (same as Unity Console).
            if (scriptConsoleAutoScroll &&
                ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 10.0f)
            {
                ImGui::SetScrollHereY(1.0f);
            }

            ImGui::EndChild();
        }
        else
        {
            ImGui::TextDisabled("Script engine not initialised.");
        }
        ImGui::End();
    }

    // Performance Overlay panel
    if (showPerformanceOverlay)
        drawPerformanceOverlay();
}

// ============================================================================
// Performance Overlay — per-frame GPU pass timings, pipeline stats, frame graph.
// ============================================================================
void EditorUI::drawPerformanceOverlay()
{
    const GpuProfiler &prof = renderer->getGpuProfiler();

    // Default to the top-right corner (first use only; later drags persist via imgui.ini)
    ImGuiViewport *vp = ImGui::GetMainViewport();
    ImVec2 pivot{1.0f, 0.0f};
    ImVec2 pos{vp->WorkPos.x + vp->WorkSize.x - 8.0f, vp->WorkPos.y + 8.0f};
    ImGui::SetNextWindowPos(pos, ImGuiCond_FirstUseEver, pivot);
    ImGui::SetNextWindowBgAlpha(0.78f);

    // Don't add ImGuiWindowFlags_NoSavedSettings — ImGui then wouldn't write the
    // window position / collapsed state to imgui.ini and it would reset on restart.
    // AlwaysAutoResize shrinks with content without affecting saved position.
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
    if (!ImGui::Begin("Performance", &showPerformanceOverlay, flags))
    {
        ImGui::End();
        return;
    }

    if (!prof.isAvailable())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.4f, 1.0f),
                           "GPU profiler unavailable on this device.");
        ImGui::End();
        return;
    }

    // === Top summary ===
    const auto &stats = prof.getPassStats();
    auto frameIt = stats.find("Frame Total");
    if (frameIt != stats.end())
    {
        const auto &fs = frameIt->second;
        float fps = fs.lastMs > 0.0001f ? (1000.0f / fs.lastMs) : 0.0f;
        ImGui::Text("Frame: %.2f ms (%.0f FPS)", fs.lastMs, fps);
        ImGui::Text("avg %.2f  p95 %.2f  max %.2f", fs.avgMs, fs.p95Ms, fs.maxMs);
        ImGui::Separator();
    }

    // === Pass list (in first-appearance order) ===
    if (ImGui::BeginTable("PassTable", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingFixedFit))
    {
        ImGui::TableSetupColumn("Pass");
        ImGui::TableSetupColumn("avg ms");
        ImGui::TableSetupColumn("p95 ms");
        ImGui::TableSetupColumn("max ms");
        ImGui::TableHeadersRow();

        for (const auto &name : prof.getPassOrder())
        {
            if (name == "Frame Total")
                continue; // already shown at top
            auto it = stats.find(name);
            if (it == stats.end())
                continue;
            const auto &s = it->second;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(name.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f", s.avgMs);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.3f", s.p95Ms);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.3f", s.maxMs);
        }
        ImGui::EndTable();
    }

    // === CPU pass list (symmetric with GPU) ===
    {
        const CpuProfiler &cprof = renderer->getCpuProfiler();
        const auto &centries = cprof.getEntries();
        if (!centries.empty())
        {
            ImGui::Separator();
            ImGui::TextDisabled("CPU Passes (ms)");

            // Top row: Frame Total (CPU)
            auto cFrameIt = centries.find("Frame Total");
            if (cFrameIt != centries.end())
            {
                const auto &fs = cFrameIt->second;
                ImGui::Text("CPU Frame: %.2f ms   avg %.2f  p95 %.2f  max %.2f",
                            fs.lastMs, fs.avgMs, fs.p95Ms, fs.maxMs);
            }

            if (ImGui::BeginTable("CpuPassTable", 4,
                                  ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingFixedFit))
            {
                ImGui::TableSetupColumn("CPU Pass");
                ImGui::TableSetupColumn("avg ms");
                ImGui::TableSetupColumn("p95 ms");
                ImGui::TableSetupColumn("max ms");
                ImGui::TableHeadersRow();
                for (const auto &name : CpuProfiler::getDisplayOrder())
                {
                    if (name == "Frame Total")
                        continue;
                    auto it = centries.find(name);
                    if (it == centries.end())
                        continue;
                    const auto &s = it->second;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(name.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.3f", s.avgMs);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.3f", s.p95Ms);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.3f", s.maxMs);
                }
                ImGui::EndTable();
            }
        }
    }

    // === Frame-time graph (last 120 frames) ===
    if (frameIt != stats.end() && !frameIt->second.history.empty())
    {
        const auto &h = frameIt->second.history;
        std::vector<float> tmp(h.begin(), h.end());
        char overlay[32];
        std::snprintf(overlay, sizeof(overlay), "%.2f ms", frameIt->second.lastMs);
        ImGui::PlotLines("Frame ms", tmp.data(), static_cast<int>(tmp.size()),
                         0, overlay, 0.0f, frameIt->second.maxMs * 1.2f, ImVec2(220, 50));
    }

    // === Pipeline Statistics ===
    const auto &ps = prof.getPipelineStats();
    if (!ps.empty())
    {
        ImGui::Separator();
        ImGui::TextDisabled("Pipeline Statistics");
        for (const auto &kv : ps)
        {
            ImGui::Text("[%s]", kv.first.c_str());
            ImGui::BulletText("Vertices    : %llu", (unsigned long long)kv.second.vertices);
            ImGui::BulletText("Primitives  : %llu", (unsigned long long)kv.second.primitives);
            ImGui::BulletText("Frag Invoc. : %llu", (unsigned long long)kv.second.fragInvocations);
            ImGui::BulletText("Clipped     : %llu", (unsigned long long)kv.second.clippedPrimitives);
        }
    }

    // === Frustum culling controls + stats ===
    ImGui::Separator();
    ImGui::TextDisabled("Culling");
    bool cullEnabled = renderer->isFrustumCullingEnabled();
    if (ImGui::Checkbox("Frustum Culling", &cullEnabled))
        renderer->setFrustumCullingEnabled(cullEnabled);
    uint32_t vis = renderer->getCullVisibleCount();
    uint32_t tot = renderer->getCullTotalCount();
    float ratio = tot > 0 ? (100.0f * static_cast<float>(vis) / static_cast<float>(tot)) : 0.0f;
    ImGui::Text("Meshes: %u / %u  (%.1f%%)", vis, tot, ratio);

    // === BVH culling controls + tree size ===
    bool bvhEnabled = renderer->isBVHCullingEnabled();
    if (ImGui::Checkbox("BVH Culling (requires Frustum Culling)", &bvhEnabled))
        renderer->setBVHCullingEnabled(bvhEnabled);
    ImGui::Text("BVH: %d leaves, %d nodes",
                renderer->getBVHLeafCount(), renderer->getBVHNodeCount());

    // === Light frustum culling controls + stats ===
    bool lcEnabled = renderer->isLightFrustumCullEnabled();
    if (ImGui::Checkbox("Light Frustum Culling", &lcEnabled))
        renderer->setLightFrustumCullEnabled(lcEnabled);
    uint32_t pVis = renderer->getPointLightVisible();
    uint32_t pTot = renderer->getPointLightTotal();
    uint32_t sVis = renderer->getSpotLightVisible();
    uint32_t sTot = renderer->getSpotLightTotal();
    ImGui::Text("Point Lights: %u / %u", pVis, pTot);
    ImGui::Text("Spot  Lights: %u / %u", sVis, sTot);

    // === Light volume controls + draw call count ===
    bool lvEnabled = renderer->isLightVolumeEnabled();
    if (ImGui::Checkbox("Light Volume (point+spot)", &lvEnabled))
        renderer->setLightVolumeEnabled(lvEnabled);
    uint32_t _pd = renderer->getLightVolumePointDraws();
    uint32_t _sd = renderer->getLightVolumeSpotDraws();
    ImGui::Text("LV Draws: %u point, %u spot, %u total", _pd, _sd, _pd + _sd);

    // === HDR exposure slider ===
    float _exposure = renderer->getExposure();
    if (ImGui::SliderFloat("Exposure", &_exposure, 0.1f, 10.0f, "%.2f"))
        renderer->setExposure(_exposure);

    // === Tone mapping curve (Linear / Reinhard / ACES) ===
    {
        const char *tonemapItems[] = {"Linear", "Reinhard", "ACES Filmic"};
        int _tm = renderer->getTonemapMode();
        if (ImGui::Combo("Tone Mapping", &_tm, tonemapItems, IM_ARRAYSIZE(tonemapItems)))
            renderer->setTonemapMode(_tm);
    }

    // === SSAA (super-sampling anti-aliasing). Default 1.0× = off.
    //    Internal resolution = panel × renderScale; ImGui Image still displays at
    //    panel pixels, the GPU sampler downsamples implicitly — geometry / specular /
    //    shadow edges all benefit. Cost: ×scale² fill rate and memory.
    {
        const char *ssaaItems[] = {"Off (1.0×)", "1.5×", "2.0×", "4.0×"};
        const float ssaaValues[] = {1.0f, 1.5f, 2.0f, 4.0f};
        float curScale = renderer->getRenderScale();
        int curIdx = 0;
        for (int i = 0; i < 4; ++i)
            if (std::abs(ssaaValues[i] - curScale) < 1e-3f)
            {
                curIdx = i;
                break;
            }
        if (ImGui::Combo("Render Scale (SSAA)", &curIdx, ssaaItems, IM_ARRAYSIZE(ssaaItems)))
            renderer->setRenderScale(ssaaValues[curIdx]);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Super-sampling anti-aliasing.\n"
                "Internal render resolution = viewport × scale.\n"
                "Cost scales with scale² (4.0× = 16× pixels). Default Off.");
    }

    // === FXAA toggle (default OFF) ===
    {
        bool _fx = renderer->getFxaaEnabled();
        if (ImGui::Checkbox("FXAA (Anti-Aliasing)", &_fx))
            renderer->setFxaaEnabled(_fx);

        // Quality slider — maps to Lottes edgeThreshold (PRESET 12 → PRESET 39).
        // Disabled when FXAA is off so the UI clearly shows it has no effect.
        ImGui::BeginDisabled(!_fx);
        float _fq = renderer->getFxaaQuality();
        if (ImGui::SliderFloat("FXAA Quality", &_fq, 0.0f, 1.0f, "%.2f"))
            renderer->setFxaaQuality(_fq);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Edge-detection sensitivity.\n0 = console preset (cheap, subtle)\n1 = extreme preset (strongest smoothing)");

        // Sub-pixel blend strength — controls how much of the FXAA-smoothed
        // sample replaces the original on detected edges.
        float _fs = renderer->getFxaaSubpixel();
        if (ImGui::SliderFloat("FXAA Sub-pixel", &_fs, 0.0f, 1.0f, "%.2f"))
            renderer->setFxaaSubpixel(_fs);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Blend factor between raw pixel and FXAA result.\n0 = no visible smoothing on edges\n1 = full Lottes blend");

        // Diagnostic — paint pixels classified as "edge" red so the user can
        // see exactly where FXAA is acting. Useful when the smoothed output
        // looks visually identical to the raw input.
        bool _fdbg = renderer->getFxaaDebugShowEdges();
        if (ImGui::Checkbox("FXAA Show Edges (debug)", &_fdbg))
            renderer->setFxaaDebugShowEdges(_fdbg);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Paint detected edge pixels red.\nQuality slider widens / narrows the red regions.");
        ImGui::EndDisabled();
    }

    // === SSAO params (intensity=0 == off) ===
    {
        float _i = renderer->getSsaoIntensity();
        if (ImGui::SliderFloat("SSAO Intensity", &_i, 0.0f, 2.0f, "%.2f"))
            renderer->setSsaoIntensity(_i);
        float _r = renderer->getSsaoRadius();
        if (ImGui::SliderFloat("SSAO Radius", &_r, 0.05f, 2.0f, "%.2f"))
            renderer->setSsaoRadius(_r);
        float _b = renderer->getSsaoBias();
        if (ImGui::SliderFloat("SSAO Bias", &_b, 0.0f, 0.1f, "%.3f"))
            renderer->setSsaoBias(_b);
    }

    // === Bloom params (intensity=0 == off) ===
    {
        float _bi = renderer->getBloomIntensity();
        if (ImGui::SliderFloat("Bloom Intensity", &_bi, 0.0f, 2.0f, "%.2f"))
            renderer->setBloomIntensity(_bi);
        float _bt = renderer->getBloomThreshold();
        if (ImGui::SliderFloat("Bloom Threshold", &_bt, 0.0f, 5.0f, "%.2f"))
            renderer->setBloomThreshold(_bt);
        float _bk = renderer->getBloomSoftKnee();
        if (ImGui::SliderFloat("Bloom Soft Knee", &_bk, 0.0f, 1.0f, "%.2f"))
            renderer->setBloomSoftKnee(_bk);
        float _bs = renderer->getBloomScatter();
        if (ImGui::SliderFloat("Bloom Scatter", &_bs, 0.1f, 2.0f, "%.2f"))
            renderer->setBloomScatter(_bs);
    }

    // === Directional shadow ===
    {
        ImGui::Separator();
        ImGui::TextDisabled("Shadows (Directional)");
        bool _se = renderer->getShadowsEnabled();
        if (ImGui::Checkbox("Shadows Enabled##DirShadow", &_se))
            renderer->setShadowsEnabled(_se);
        float _sd = renderer->getShadowDistance();
        if (ImGui::SliderFloat("Shadow Distance", &_sd, 1.0f, 200.0f, "%.1f"))
            renderer->setShadowDistance(_sd);
        // Resolution via dropdown (a slider's arbitrary values would resize too often)
        const int kResolutions[] = {512, 1024, 2048, 4096};
        const char *kResLabels[] = {"512 (Low)", "1024 (Medium)", "2048 (High)", "4096 (Ultra)"};
        int curRes = renderer->getShadowMapResolution();
        int curIdx = 2; // default 2048
        for (int i = 0; i < 4; ++i)
            if (kResolutions[i] == curRes)
            {
                curIdx = i;
                break;
            }
        if (ImGui::Combo("Shadow Map Size", &curIdx, kResLabels, IM_ARRAYSIZE(kResLabels)))
            renderer->setShadowMapResolution(kResolutions[curIdx]);
    }

    // === Target FPS cap (0 = unlocked; benchmark recommends 0) ===
    int _targetFps = renderer->getTargetFps();
    if (ImGui::SliderInt("Target FPS", &_targetFps, 0, 240, _targetFps == 0 ? "off" : "%d"))
        renderer->setTargetFps(_targetFps);

    // === Scene stats (real scene-scale numbers) ===
    {
        auto &reg = renderer->scene.registry;
        auto view = reg.view<MeshComponent>();
        uint32_t sceneMeshes = 0, sceneVerts = 0, sceneTris = 0;
        for (auto e : view)
        {
            const auto &mc = view.get<MeshComponent>(e);
            if (!mc.mesh)
                continue;
            ++sceneMeshes;
            sceneVerts += mc.mesh->vertexCount;
            sceneTris += mc.mesh->indexCount / 3;
        }
        const auto &dc0 = renderer->getDrawCallStats();
        ImGui::Separator();
        ImGui::TextDisabled("Scene Stats");
        ImGui::Text("  Meshes  : %u (drawn %u)", sceneMeshes, dc0.geometry);
        // %.1fK / %.2fM formats read better
        auto fmtCount = [](uint32_t n, char *buf, size_t cap)
        {
            if (n >= 1000000)
                std::snprintf(buf, cap, "%.2fM", n / 1.0e6);
            else if (n >= 1000)
                std::snprintf(buf, cap, "%.1fK", n / 1.0e3);
            else
                std::snprintf(buf, cap, "%u", n);
        };
        char vbuf[32], tbuf[32];
        fmtCount(sceneVerts, vbuf, sizeof(vbuf));
        fmtCount(sceneTris, tbuf, sizeof(tbuf));
        ImGui::Text("  Vertices: %s", vbuf);
        ImGui::Text("  Triangles: %s", tbuf);
    }

    // === Draw call counter (for Unity perf comparisons) ===
    {
        const auto &dc = renderer->getDrawCallStats();
        ImGui::Separator();
        ImGui::TextDisabled("Draw Calls (last frame)");
        ImGui::Text("  Geometry  : %u", dc.geometry);
        ImGui::Text("  LightVolume: %u", dc.lightVolume);
        ImGui::Text("  Lighting   : %u", dc.lighting);
        ImGui::Text("  Gizmo      : %u", dc.gizmo);
        ImGui::Text("  ImGui      : %u", dc.imgui);
        ImGui::Text("  TOTAL      : %u", dc.total());
    }

    // === GPU memory (VMA heap budgets) ===
    auto heapStats = renderer->allocator.getHeapStats();
    if (!heapStats.empty())
    {
        ImGui::Separator();
        ImGui::TextDisabled("GPU Memory");
        const double MB = 1024.0 * 1024.0;
        for (const auto &h : heapStats)
        {
            const bool deviceLocal = (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            ImGui::Text("Heap %u %s", h.heapIndex, deviceLocal ? "(DEVICE_LOCAL)" : "(HOST)");
            float frac = (h.budget > 0)
                             ? static_cast<float>(static_cast<double>(h.usage) / static_cast<double>(h.budget))
                             : 0.0f;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.1f / %.1f MB",
                          h.usage / MB, h.budget / MB);
            ImGui::ProgressBar(frac, ImVec2(220, 0), buf);
            ImGui::Text("  Allocs: %u   Block: %.1f MB",
                        h.allocationCount, h.blockBytes / MB);
        }
    }

    // === Benchmark mode ===
    {
        ImGui::Separator();
        ImGui::TextDisabled("Benchmark");
        BenchmarkRunner &runner = renderer->getBenchmarkRunner();

        if (runner.isRunning())
        {
            const char *phase = (runner.getState() == BenchmarkRunner::State::Warmup)
                                    ? "Warmup"
                                    : "Sampling";
            double dur = (runner.getState() == BenchmarkRunner::State::Warmup)
                             ? runner.getWarmupDuration()
                             : runner.getSamplingDuration();
            double elapsed = runner.getPhaseElapsedSec();
            float frac = dur > 0 ? static_cast<float>(elapsed / dur) : 0.0f;
            if (frac > 1.0f)
                frac = 1.0f;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s %.1f / %.1fs", phase, elapsed, dur);
            ImGui::ProgressBar(frac, ImVec2(220, 0), buf);
            ImGui::TextDisabled("scene: %s", runner.getScene().c_str());
            if (ImGui::Button("Abort"))
                runner.abort();
        }
        else
        {
            // Scene name input (inferring it from the selected benchmark preset isn't
            // viable; use a static buffer initialized to "Manual", user-editable)
            static char sceneBuf[64] = "Manual";
            ImGui::SetNextItemWidth(140.0f);
            ImGui::InputText("Scene Name", sceneBuf, sizeof(sceneBuf));
            if (ImGui::Button("Run Benchmark (5s warmup + 10s sample)"))
                runner.start(sceneBuf);

            const auto &r = runner.getLastResult();
            if (!r.scene.empty())
            {
                ImGui::Separator();
                ImGui::TextDisabled("Last Result");
                ImGui::Text("  scene     : %s", r.scene.c_str());
                ImGui::Text("  fps avg   : %.1f", r.fpsAvg);
                ImGui::Text("  fps p95(low): %.1f", r.fpsP95);
                ImGui::Text("  fps max   : %.1f", r.fpsMax);
                ImGui::Text("  frame avg : %.2f ms", r.frameMsAvg);
                ImGui::Text("  frame p95 : %.2f ms", r.frameMsP95);
                ImGui::Text("  gpu avg   : %.2f ms", r.gpuMsAvg);
                ImGui::Text("  cpu avg   : %.2f ms", r.cpuMsAvg);
                ImGui::Text("  draws avg : %.1f", r.drawsAvg);
                ImGui::Text("  samples   : %zu", r.samples);
                if (!r.csvPath.empty())
                {
                    ImGui::TextDisabled("CSV:");
                    ImGui::SameLine();
                    ImGui::TextWrapped("%s", r.csvPath.c_str());
                }
            }
        }
    }

    ImGui::End();
}
