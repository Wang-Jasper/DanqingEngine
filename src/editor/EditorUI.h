#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <entt/entt.hpp>
#include "core/Allocator.h"
#include "utils/DepthUtils.h"
#include "scene/SceneSetup.h"
#include <imgui.h>
#include <ImGuizmo.h>

class Renderer;

// ImGui editor UI: dock space, panels, viewport, ImGui backend.
class EditorUI
{
public:
    void init(Renderer &renderer);
    void cleanup();

    // Offscreen render targets (Lighting Pass → ImGui viewport).
    void createOffscreenResources();
    void cleanupOffscreenResources();

    // ImGui backend.
    void initImGui();
    void cleanupImGui();

    // DPI / UI scaling: final scale = monitorDpi * userUiScale.
    // applyUiScale() resets the style to the baseline template before applying
    // the current scale, so ScaleAllSizes never multiplies already-scaled values.
    void applyUiScale();
    float getMonitorDpi() const { return monitorDpi; }
    float getUserUiScale() const { return userUiScale; }
    void setUserUiScale(float s)
    {
        userUiScale = s;
        applyUiScale();
    }

    // Draw the editor UI each frame.
    void drawEditorUI();

    // Accessors.
    AllocatedImage &getOffscreenImage() { return offscreenImage; }
    // HDR lighting RT (FP16). Lighting / Light Volume passes write here;
    // the Composite pass reads here and writes the LDR ldrPreFxaaImage.
    AllocatedImage &getHdrImage() { return hdrImage; }
    // LDR target written by the Composite pass and read by the FXAA pass,
    // which writes its result into offscreenImage (the ImGui-bound target).
    AllocatedImage &getLdrPreFxaaImage() { return ldrPreFxaaImage; }
    VkDescriptorSet getOffscreenImGuiDS() const { return offscreenImGuiDS; }
    VkDescriptorPool getImGuiDescriptorPool() const { return imguiDescriptorPool; }

private:
    // --- UI theme (modern dark theme) ---
    // Call after StyleColorsDark() but before applyUiScale() takes its first
    // baseline snapshot, so the theme is absorbed into baselineStyle and runtime
    // UI scale changes keep the colors.
    void applyModernDarkTheme();

    Renderer *renderer = nullptr; // Main renderer, for shared state access.

    // --- ImGui ---
    VkDescriptorPool imguiDescriptorPool = VK_NULL_HANDLE;

    // --- Offscreen rendering (offscreen RT → ImGui viewport) ---
    // Split into HDR + LDR. Lighting / light-volume passes render to hdrImage
    // (FP16); the Composite pass reads hdrImage and writes ldrPreFxaaImage
    // (sRGB B8G8R8A8); the FXAA pass reads that and writes offscreenImage
    // (final, ImGui-bound).
    AllocatedImage hdrImage{};        // FP16 HDR — lighting RT
    AllocatedImage ldrPreFxaaImage{}; // sRGB LDR — composite output, FXAA input
    AllocatedImage offscreenImage{};  // sRGB LDR — final, ImGui-bound
    VkSampler offscreenSampler = VK_NULL_HANDLE;
    VkDescriptorSet offscreenImGuiDS = VK_NULL_HANDLE;

    // --- Load Material popup temp state ---
    std::vector<ParsedMaterial> loadedMaterials;
    entt::entity loadMaterialTarget = entt::null;

    // --- Script Console panel state ---
    // Standalone dock panel that mirrors ScriptEngine::logBuffer_, with
    // substring filtering, auto-scroll, clear, and Reload All Scripts. State
    // stays inside EditorUI so it does not leak into the ScriptEngine API.
    char scriptConsoleFilter[128] = ""; // Substring filter; empty = show all
    bool scriptConsoleAutoScroll = true;
    bool showScriptConsole = true; // View menu can hook a toggle here later

    // --- ImGuizmo gizmo state ---
    ImGuizmo::OPERATION gizmoOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE gizmoMode = ImGuizmo::WORLD;
    bool colliderEditMode = false; // Inspector button toggles collider edit mode

    // Whether gizmos may interact (select/drag/rotate/scale) during Play.
    //   - Outside Play: gizmos always active; this flag is ignored.
    //   - During Play: false (default) hides gizmos and disables dragging
    //     (a safe "record runtime state" default); true draws them, and
    //     dragging a Dynamic body writes its pose back via syncTransformToPhysics.
    //   - Toggled by the Inspector's Play Gizmo button; not reset on Stop/Load
    //     Scene, so the user's preference survives.
    bool allowPlayModeGizmo = false;

    // Performance Overlay visibility toggle (View → Performance Overlay).
    bool showPerformanceOverlay = true;
    void drawPerformanceOverlay();

    // --- DPI / UI scale state ---
    float monitorDpi = 1.0f;    // Read once at startup from glfwGetWindowContentScale
    float userUiScale = 1.0f;   // Manual user scale (Render Settings → UI)
    float baseFontSize = 16.0f; // Logical base font size (pixels at 1.0x)

    // Unscaled 1.0x baseline style copy. applyUiScale() copies it back wholesale
    // before scaling, so ScaleAllSizes never compounds on already-scaled values.
    ImGuiStyle baselineStyle{};
    bool baselineStyleCaptured = false;
};
