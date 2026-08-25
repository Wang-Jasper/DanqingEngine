// main.cpp — program entry point and GLFW window management.
//
// GLFW callbacks are C-style free functions with a fixed signature (no `this`),
// so a global g_renderer pointer bridges them to the renderer instance.
#include "renderer/Renderer.h"
#include "scene/SceneSetup.h" // CLI startup loads benchmark presets

// Standard library
#include <iostream>  // console hints and errors
#include <stdexcept> // catching init / pipeline failures
#include <algorithm> // std::transform (drop callback lowercases the extension)
#include <cctype>    // std::tolower
#include <string>    // CLI argument parsing
#include <cstring>
#include <cstdlib>

// GLFW callbacks have a fixed C-style signature and cannot carry a `this`
// pointer, so a global pointer bridges them to the renderer.
static Renderer *g_renderer = nullptr;

// Fired when the window framebuffer is resized; the swapchain must be rebuilt.
static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
    if (g_renderer)
    {
        // Renderer rebuilds the swapchain in the next drawFrame().
        g_renderer->onFramebufferResize();
    }
}

// Mouse move; only handled in FPS camera mode (cursor captured).
static void mouseCallback(GLFWwindow *window, double xpos, double ypos)
{
    if (g_renderer && g_renderer->isCursorCaptured())
    {
        g_renderer->getCamera().processMouse(xpos, ypos);
    }
}

// Mouse wheel; only handled in FPS camera mode.
static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    if (g_renderer && g_renderer->isCursorCaptured())
    {
        g_renderer->getCamera().processScroll(yoffset);
    }
}

// Drag-and-drop: auto-import .obj files dropped on the window.
static void dropCallback(GLFWwindow *window, int count, const char **paths)
{
    if (!g_renderer)
        return;
    for (int i = 0; i < count; i++)
    {
        std::string path = paths[i];
        // Lowercase the extension so .obj/.OBJ/.Obj all match.
        std::string ext = path.substr(path.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        if (ext == "obj")
        {
            g_renderer->doImportModel(path);
        }
    }
}

static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    // ESC exits FPS camera mode (entered by clicking the viewport panel).
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        if (g_renderer && g_renderer->isCursorCaptured())
        {
            g_renderer->setCursorCaptured(false);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }

    // Q sets the close flag; the main loop exits on the next iteration.
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    // Function keys switch G-Buffer debug visualization.
    if (action == GLFW_PRESS && g_renderer)
    {
        if (key == GLFW_KEY_F5)
            g_renderer->setDebugMode(0);
        if (key == GLFW_KEY_F1)
            g_renderer->setDebugMode(1);
        if (key == GLFW_KEY_F2)
            g_renderer->setDebugMode(2);
        if (key == GLFW_KEY_F3)
            g_renderer->setDebugMode(3);
        if (key == GLFW_KEY_F4)
            g_renderer->setDebugMode(4);
        if (key == GLFW_KEY_F6)
            g_renderer->setDebugMode(5);
    }
}

// CLI argument results.
struct CliArgs
{
    std::string benchmarkPreset; // Empty = interactive mode (no benchmark)
    uint64_t frameLimit = 0;     // 0 = unlimited
    bool autoExit = false;       // Exit after the benchmark finishes
    bool showHelp = false;
};

static void printUsage()
{
    std::cout << "Usage: Danqing [options]\n"
                 "Options:\n"
                 "  --benchmark <preset>   Load preset and start benchmark on startup.\n"
                 "                         preset \xE2\x88\x88 {Empty, Stress100, Stress1000, LightStress, TextureStress}\n"
                 "  --frames <N>           Exit after N frames (safety net).\n"
                 "  --auto-exit            Auto-close window when benchmark finishes.\n"
                 "  --help, -h             Show this help and exit.\n";
}

static CliArgs parseCli(int argc, char **argv)
{
    CliArgs out;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--help" || a == "-h")
            out.showHelp = true;
        else if (a == "--benchmark" && i + 1 < argc)
            out.benchmarkPreset = argv[++i];
        else if (a == "--frames" && i + 1 < argc)
            out.frameLimit = static_cast<uint64_t>(std::atoll(argv[++i]));
        else if (a == "--auto-exit")
            out.autoExit = true;
        else
            std::cerr << "[CLI] unknown arg: " << a << "\n";
    }
    return out;
}

int main(int argc, char **argv)
{
    CliArgs cli = parseCli(argc, argv);
    if (cli.showHelp)
    {
        printUsage();
        return 0;
    }
    // --- Init GLFW ---
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW!\n";
        return -1;
    }

    // --- Configure window hints ---
    // GLFW_NO_API: no OpenGL context (Vulkan only); GLFW would otherwise
    // create one by default.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Resizable window.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    // Per-Monitor DPI awareness: glfwGetWindowContentScale() then returns the
    // true DPI scale; without it Windows stretches the window (blurry text).
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);

    // --- Create window ---
    GLFWwindow *window = glfwCreateWindow(1280, 720, "Danqing", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window!\n";
        glfwTerminate();
        return -1;
    }

    // --- Register callbacks ---
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetDropCallback(window, dropCallback);

    // --- Initial mouse mode: normal (ImGui interactive) ---
    // Click the viewport panel for FPS camera mode; ESC exits it.
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // --- Create the renderer and publish the global pointer ---
    Renderer renderer;
    g_renderer = &renderer;

    try
    {
        // --- Init renderer ---
        renderer.init(window);

        // --- CLI startup logic ---
        if (cli.frameLimit > 0)
        {
            renderer.setFrameLimit(cli.frameLimit);
            std::cout << "[CLI] frame limit: " << cli.frameLimit << "\n";
        }
        if (cli.autoExit)
        {
            renderer.setAutoExitOnBenchmarkDone(true);
            std::cout << "[CLI] auto-exit on benchmark done\n";
        }
        if (!cli.benchmarkPreset.empty())
        {
            if (!renderer.startBenchmarkByName(cli.benchmarkPreset))
            {
                std::cerr << "[CLI] unknown preset '" << cli.benchmarkPreset
                          << "'. Valid: Empty / Stress100 / Stress1000 / LightStress / TextureStress\n";
                renderer.cleanup();
                glfwDestroyWindow(window);
                glfwTerminate();
                return 2;
            }
            std::cout << "[CLI] benchmark mode: preset='" << cli.benchmarkPreset << "'\n";
        }

        // --- Print control hints ---
        std::cout << "\n";
        std::cout << "=== Controls ===\n";
        std::cout << "  WASD       - Move\n";
        std::cout << "  Mouse      - Look around\n";
        std::cout << "  Space/Shift- Up / Down\n";
        std::cout << "  Scroll     - Zoom (FOV)\n";
        std::cout << "  ESC        - Toggle mouse capture\n";
        std::cout << "  Q          - Quit\n";
        std::cout << "  0          - Final render\n";
        std::cout << "  1          - G-Buffer: Position\n";
        std::cout << "  2          - G-Buffer: Normal\n";
        std::cout << "  3          - G-Buffer: Albedo\n";
        std::cout << "  4          - G-Buffer: Depth\n";
        std::cout << "================\n\n";

        // --- Main loop: poll window events, then draw one frame ---
        while (!renderer.shouldClose())
        {
            glfwPollEvents();     // Dispatch pending window events
            renderer.drawFrame(); // Render one frame
        }

        // --- Clean up renderer resources ---
        // vkDeviceWaitIdle waits for the GPU, then Vulkan resources are
        // destroyed in reverse order.
        renderer.cleanup();
    }
    catch (const std::exception &e)
    {
        // --- Exception handling: best-effort cleanup before exiting ---
        std::cerr << "[FATAL] " << e.what() << "\n";
        renderer.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // --- Destroy window and terminate GLFW ---
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "[Main] Exited cleanly.\n";
    return 0;
}
