#pragma once

// ScriptEngine — embedded CPython interpreter via pybind11::embed. The engine
// owns one scoped_interpreter for its whole lifetime.
//
// - pybind11 is included only here, in ScriptEngine.cpp, and in the
//   Bindings_*.cpp files; the rest of the codebase never sees it (PIMPL +
//   forward declarations keep ECS/Vulkan includes out of this header).
// - Every C++ → Python call is wrapped in try/catch(py::error_already_set).
//   On exception the traceback is printed (PyErr_Print) into logBuffer and
//   scriptFaulted_ is set so the editor disables further calls until the user
//   reloads the script.
// - The GIL is held implicitly by scoped_interpreter on the main thread; as
//   long as drawFrame stays single-threaded no explicit gil_scoped_acquire is
//   needed. Revisit if worker threads ever invoke Python.
// - Created at the end of Renderer::init() and destroyed at the start
//   of Renderer::cleanup(): nothing in the Vulkan/ECS shutdown path
//   may run after Python is torn down, because user scripts can still hold
//   callbacks pointing at C++ data.

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include <entt/entt.hpp> // entt::entity is the key type for per-entity scripts

// Forward-declare the renderer to avoid pulling Vulkan headers into every
// translation unit that includes ScriptEngine.h. The real handle is only
// needed inside ScriptEngine.cpp.
class Renderer;

// Forward-declare the ECS / camera types we surface to Python via
// setSceneContext(). Keeping these as pointers here means callers
// (Renderer) include the full headers, but ScriptEngine.h itself
// stays free of ECS / Vulkan transitive includes.
class ECSScene;
class Camera;

// We keep pybind11 includes out of this header on purpose. Members that
// would otherwise need py::module_ / py::scoped_interpreter are hidden behind
// a PIMPL-style forward declaration so most of the codebase only sees a
// vanilla C++ class.
class ScriptEngine
{
public:
    ScriptEngine();
    ~ScriptEngine();

    // Non-copyable / non-movable: there is exactly one Python interpreter
    // per process and we own the scoped_interpreter inside.
    ScriptEngine(const ScriptEngine &) = delete;
    ScriptEngine &operator=(const ScriptEngine &) = delete;
    ScriptEngine(ScriptEngine &&) = delete;
    ScriptEngine &operator=(ScriptEngine &&) = delete;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------
    // init() spins up the interpreter and prepares sys.path so user scripts
    // resolve `import engine` (the embedded module) and can be loaded by
    // module name from `${PROJECT_DIR}/scripts/`.
    //
    // The renderer reference is stored for setSceneContext(); keeping the
    // constructor cheap defers interpreter spin-up to init().
    void init(Renderer &renderer, const std::filesystem::path &scriptsRoot);

    // shutdown() unloads the user module and releases interpreter resources.
    // Safe to call multiple times. Called by ~ScriptEngine() automatically.
    void shutdown();

    // ------------------------------------------------------------------
    // Scene / Camera context publishing
    // ------------------------------------------------------------------
    // setSceneContext() makes the engine's ECSScene and Camera reachable
    // from Python as `engine.scene` / `engine.camera`. Must be called AFTER
    // init() (interpreter exists) and BEFORE any user script that touches
    // those attributes is loaded.
    //
    // The pointers are stored as non-owning views; the caller (typically
    // Renderer) owns lifetime and MUST call clearSceneContext()
    // before tearing down the underlying ECSScene / Camera. shutdown()
    // calls clearSceneContext() automatically.
    void setSceneContext(ECSScene *scene, Camera *camera);
    void clearSceneContext();

    // setDeltaTime() updates the `engine.delta_time` module attribute.
    // callOnUpdate() invokes this before forwarding the call to user code,
    // so scripts always see the same dt their `on_update` argument carries.
    // Exposed publicly so the editor / tests can override it independently
    // (e.g. when stepping a scripted simulation manually).
    void setDeltaTime(float dt);

    // ------------------------------------------------------------------
    // Script loading
    // ------------------------------------------------------------------
    // Loads (or reloads) a single Python script by absolute path. Returns
    // true on success. On failure the traceback is appended to logBuffer
    // and the previously loaded module (if any) is left untouched.
    //
    // The path may live anywhere on disk; we add its parent directory to
    // sys.path on the fly so `import` statements inside the script resolve.
    bool loadScript(const std::filesystem::path &scriptPath);

    // ------------------------------------------------------------------
    // Per-entity script management
    // ------------------------------------------------------------------
    // Each ECS entity that carries a ScriptComponent gets its OWN module
    // namespace (independent globals dict). Per-entity scripts coexist
    // with the legacy single-slot path used by smoke / verify / list_scene
    // (they go through loadScript() + callOnStart/Update/Stop()).
    //
    // Lifecycle of an entity-attached script:
    //   syncFromScene() → attachScript()   on first sight of ScriptComponent
    //   syncFromScene() → reloadScript()   when scriptPath changes
    //   syncFromScene() → detachScript()   when entity destroyed / handle invalid
    //   onBeforeDestroyEntity hook → detachScript(e) (hard cleanup path)
    //
    // Inside callOnStartAll / callOnUpdateAll / callOnStopAll:
    //   * `self` is bound to the live PyEntity handle for `e` BEFORE the
    //     callback runs, so user scripts write `self.get_TransformComponent()`
    //     directly without any global lookup.
    //   * `engine.delta_time` is updated once per frame, then each script's
    //     on_update is invoked (no further delta_time mutation between calls).
    //   * Per-entity faulted flag isolates exceptions: one script throwing
    //     does NOT disable callbacks on other entities.
    //   * ScriptComponent.enabled=false skips that entity entirely (Unity
    //     MonoBehaviour.enabled semantics).
    //
    // attach/detach/reload accept a raw entt::entity + ECSScene*. We do NOT
    // resolve ScriptComponent fields here — the caller is responsible for
    // ensuring the path argument is non-empty and points at a real .py file.
    // ------------------------------------------------------------------
    bool attachScript(class ECSScene &scene, entt::entity e, const std::string &scriptPath);
    void detachScript(entt::entity e);
    bool reloadScript(class ECSScene &scene, entt::entity e);

    // Walks the scene's ScriptComponent view and reconciles it with the
    // internal `entityScripts_` map: attach new entries, reload on path
    // change, detach destroyed entities. Returns the number of scripts
    // that ended up attached afterwards. Safe to call every frame; cheap
    // when nothing changed.
    int syncFromScene(class ECSScene &scene);

    // Per-entity equivalents of callOnStart/Update/Stop. Each iterates
    // entityScripts_, skipping faulted / disabled entities. dt is published
    // ONCE before the loop so all scripts see the same value.
    void callOnStartAll(class ECSScene &scene);
    void callOnUpdateAll(class ECSScene &scene, float deltaTime);
    void callOnStopAll(class ECSScene &scene);

    // Test/inspector helpers: query whether a particular entity currently
    // has an attached EntityScript and inspect its faulted state. Used by
    // the Inspector status badges.
    bool hasEntityScript(entt::entity e) const;
    bool isEntityFaulted(entt::entity e) const;
    std::string getEntityLastError(entt::entity e) const;
    void clearEntityFault(entt::entity e);

    // ------------------------------------------------------------------
    // Per-frame / per-mode hooks
    // ------------------------------------------------------------------
    // State queries (used by the editor)
    // ------------------------------------------------------------------
    bool isInitialized() const { return initialized_; }
    bool hasScript() const { return scriptLoaded_; }
    bool isFaulted() const { return scriptFaulted_; }
    const std::string &currentScriptPath() const { return currentScriptPath_; }

    // Returns a snapshot of the log buffer (stdout/stderr from Python +
    // C++-side error messages). Thread-safe via the internal mutex.
    std::string getLogSnapshot() const;
    void clearLog();

    // Append a line to the log buffer from C++ side. Mostly used by the
    // embedded `engine.log(...)` Python function, but exposed so the
    // editor can also write status lines into the same console.
    void appendLog(const std::string &line);

    // Python stdout / stderr redirection target. The embedded
    // `engine._StdoutRedirect` class (registered in PYBIND11_EMBEDDED_MODULE)
    // forwards every `write()` call here. We split on '\n' internally, prefix
    // with the active EntityScript label (if any) and the supplied stream
    // tag ("stdout" / "stderr"), then funnel everything into logBuffer.
    //
    // This is a public method instead of routing through appendLog: stdout
    // writes arrive in arbitrary fragments (Python may flush "hello" then
    // "\n" separately) and need a small piece of line-buffering state — kept
    // inside ScriptEngine so the redirector class itself stays stateless.
    void writeStdoutChunk(const std::string &chunk, bool isStderr);

    // Push / pop the "current EntityScript" tag used by stdout redirection
    // to prefix lines with `[Entity#<id> Name]`. callOn*All bracket each user
    // callback with these so a `print(...)` inside on_update lands tagged
    // with the right entity.
    //
    // The stack is thread-local (we only run scripts on the main thread, but
    // making it TLS removes any risk of one entity's tag leaking into a
    // re-entrant logger call from a worker thread).
    void pushCallbackContext(const std::string &label);
    void popCallbackContext();

    // Reset the fault flag (called by the Reload button).
    void clearFault() { scriptFaulted_ = false; }

    // ------------------------------------------------------------------
    // Statics for the embedded `engine` module callbacks
    // ------------------------------------------------------------------
    // The PYBIND11_EMBEDDED_MODULE macro defines a free function with no
    // access to `this`. We bridge that by exposing a process-wide singleton
    // pointer that the embedded module functions read. Set during init().
    static ScriptEngine *currentInstance() { return s_instance_; }

private:
    // Forward declared so we can keep all pybind11 types out of the header.
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Renderer *renderer_ = nullptr;
    std::filesystem::path scriptsRoot_{};

    bool initialized_ = false;
    bool scriptLoaded_ = false;
    bool scriptFaulted_ = false;
    std::string currentScriptPath_{};

    // Captured by Python stdout/stderr redirection (added in 13.5) AND by
    // appendLog() / engine.log(). Mutex-protected so we can render it
    // safely from the editor thread later.
    mutable std::mutex logMutex_;
    std::string logBuffer_;

    // Line-buffering state for stdout/stderr redirection. Python typically
    // calls sys.stdout.write("hello"); sys.stdout.write("\n") separately. We
    // accumulate non-newline chunks here and flush a full prefixed line into
    // logBuffer_ when '\n' arrives. Protected by logMutex_.
    std::string stdoutLineBuffer_;
    std::string stderrLineBuffer_;

    static ScriptEngine *s_instance_;

    // Implementation helper: dump Python error state to logBuffer and set
    // scriptFaulted_. Must be called only while a Python exception is live.
    void capturePythonError(const char *contextLabel);
};
