// ScriptEngine.cpp - embedded CPython interpreter implementation.
// Only this translation unit (and the Bindings_*.cpp) may include pybind11
// headers; everything else sees ScriptEngine as an opaque PIMPL class.

#include "scripting/ScriptEngine.h"

#include <pybind11/embed.h> // scoped_interpreter, PYBIND11_EMBEDDED_MODULE
#include <pybind11/pybind11.h>

#include "ecs/Components.h" // ScriptComponent
#include "ecs/ECSScene.h"   // registry view in syncFromScene

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace py = pybind11;

// ----------------------------------------------------------------------------
// PIMPL - keep py::scoped_interpreter / py::module_ out of the header so
// nothing else in the engine has to drag in <Python.h>.
// ----------------------------------------------------------------------------
struct ScriptEngine::Impl
{
    // The interpreter itself. RAII: created in init(), destroyed in shutdown().
    // We use a unique_ptr because scoped_interpreter is non-copyable / non-
    // movable AND because we want shutdown() to be idempotent.
    std::unique_ptr<py::scoped_interpreter> interpreter;

    // The currently loaded user module (e.g. `smoke`, `spin_cube`). Kept as
    // a py::object so we can release it explicitly during shutdown(); a bare
    // py::module_ would be tied to the interpreter lifetime by default.
    py::object userModule; // None when no script is loaded

    // Per-entity script storage. Each entity with a ScriptComponent gets its
    // OWN globals dict (independent namespace), independent of the legacy
    // single-slot userModule above.
    //
    // A globals dict rather than a SimpleNamespace like loadScript: we need
    // to inject `self` BEFORE every callback, and SimpleNamespace would
    // require copying the dict back each time. Storing the live globals dict
    // means `globals["self"] = entity` is a single attribute write that the
    // user's `on_update(...)` body sees immediately as `self`.
    struct EntityScript
    {
        entt::entity entity = entt::null;
        std::string scriptPath;        // Relative path (matches ScriptComponent.scriptPath)
        std::filesystem::path absPath; // Cached resolved absolute path
        py::dict globals;              // Per-script namespace (also serves as "module")
        bool faulted = false;
        bool startCalled = false; // Per-entity Play->Stop bookkeeping (enabled toggle)
        std::string lastError;
    };
    std::unordered_map<entt::entity, EntityScript> entityScripts;
};

// Host helper exposed by Bindings_Ecs.cpp; lets us create a PyEntity
// (defined in that TU's anonymous namespace) without leaking the type
// through a header. Used by callOn*All to publish `self`.
namespace pybindings_ecs_internal
{
    py::object makeEntityHandle(ECSScene *scene, entt::entity id);
}

// Process-wide singleton pointer used by the PYBIND11_EMBEDDED_MODULE macro,
// which exposes plain C functions with no `this` available.
ScriptEngine *ScriptEngine::s_instance_ = nullptr;

// Embedded module: `import engine`.
//
// IMPORTANT: this macro must appear at namespace scope and runs at static
// initialisation time. The generated registration is invoked the first time
// `import engine` is executed inside the interpreter.

// Forward declarations for the binding registration entry points implemented
// in Bindings_Glm.cpp / Bindings_Components.cpp / Bindings_Ecs.cpp. Kept
// local here rather than in a shared header because the full pybind11 type
// only matters inside this TU; adding a new Bindings_*.cpp is a one-line
// append both here and in the embedded module body.
void registerBindings_Glm(pybind11::module_ &m);
void registerBindings_Components(pybind11::module_ &m);
// Bindings_Ecs.cpp also exposes pybindings_ecs_internal::publish*() runtime
// helpers ScriptEngine uses to wire actual instances.
void registerBindings_Ecs(pybind11::module_ &m);
namespace pybindings_ecs_internal
{
    void publishContext(pybind11::module_ &engineModule, ECSScene *scene, Camera *camera);
    void publishDeltaTime(pybind11::module_ &engineModule, float dt);
    void clearContext(pybind11::module_ &engineModule);
} // namespace pybindings_ecs_internal

PYBIND11_EMBEDDED_MODULE(engine, m)
{
    m.doc() = "Renderer embedded scripting API (Phase 13).";

    // engine.log(message: str) - append a line to the script console buffer
    // (and mirror to stdout so it also shows up in the terminal).
    m.def(
        "log",
        [](const std::string &msg)
        {
            if (auto *self = ScriptEngine::currentInstance())
            {
                self->appendLog(msg);
            }
            // Mirror to stdout so output is visible even without the console panel.
            std::cout << "[script] " << msg << std::endl;
        },
        py::arg("message"),
        "Append a line to the engine script console.");

    // _StdoutRedirect is a tiny "file-like" Python class assigned to
    // sys.stdout / sys.stderr after the interpreter boots. Every write()
    // call forwards to ScriptEngine::writeStdoutChunk, which does
    // line-buffering, EntityScript prefixing and mirroring to std::cout.
    //
    // It lives in the embedded module (rather than a .py file) so it is
    // available before any user script executes - including the very first
    // `print(...)` from smoke.py / verify_bindings.py.
    //
    // The bound C++ type is a real struct (not int) - py::class_<int> would
    // have every instance alias the CPython small-integer singleton, making
    // two redirectors share one __dict__ and clobber each other's `kind`.
    struct StdoutRedirect
    {
        bool isStderr = false;
    };
    py::class_<StdoutRedirect>(m, "_StdoutRedirect",
                               "Internal: replaces sys.stdout / sys.stderr.")
        .def(py::init([](const std::string &kind)
                      {
                          StdoutRedirect r;
                          r.isStderr = (kind == "stderr");
                          return r; }),
             py::arg("kind") = std::string("stdout"))
        .def("write", [](StdoutRedirect &self, const std::string &text)
             {
                 if (auto *engine = ScriptEngine::currentInstance())
                 {
                     engine->writeStdoutChunk(text, self.isStderr);
                 }
                 // Python's print() expects write() to return the number
                 // of bytes consumed. Returning len(text) keeps it happy.
                 return static_cast<py::ssize_t>(text.size()); }, py::arg("text"))
        .def("flush", [](StdoutRedirect &) {}) // no-op; we line-buffer internally
        .def("isatty", [](StdoutRedirect &)
             { return false; })
        .def("writable", [](StdoutRedirect &)
             { return true; });
    // Glm first: Component bindings reference vec3 etc.
    registerBindings_Glm(m);
    registerBindings_Components(m);
    // Entity/Scene/Camera last: Component method registration relies on the
    // Component classes already existing (looked up by type via pybind11).
    registerBindings_Ecs(m);

    // Seed `engine.scene` / `engine.camera` / `engine.delta_time` to safe
    // defaults so `from engine import scene` doesn't AttributeError before
    // setSceneContext() runs. The actual instances replace these the
    // moment Renderer hands them to us.
    pybindings_ecs_internal::clearContext(m);
}

// ============================================================================
// ScriptEngine - ctor / dtor
// ============================================================================
ScriptEngine::ScriptEngine() : impl_(std::make_unique<Impl>())
{
    // Defer interpreter creation to init() so we can record the renderer
    // pointer / scriptsRoot together. Constructing scoped_interpreter here
    // would force shutdown ordering through the destructor only.
}

ScriptEngine::~ScriptEngine()
{
    shutdown();
}

// ============================================================================
// init() - spin up the interpreter, expose `engine`, prep sys.path
// ============================================================================
void ScriptEngine::init(Renderer &renderer, const std::filesystem::path &scriptsRoot)
{
    if (initialized_)
        return;

    renderer_ = &renderer;
    scriptsRoot_ = scriptsRoot;

    // Publish ourselves before constructing the interpreter so the embedded
    // module callbacks (engine.log) can find us if called during start-up.
    s_instance_ = this;

    try
    {
        // Point CPython at the real install root before scoped_interpreter
        // triggers Py_Initialize. Without this, on Windows 11 systems where
        // PATH starts with the Microsoft Store `WindowsApps\python3.exe`
        // stub, Py_Initialize() picks that directory as PYTHONHOME, fails to
        // find encodings/os, and aborts with "ImportError: initialization
        // failed".
        //
        // RENDERER_PYTHON_HOME / RENDERER_PYTHON_STDLIB are baked in by
        // CMakeLists.txt from the same Python3 install pybind11 linked
        // against, so they always agree with python3X.dll's expectations.
#ifdef RENDERER_PYTHON_HOME
        {
            // Convert the UTF-8 macro string to the wide-char encoding
            // CPython expects on Windows. We keep the wstring alive for
            // the entire process: Py_SetPythonHome stores the pointer,
            // not a copy, so a temporary would dangle the moment the
            // try block exits.
            static std::wstring pythonHomeW = []()
            {
                const std::string home = RENDERER_PYTHON_HOME;
                std::wstring w;
                w.reserve(home.size());
                for (char c : home)
                    w.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
                return w;
            }();
            if (!pythonHomeW.empty())
            {
                Py_SetPythonHome(pythonHomeW.c_str());
                std::cout << "[ScriptEngine] PYTHONHOME = " << RENDERER_PYTHON_HOME << "\n";
            }
        }
#endif

        // Construct the interpreter. pybind11 handles all PyConfig wiring.
        impl_->interpreter = std::make_unique<py::scoped_interpreter>();

        // Sanity log - confirms which Python install actually got picked up
        // and that sys.prefix matches the baked-in PYTHONHOME.
        try
        {
            py::module_ sysMod = py::module_::import("sys");
            std::cout << "[ScriptEngine] Python "
                      << py::cast<std::string>(sysMod.attr("version").attr("split")(" ").attr("__getitem__")(0))
                      << " @ "
                      << py::cast<std::string>(sysMod.attr("prefix")) << "\n";
        }
        catch (...)
        {
            // Diagnostics only - never fatal. If sys is unimportable we'll
            // hit the next import("engine") and surface a real error there.
        }

        // Make sure `import engine` resolves: the PYBIND11_EMBEDDED_MODULE
        // registration is automatic, but we trigger an import here so any
        // failure surfaces at init() rather than at first script load.
        py::module_::import("engine");

        // Add `${PROJECT_DIR}/scripts/` to sys.path so user scripts loaded
        // by name resolve, and so scripts can `import` helper modules from
        // the same folder.
        if (!scriptsRoot_.empty())
        {
            const std::string rootStr = scriptsRoot_.generic_string();
            py::module_::import("sys").attr("path").attr("insert")(0, rootStr);
        }

        // Replace sys.stdout / sys.stderr with engine._StdoutRedirect so
        // every Python `print(...)` (and tracebacks emitted via PyErr_Print
        // before our explicit handlers run) lands in logBuffer_. Done after
        // the `engine` import above so the class is guaranteed registered,
        // and inside init's try block so failures hit capturePythonError.
        try
        {
            py::module_ sysMod = py::module_::import("sys");
            py::module_ engMod = py::module_::import("engine");
            sysMod.attr("stdout") = engMod.attr("_StdoutRedirect")(py::str("stdout"));
            sysMod.attr("stderr") = engMod.attr("_StdoutRedirect")(py::str("stderr"));
        }
        catch (const py::error_already_set &e)
        {
            // Non-fatal: scripts can still run without redirection (output goes
            // to the real terminal). Log the failure and continue.
            std::cerr << "[ScriptEngine] sys.stdout redirect install failed: "
                      << e.what() << std::endl;
        }

        initialized_ = true;
        std::cout << "[ScriptEngine] CPython interpreter initialised. scripts root: "
                  << scriptsRoot_.generic_string() << std::endl;
    }
    catch (const py::error_already_set &e)
    {
        // Print the pybind11-cached exception text (what()) AND attempt the
        // structured traceback path. The first is reliable even when the
        // CPython error state has already been drained; the second gives
        // line numbers when it works. We deliberately keep both so a future
        // regression doesn't mask itself behind "NoneType: None".
        std::cerr << "[ScriptEngine] init failed (Python): " << e.what() << std::endl;
        appendLog(std::string("[ScriptEngine] init failed (Python): ") + e.what());
        capturePythonError("ScriptEngine::init");
        impl_->interpreter.reset();
        s_instance_ = nullptr;
        initialized_ = false;
    }
    catch (const std::exception &e)
    {
        appendLog(std::string("[ScriptEngine] init failed: ") + e.what());
        std::cerr << "[ScriptEngine] init failed: " << e.what() << std::endl;
        impl_->interpreter.reset();
        s_instance_ = nullptr;
        initialized_ = false;
    }
}

// ============================================================================
// shutdown() - destroy user module first, then interpreter. Idempotent.
// ============================================================================
void ScriptEngine::shutdown()
{
    if (!initialized_)
    {
        // Even when init() failed we want to clear the singleton pointer
        // and ensure impl_ doesn't carry a half-built interpreter.
        s_instance_ = nullptr;
        if (impl_)
            impl_->interpreter.reset();
        return;
    }

    // Drop any singleton attributes pointing at host-owned ECSScene / Camera
    // before the interpreter goes down. This guarantees `from engine import
    // scene` returns a non-bound wrapper after shutdown rather than a stale
    // pointer (helps if someone keeps the module in a Python-side variable
    // across reloads).
    try
    {
        clearSceneContext();
    }
    catch (...)
    {
        // Best-effort: if Python already errored we still want to release
        // the interpreter below.
    }

    // Release the user module BEFORE the interpreter goes away - its
    // destructor calls into Python and would crash post-Py_Finalize.
    if (impl_)
    {
        try
        {
            // Drop all per-entity script globals first. Each dict holds
            // Python objects (functions, captured locals) that would
            // otherwise be reclaimed by the interpreter destructor out of
            // order, raising warnings in CPython >= 3.11.
            impl_->entityScripts.clear();
            impl_->userModule = py::object{}; // drop ref under live interpreter
        }
        catch (...)
        {
            // Even if Python is in a bad state we proceed; we are shutting
            // down, the worst outcome is a leaked PyObject which the
            // interpreter destructor will reclaim.
        }
        impl_->interpreter.reset();
    }

    scriptLoaded_ = false;
    scriptFaulted_ = false;
    currentScriptPath_.clear();
    initialized_ = false;
    s_instance_ = nullptr;

    std::cout << "[ScriptEngine] CPython interpreter shut down.\n";
}

// ============================================================================
// loadScript() - exec a .py file as a module and stash the resulting object
// ============================================================================
bool ScriptEngine::loadScript(const std::filesystem::path &scriptPath)
{
    if (!initialized_)
    {
        appendLog("[ScriptEngine] loadScript called before init().");
        return false;
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(scriptPath, ec))
    {
        appendLog("[ScriptEngine] Script not found: " + scriptPath.generic_string());
        return false;
    }

    try
    {
        // We deliberately use exec() rather than import_module() here:
        //   - exec() does not require the file to live on sys.path
        //   - we get a fresh namespace dict on every (re)load, which makes
        //     reload behaviour trivial without importlib.reload()
        const std::string pathStr = scriptPath.generic_string();
        py::dict globals;
        globals["__name__"] = py::str("__user_script__");
        globals["__file__"] = py::str(pathStr);

        // Read file via Python so encoding (utf-8 / BOM) is consistent with
        // how `import` would have handled it.
        py::object builtins = py::module_::import("builtins");
        py::object openFn = builtins.attr("open");
        py::object fileObj = openFn(pathStr, "r", py::arg("encoding") = "utf-8");
        std::string source = py::cast<std::string>(fileObj.attr("read")());
        fileObj.attr("close")();

        // compile() so syntax errors point at the right filename in tracebacks.
        py::object compileFn = builtins.attr("compile");
        py::object code = compileFn(source, pathStr, "exec");

        // exec() the compiled code in our globals dict; user `def on_update`
        // etc. will land in `globals` and we keep that dict as the module.
        py::object execFn = builtins.attr("exec");
        execFn(code, globals);

        // SimpleNamespace gives us attribute access (mod.on_update) over
        // an arbitrary dict without inventing a custom class.
        py::object types = py::module_::import("types");
        py::object simpleNs = types.attr("SimpleNamespace");

        // Forward each callable / value we care about into the namespace.
        // We just splat the whole dict - SimpleNamespace(**globals) is fine.
        impl_->userModule = simpleNs(**globals);

        currentScriptPath_ = pathStr;
        scriptLoaded_ = true;
        scriptFaulted_ = false;

        appendLog("[ScriptEngine] Loaded script: " + pathStr);
        return true;
    }
    catch (const py::error_already_set &)
    {
        capturePythonError("ScriptEngine::loadScript");
        return false;
    }
    catch (const std::exception &e)
    {
        appendLog(std::string("[ScriptEngine] loadScript failed: ") + e.what());
        return false;
    }
}

// ============================================================================
// callOnStart / callOnUpdate / callOnStop
// ----------------------------------------------------------------------------
// Hooks invoked by EditorUI / Renderer. Optional on
// the script side: if the function is missing we silently no-op.
// ============================================================================
namespace
{
    // Common helper: call `module.func(*args)` if it exists. Returns true
    // when the call ran (success or not), false when the attribute was
    // simply absent (so the caller can choose to log or stay silent).
    template <typename... Args>
    bool tryCall(py::object &mod, const char *funcName, Args &&...args)
    {
        if (!mod || mod.is_none())
            return false;
        if (!py::hasattr(mod, funcName))
            return false;
        py::object fn = mod.attr(funcName);
        if (!py::isinstance<py::function>(fn) && !PyCallable_Check(fn.ptr()))
            return false;
        fn(std::forward<Args>(args)...);
        return true;
    }
} // namespace

// ============================================================================
// Logging helpers
// ============================================================================
std::string ScriptEngine::getLogSnapshot() const
{
    std::lock_guard<std::mutex> lk(logMutex_);
    return logBuffer_;
}

void ScriptEngine::clearLog()
{
    std::lock_guard<std::mutex> lk(logMutex_);
    logBuffer_.clear();
}

void ScriptEngine::appendLog(const std::string &line)
{
    std::lock_guard<std::mutex> lk(logMutex_);
    logBuffer_.append(line);
    if (line.empty() || line.back() != '\n')
        logBuffer_.push_back('\n');
}

// Python's print() typically calls sys.stdout.write("hello");
// sys.stdout.write("\n") as two separate calls, so we line-buffer here. Each
// chunk gets concatenated into stdoutLineBuffer_ / stderrLineBuffer_;
// whenever we see '\n', we flush the buffered line into logBuffer_ with the
// active EntityScript prefix (if any) and a stream tag. Mirrored to
// std::cout / std::cerr so terminal users still see output even without the
// Script Console panel open.
namespace
{
    // Thread-local stack of EntityScript labels. We push before running a
    // user callback (callOnStartAll / callOnUpdateAll / callOnStopAll) and
    // pop in the matching scope guard. The top label, if non-empty, is
    // prepended to every stdout/stderr line.
    //
    // Why thread-local? Scripts only run on the main thread today, but any
    // future worker-thread invocation (e.g. NN inference) must not leak its
    // tag into another thread's logger. TLS gives that for free.
    thread_local std::vector<std::string> tls_callbackContextStack;

    const std::string &currentContextLabel()
    {
        static const std::string empty;
        return tls_callbackContextStack.empty() ? empty : tls_callbackContextStack.back();
    }

    // Build a short "#<id> Name" label for an entity, used as
    // the active callback context tag so per-entity stdout writes get
    // prefixed with the right entity in the Script Console.
    std::string makeEntityLabel(ECSScene &scene, entt::entity e, const char *hookName)
    {
        std::ostringstream oss;
        oss << hookName << " #" << static_cast<uint32_t>(e);
        if (scene.registry.valid(e) && scene.registry.all_of<NameComponent>(e))
        {
            const auto &nc = scene.registry.get<NameComponent>(e);
            if (!nc.name.empty())
                oss << " " << nc.name;
        }
        return oss.str();
    }

    // RAII guard pairing pushCallbackContext / popCallbackContext, so any
    // exception escaping the user callback still pops cleanly. Using a
    // helper class keeps the three callOn*All bodies uncluttered.
    struct ScopedCallbackContext
    {
        ScriptEngine *engine;
        explicit ScopedCallbackContext(ScriptEngine *eng, const std::string &label)
            : engine(eng)
        {
            engine->pushCallbackContext(label);
        }
        ~ScopedCallbackContext() { engine->popCallbackContext(); }
        ScopedCallbackContext(const ScopedCallbackContext &) = delete;
        ScopedCallbackContext &operator=(const ScopedCallbackContext &) = delete;
    };
} // namespace

void ScriptEngine::pushCallbackContext(const std::string &label)
{
    tls_callbackContextStack.push_back(label);
}

void ScriptEngine::popCallbackContext()
{
    if (!tls_callbackContextStack.empty())
        tls_callbackContextStack.pop_back();
}

void ScriptEngine::writeStdoutChunk(const std::string &chunk, bool isStderr)
{
    if (chunk.empty())
        return;

    // Line-buffer state lives on the engine instance (not TLS) - Python's
    // GIL guarantees only one thread writes at a time, and we want sequential
    // chunks ("hello", "\n") to coalesce regardless of which call site they
    // come from. Mutex is reused for the line buffer too.
    std::string completedLines; // accumulate fully-terminated lines for one log append

    {
        std::lock_guard<std::mutex> lk(logMutex_);
        std::string &buf = isStderr ? stderrLineBuffer_ : stdoutLineBuffer_;

        for (char c : chunk)
        {
            if (c == '\n')
            {
                // Build prefix lazily: tag + optional [Entity#xxx Name].
                // Format kept short so the console stays readable.
                const std::string &ctx = currentContextLabel();
                std::string prefix = isStderr ? "[stderr] " : "[stdout] ";
                if (!ctx.empty())
                {
                    prefix += "[" + ctx + "] ";
                }
                logBuffer_.append(prefix);
                logBuffer_.append(buf);
                logBuffer_.push_back('\n');

                // Also collect for the std::cout mirror below (after we
                // release the mutex, since std::cout itself may block).
                completedLines.append(prefix);
                completedLines.append(buf);
                completedLines.push_back('\n');

                buf.clear();
            }
            else
            {
                buf.push_back(c);
            }
        }
    } // unlock before mirroring to terminal

    if (!completedLines.empty())
    {
        // Mirror to the real terminal so out-of-process tooling (CI logs,
        // tail -f) still sees script output. stderr stays on stderr to
        // preserve any color-coding tooling downstream applies.
        if (isStderr)
            std::cerr << completedLines;
        else
            std::cout << completedLines;
    }
}

// ============================================================================
// Scene / Camera / delta_time module attributes
// ----------------------------------------------------------------------------
// These wrap the helpers in pybindings_ecs_internal so the renderer never
// has to touch pybind11 directly. All three guard against being called
// without a live interpreter - silent no-op so Renderer can call
// them unconditionally during init/shutdown sequencing.
// ============================================================================
void ScriptEngine::setSceneContext(ECSScene *scene, Camera *camera)
{
    if (!initialized_)
        return;
    try
    {
        py::module_ engineModule = py::module_::import("engine");
        pybindings_ecs_internal::publishContext(engineModule, scene, camera);
    }
    catch (const py::error_already_set &)
    {
        capturePythonError("ScriptEngine::setSceneContext");
    }
}

void ScriptEngine::clearSceneContext()
{
    if (!initialized_)
        return;
    try
    {
        py::module_ engineModule = py::module_::import("engine");
        pybindings_ecs_internal::clearContext(engineModule);
    }
    catch (const py::error_already_set &)
    {
        // Don't recurse into capturePythonError if we're shutting down -
        // just swallow. Logging here would get attached to the next
        // unrelated context label, which is misleading.
        PyErr_Clear();
    }
}

void ScriptEngine::setDeltaTime(float dt)
{
    if (!initialized_)
        return;
    try
    {
        py::module_ engineModule = py::module_::import("engine");
        pybindings_ecs_internal::publishDeltaTime(engineModule, dt);
    }
    catch (const py::error_already_set &)
    {
        // Same rationale as clearSceneContext: dt updates fire every frame,
        // logging a Python error every frame would saturate the console.
        PyErr_Clear();
    }
}

// ============================================================================
// capturePythonError - read the live exception, write traceback to logBuffer
// ----------------------------------------------------------------------------
// MUST only be called while a Python exception is currently set (i.e. inside
// a `catch (py::error_already_set&)` block). Uses `traceback.format_exception`
// to produce the same multi-line output you would see at the REPL.
// ============================================================================
void ScriptEngine::capturePythonError(const char *contextLabel)
{
    scriptFaulted_ = true;

    std::ostringstream oss;
    oss << "[ScriptEngine] Python error in " << (contextLabel ? contextLabel : "<unknown>") << ":\n";

    try
    {
        // Pull the current exception out of the interpreter into Python
        // values we can format. PyErr_Fetch / Restore would also work but
        // pybind11's error_scope handles ownership cleanly.
        PyObject *type = nullptr;
        PyObject *value = nullptr;
        PyObject *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        if (traceback)
            PyException_SetTraceback(value, traceback);

        // Wrap into pybind11 handles for safe ref counting.
        py::object pyType = type ? py::reinterpret_steal<py::object>(type) : py::none();
        py::object pyValue = value ? py::reinterpret_steal<py::object>(value) : py::none();
        py::object pyTb = traceback ? py::reinterpret_steal<py::object>(traceback) : py::none();

        py::module_ traceMod = py::module_::import("traceback");
        py::object formatted = traceMod.attr("format_exception")(pyType, pyValue, pyTb);

        for (auto item : formatted)
        {
            oss << py::cast<std::string>(item);
        }
    }
    catch (...)
    {
        oss << "(failed to format Python traceback)\n";
    }

    appendLog(oss.str());

    // Mirror to stderr so terminal users see it even without the script
    // console panel.
    std::cerr << oss.str();
}

// ============================================================================
// Per-Entity script management implementation
// ----------------------------------------------------------------------------
// Why a fresh globals dict per entity?
//   * Independent variable namespace: two entities running the same .py
//     file don't share `_target_entity` / `_starting_rotation` etc.
//   * Per-call `self` injection: we write `globals["self"] = handle` right
//     before the call so `def on_update(self, dt): self.get_TransformComponent()`
//     reads the live ECS handle without any global lookup.
//   * Reload semantics match the legacy loadScript path: we just re-exec
//     the file contents into a brand-new globals dict.
//
// We do NOT touch `sys.modules`. That was the old plan, but using sys.modules
// would force every per-entity instance to share Python's module cache,
// defeating the point of independent namespaces. Using a dict-as-namespace
// (same trick legacy loadScript already uses with SimpleNamespace) keeps
// us cache-free.
// ============================================================================
namespace
{
    // Shared file->source loader used by both attach and reload. Encapsulates
    // the "open with Python so encodings/BOM agree, compile so tracebacks
    // print the real filename, exec into the supplied globals" sequence.
    void execScriptInto(const std::filesystem::path &abs, py::dict &globals)
    {
        const std::string pathStr = abs.generic_string();
        // Stamp __name__ / __file__ so introspection inside the script
        // (e.g. logger formatting) sees a sensible identity.
        globals["__name__"] = py::str("__entity_script__");
        globals["__file__"] = py::str(pathStr);

        py::object builtins = py::module_::import("builtins");
        py::object openFn = builtins.attr("open");
        py::object fileObj = openFn(pathStr, "r", py::arg("encoding") = "utf-8");
        std::string source = py::cast<std::string>(fileObj.attr("read")());
        fileObj.attr("close")();

        py::object compileFn = builtins.attr("compile");
        py::object code = compileFn(source, pathStr, "exec");

        py::object execFn = builtins.attr("exec");
        execFn(code, globals);
    }

    // Resolve a ScriptComponent.scriptPath against the engine's scripts
    // root. Empty input => empty output (caller must guard). If the input
    // is already absolute we leave it alone; otherwise append to root.
    std::filesystem::path resolveScriptAbsPath(const std::filesystem::path &root,
                                               const std::string &rel)
    {
        if (rel.empty())
            return {};
        std::filesystem::path p(rel);
        if (p.is_absolute())
            return p;
        return root / p;
    }

    // Try to call `globals[funcName](*args)` if present + callable. Behaves
    // exactly like the legacy `tryCall` helper but operates on a dict
    // instead of a SimpleNamespace.
    template <typename... Args>
    bool tryCallDict(py::dict &globals, const char *funcName, Args &&...args)
    {
        if (!globals.contains(funcName))
            return false;
        py::object fn = globals[funcName];
        if (!fn || fn.is_none())
            return false;
        if (!py::isinstance<py::function>(fn) && !PyCallable_Check(fn.ptr()))
            return false;
        fn(std::forward<Args>(args)...);
        return true;
    }
} // namespace

bool ScriptEngine::attachScript(ECSScene &scene, entt::entity e, const std::string &scriptPath)
{
    if (!initialized_)
    {
        appendLog("[ScriptEngine] attachScript called before init().");
        return false;
    }
    if (!scene.registry.valid(e))
    {
        appendLog("[ScriptEngine] attachScript: invalid entity handle.");
        return false;
    }
    if (scriptPath.empty())
    {
        // Empty path is a perfectly valid "ScriptComponent placeholder"
        // (Inspector showing 'Empty'). Treat as a successful no-op so
        // syncFromScene doesn't keep hammering us.
        return true;
    }

    // If we already have an EntityScript for this entity, drop it first.
    // Centralises the "rebind on path change" path through reloadScript.
    auto it = impl_->entityScripts.find(e);
    if (it != impl_->entityScripts.end())
    {
        // Best-effort: nothing dangerous if globals dict is dropped here;
        // only Python objects with __del__ side-effects would notice.
        impl_->entityScripts.erase(it);
    }

    const std::filesystem::path abs = resolveScriptAbsPath(scriptsRoot_, scriptPath);
    std::error_code ec;
    if (!std::filesystem::exists(abs, ec))
    {
        appendLog("[ScriptEngine] attachScript: file not found: " + abs.generic_string());
        // Still record an EntityScript so syncFromScene doesn't retry every
        // frame - but mark it faulted with a clear lastError. Inspector
        // will surface the red badge.
        Impl::EntityScript es;
        es.entity = e;
        es.scriptPath = scriptPath;
        es.absPath = abs;
        es.faulted = true;
        es.lastError = "Script file not found: " + abs.generic_string();
        // Mirror into ScriptComponent so the Inspector's status badge
        // reflects the failure even before a callback runs.
        if (auto *sc = scene.registry.try_get<ScriptComponent>(e))
        {
            sc->loaded = false;
            sc->faulted = true;
            sc->lastError = es.lastError;
        }
        impl_->entityScripts.emplace(e, std::move(es));
        return false;
    }

    try
    {
        Impl::EntityScript es;
        es.entity = e;
        es.scriptPath = scriptPath;
        es.absPath = abs;
        es.globals = py::dict(); // fresh, isolated namespace
        execScriptInto(abs, es.globals);

        // Mirror the loaded state back into ScriptComponent so the Inspector
        // doesn't have to query ScriptEngine for a yes/no.
        if (auto *sc = scene.registry.try_get<ScriptComponent>(e))
        {
            sc->loaded = true;
            sc->faulted = false;
            sc->lastError.clear();
        }

        appendLog("[ScriptEngine] Attached '" + scriptPath + "' to entity #" +
                  std::to_string(static_cast<uint32_t>(e)));
        impl_->entityScripts.emplace(e, std::move(es));
        return true;
    }
    catch (const py::error_already_set &)
    {
        // Capture into the global log for visibility, then also stash a
        // per-entity message so the Inspector can show the red badge.
        capturePythonError(("attachScript[" + scriptPath + "]").c_str());
        Impl::EntityScript es;
        es.entity = e;
        es.scriptPath = scriptPath;
        es.absPath = abs;
        es.faulted = true;
        es.lastError = "Exception during attach (see Script Console for traceback)";
        if (auto *sc = scene.registry.try_get<ScriptComponent>(e))
        {
            sc->loaded = false;
            sc->faulted = true;
            sc->lastError = es.lastError;
        }
        impl_->entityScripts.emplace(e, std::move(es));
        return false;
    }
}

void ScriptEngine::detachScript(entt::entity e)
{
    if (!initialized_)
    {
        // If the interpreter is already gone there is nothing to free.
        // entityScripts may still hold dead handles; clear() in shutdown()
        // will mop them up under the GIL.
        return;
    }
    auto it = impl_->entityScripts.find(e);
    if (it == impl_->entityScripts.end())
        return;
    // Drop the globals dict explicitly so any captured `self` references
    // release their hold on PyEntity wrappers BEFORE the entity itself
    // disappears from the registry.
    try
    {
        it->second.globals = py::dict();
    }
    catch (...)
    {
        // GIL release path during shutdown can throw; not fatal here.
    }
    impl_->entityScripts.erase(it);
}

bool ScriptEngine::reloadScript(ECSScene &scene, entt::entity e)
{
    if (!initialized_)
        return false;
    auto it = impl_->entityScripts.find(e);
    if (it == impl_->entityScripts.end())
    {
        // Reloading something we never attached: defer to attachScript via
        // ScriptComponent. Caller usually goes through Inspector, which
        // already wrote the new path into the component before pressing
        // Reload - so look it up here.
        if (auto *sc = scene.registry.try_get<ScriptComponent>(e))
        {
            return attachScript(scene, e, sc->scriptPath);
        }
        return false;
    }
    // Capture the path before we drop the entry; attachScript clears it.
    std::string path = it->second.scriptPath;
    // Allow the caller to have updated ScriptComponent.scriptPath; pick
    // the freshest value so Inspector edits survive Reload.
    if (auto *sc = scene.registry.try_get<ScriptComponent>(e))
    {
        if (!sc->scriptPath.empty())
            path = sc->scriptPath;
    }
    return attachScript(scene, e, path);
}

int ScriptEngine::syncFromScene(ECSScene &scene)
{
    if (!initialized_)
        return 0;

    auto &reg = scene.registry;

    // Pass 1: walk ScriptComponent view, attach/reload as needed.
    auto view = reg.view<ScriptComponent>();
    for (auto e : view)
    {
        auto &sc = view.get<ScriptComponent>(e);
        auto it = impl_->entityScripts.find(e);

        if (it == impl_->entityScripts.end())
        {
            // Brand-new attachment (or first sight after scene load).
            if (!sc.scriptPath.empty())
            {
                attachScript(scene, e, sc.scriptPath);
            }
            continue;
        }

        // Already attached; check for path change. We compare against the
        // EntityScript's cached scriptPath rather than re-resolving paths,
        // because ScriptComponent.scriptPath is the user-facing source of
        // truth. Empty path on the component = user blanked it => detach.
        if (sc.scriptPath.empty())
        {
            detachScript(e);
            continue;
        }
        if (sc.scriptPath != it->second.scriptPath)
        {
            attachScript(scene, e, sc.scriptPath); // also drops the old entry
        }
    }

    // Pass 2: detect entries whose ScriptComponent disappeared (e.g. the
    // user removed the component but kept the entity). We can't iterate
    // unordered_map while erasing through detachScript, so collect first.
    std::vector<entt::entity> stale;
    for (auto &kv : impl_->entityScripts)
    {
        if (!reg.valid(kv.first) || !reg.all_of<ScriptComponent>(kv.first))
            stale.push_back(kv.first);
    }
    for (auto e : stale)
        detachScript(e);

    return static_cast<int>(impl_->entityScripts.size());
}

void ScriptEngine::callOnStartAll(ECSScene &scene)
{
    if (!initialized_)
        return;
    auto &reg = scene.registry;
    for (auto &kv : impl_->entityScripts)
    {
        auto e = kv.first;
        auto &es = kv.second;
        if (!reg.valid(e) || es.faulted)
            continue;
        // Per-entity enabled gate (Unity MonoBehaviour.enabled semantics).
        if (auto *sc = reg.try_get<ScriptComponent>(e))
        {
            if (!sc->enabled)
                continue;
        }
        ScopedCallbackContext ctx(this, makeEntityLabel(scene, e, "on_start"));
        try
        {
            // Inject `self` (live PyEntity) and `engine` (the embedded
            // module) so the script body can do `self.get_*()` directly
            // and access bindings without an `import engine` line.
            es.globals["self"] = pybindings_ecs_internal::makeEntityHandle(&scene, e);
            es.globals["engine"] = py::module_::import("engine");
            tryCallDict(es.globals, "on_start", es.globals["self"]);
            es.startCalled = true;
        }
        catch (const py::error_already_set &)
        {
            es.faulted = true;
            std::ostringstream label;
            label << "on_start[entity #" << static_cast<uint32_t>(e) << "]";
            capturePythonError(label.str().c_str());
            es.lastError = "Exception in on_start (see Script Console)";
            if (auto *sc = reg.try_get<ScriptComponent>(e))
            {
                sc->faulted = true;
                sc->lastError = es.lastError;
            }
        }
    }
}

void ScriptEngine::callOnUpdateAll(ECSScene &scene, float deltaTime)
{
    if (!initialized_)
        return;
    // Publish dt once; matches legacy callOnUpdate's contract that scripts
    // can read either the on_update argument or `engine.delta_time`.
    setDeltaTime(deltaTime);

    auto &reg = scene.registry;
    for (auto &kv : impl_->entityScripts)
    {
        auto e = kv.first;
        auto &es = kv.second;
        if (!reg.valid(e) || es.faulted)
            continue;
        if (auto *sc = reg.try_get<ScriptComponent>(e))
        {
            if (!sc->enabled)
                continue;
        }
        ScopedCallbackContext ctx(this, makeEntityLabel(scene, e, "on_update"));
        try
        {
            es.globals["self"] = pybindings_ecs_internal::makeEntityHandle(&scene, e);
            es.globals["engine"] = py::module_::import("engine");
            tryCallDict(es.globals, "on_update", es.globals["self"], deltaTime);
        }
        catch (const py::error_already_set &)
        {
            es.faulted = true;
            std::ostringstream label;
            label << "on_update[entity #" << static_cast<uint32_t>(e) << "]";
            capturePythonError(label.str().c_str());
            es.lastError = "Exception in on_update (see Script Console)";
            if (auto *sc = reg.try_get<ScriptComponent>(e))
            {
                sc->faulted = true;
                sc->lastError = es.lastError;
            }
        }
    }
}

void ScriptEngine::callOnStopAll(ECSScene &scene)
{
    if (!initialized_)
        return;
    auto &reg = scene.registry;
    for (auto &kv : impl_->entityScripts)
    {
        auto e = kv.first;
        auto &es = kv.second;
        if (!reg.valid(e) || es.faulted)
        {
            es.startCalled = false; // even faulted scripts get reset for next Play
            continue;
        }
        // Only fire on_stop if we actually fired on_start for this entity.
        // Avoids the surprising "Stop without Play" path firing user code
        // (e.g. when Play press itself failed and Stop is the recovery).
        if (!es.startCalled)
            continue;
        if (auto *sc = reg.try_get<ScriptComponent>(e))
        {
            if (!sc->enabled)
            {
                es.startCalled = false;
                continue;
            }
        }
        ScopedCallbackContext ctx(this, makeEntityLabel(scene, e, "on_stop"));
        try
        {
            es.globals["self"] = pybindings_ecs_internal::makeEntityHandle(&scene, e);
            es.globals["engine"] = py::module_::import("engine");
            tryCallDict(es.globals, "on_stop", es.globals["self"]);
        }
        catch (const py::error_already_set &)
        {
            es.faulted = true;
            std::ostringstream label;
            label << "on_stop[entity #" << static_cast<uint32_t>(e) << "]";
            capturePythonError(label.str().c_str());
            es.lastError = "Exception in on_stop (see Script Console)";
            if (auto *sc = reg.try_get<ScriptComponent>(e))
            {
                sc->faulted = true;
                sc->lastError = es.lastError;
            }
        }
        es.startCalled = false;
    }
}

bool ScriptEngine::hasEntityScript(entt::entity e) const
{
    return impl_ && impl_->entityScripts.find(e) != impl_->entityScripts.end();
}

bool ScriptEngine::isEntityFaulted(entt::entity e) const
{
    if (!impl_)
        return false;
    auto it = impl_->entityScripts.find(e);
    return it != impl_->entityScripts.end() && it->second.faulted;
}

std::string ScriptEngine::getEntityLastError(entt::entity e) const
{
    if (!impl_)
        return {};
    auto it = impl_->entityScripts.find(e);
    return it != impl_->entityScripts.end() ? it->second.lastError : std::string{};
}

void ScriptEngine::clearEntityFault(entt::entity e)
{
    if (!impl_)
        return;
    auto it = impl_->entityScripts.find(e);
    if (it == impl_->entityScripts.end())
        return;
    it->second.faulted = false;
    it->second.lastError.clear();
}
