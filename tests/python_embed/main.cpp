// ============================================================================
// python_embed/main.cpp — minimal CPython embed sanity check.
// ----------------------------------------------------------------------------
// Bypasses the renderer entirely so we can tell whether Phase 13 build issues
// are intrinsic to our environment (Python install, DLL loading) or specific
// to ScriptEngine / pybind11 config.
// ============================================================================

#include <pybind11/embed.h>
#include <iostream>
#include <string>

namespace py = pybind11;

int main()
{
#ifdef RENDERER_PYTHON_HOME
    static std::wstring home = []()
    {
        const std::string s = RENDERER_PYTHON_HOME;
        std::wstring w;
        for (char c : s)
            w.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
        return w;
    }();
    Py_SetPythonHome(home.c_str());
    std::cout << "PYTHONHOME = " << RENDERER_PYTHON_HOME << "\n";
#endif

    std::cout << "Py_GetVersion (pre-init) = " << Py_GetVersion() << "\n";

    try
    {
        py::scoped_interpreter guard{};
        std::cout << "scoped_interpreter constructed OK\n";

        py::module_ sys = py::module_::import("sys");
        std::cout << "sys.version = " << py::cast<std::string>(sys.attr("version")) << "\n";
        std::cout << "sys.prefix  = " << py::cast<std::string>(sys.attr("prefix")) << "\n";
        std::cout << "sys.exec_prefix = " << py::cast<std::string>(sys.attr("exec_prefix")) << "\n";
        py::print("hello from embedded python");
    }
    catch (const py::error_already_set &e)
    {
        std::cerr << "py error: " << e.what() << std::endl;
        return 2;
    }
    catch (const std::exception &e)
    {
        std::cerr << "std exception: " << e.what() << std::endl;
        return 3;
    }
    return 0;
}
