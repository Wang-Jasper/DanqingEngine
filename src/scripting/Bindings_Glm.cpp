// Bindings_Glm — pybind11 bindings for glm vec3 / vec4 / quat / mat4.
//
// Surface area is intentionally narrow: vec3 is the only type scripts
// construct directly (full read/write + arithmetic + dot/length/normalize).
// vec4 / quat / mat4 expose read-only accessors — world matrices must not be
// modified from Python, since Systems::updateTransforms recomputes worldMatrix
// from TRS every frame and a Python-side clobber would desync by the next pass.
//
// This TU is the only place that includes pybind11 + glm together.
// ScriptEngine.cpp calls registerBindings_Glm(m) from the embedded module
// initialiser so all bindings live in `engine.<name>`.

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>

#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace
{
    // Tiny helper: format a float with 4 decimals, mirroring Python's
    // default float __repr__ enough for human inspection without dragging
    // <format> into the project.
    template <typename T>
    std::string fmtFloat(T v)
    {
        std::ostringstream oss;
        oss.precision(4);
        oss << std::fixed << v;
        return oss.str();
    }
} // namespace

// ============================================================================
// registerBindings_Glm — called from ScriptEngine's PYBIND11_EMBEDDED_MODULE
// ============================================================================
void registerBindings_Glm(py::module_ &m)
{
    // ------------------------------------------------------------------
    // glm::vec3 — first-class scriptable value type
    // ------------------------------------------------------------------
    py::class_<glm::vec3>(m, "vec3", "3D float vector (x, y, z).")
        .def(py::init<>(),
             "Default-construct a zero vector.")
        .def(py::init<float>(),
             py::arg("scalar"),
             "Construct (s, s, s).")
        .def(py::init<float, float, float>(),
             py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &glm::vec3::x)
        .def_readwrite("y", &glm::vec3::y)
        .def_readwrite("z", &glm::vec3::z)

        // __getitem__ / __setitem__: keep semantics aligned with x/y/z.
        // Python idiom — raise IndexError for OOB instead of UB-via-assert.
        .def("__getitem__",
             [](const glm::vec3 &v, int i)
             {
                 if (i < 0 || i >= 3)
                     throw py::index_error("vec3 index out of range");
                 return v[i];
             })
        .def("__setitem__",
             [](glm::vec3 &v, int i, float val)
             {
                 if (i < 0 || i >= 3)
                     throw py::index_error("vec3 index out of range");
                 v[i] = val;
             })
        .def("__len__", [](const glm::vec3 &)
             { return 3; })
        .def("__repr__",
             [](const glm::vec3 &v)
             {
                 return "vec3(" + fmtFloat(v.x) + ", " +
                        fmtFloat(v.y) + ", " + fmtFloat(v.z) + ")";
             })

        // Equality / hashing (only equality; vec3 isn't intended as a dict key)
        .def(py::self == py::self)
        .def(py::self != py::self)

        // Arithmetic — return-by-value stays consistent with glm's free fns.
        // We stick to the operators glm itself provides; no /= scalar to
        // avoid divide-by-zero ambiguity in scripts.
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)           // unary negate
        .def(py::self * float())  // vec * scalar
        .def(float() * py::self)  // scalar * vec
        .def(py::self * py::self) // component-wise multiply
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= float())

        // Geometric helpers — all "free function" style (taking *self)
        // because Python users expect `v.dot(w)` and `v.length()`.
        .def("dot", [](const glm::vec3 &a, const glm::vec3 &b)
             { return glm::dot(a, b); }, py::arg("other"))
        .def("cross", [](const glm::vec3 &a, const glm::vec3 &b)
             { return glm::cross(a, b); }, py::arg("other"))
        .def("length", [](const glm::vec3 &v)
             { return glm::length(v); })
        .def("length_squared", [](const glm::vec3 &v)
             { return glm::dot(v, v); })
        .def("normalize", [](const glm::vec3 &v)
             {
                 // Mirror glm: passing a zero vector yields a NaN; we
                 // intercept it for friendlier script-level semantics.
                 float len2 = glm::dot(v, v);
                 if (len2 <= 1e-20f)
                     return glm::vec3(0.0f);
                 return v / glm::sqrt(len2); }, "Return a unit-length copy. Zero vector -> zero (no NaN).");

    // ------------------------------------------------------------------
    // glm::vec4 — read-only sugar
    // ------------------------------------------------------------------
    // Scripts mostly touch vec4 when peeking at quat components / shader-
    // facing data. Full constructors are still allowed because there's
    // no good reason to ban them and they don't risk world-state drift.
    py::class_<glm::vec4>(m, "vec4", "4D float vector (x, y, z, w).")
        .def(py::init<>())
        .def(py::init<float>(), py::arg("scalar"))
        .def(py::init<float, float, float, float>(),
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w"))
        .def_readonly("x", &glm::vec4::x)
        .def_readonly("y", &glm::vec4::y)
        .def_readonly("z", &glm::vec4::z)
        .def_readonly("w", &glm::vec4::w)
        .def("__getitem__",
             [](const glm::vec4 &v, int i)
             {
                 if (i < 0 || i >= 4)
                     throw py::index_error("vec4 index out of range");
                 return v[i];
             })
        .def("__len__", [](const glm::vec4 &)
             { return 4; })
        .def("__repr__",
             [](const glm::vec4 &v)
             {
                 return "vec4(" + fmtFloat(v.x) + ", " + fmtFloat(v.y) +
                        ", " + fmtFloat(v.z) + ", " + fmtFloat(v.w) + ")";
             });

    // ------------------------------------------------------------------
    // glm::quat — read-only handle (modify rotation via Euler/axis helpers
    // surfaced on TransformComponent in Bindings_Components.cpp)
    // ------------------------------------------------------------------
    py::class_<glm::quat>(m, "quat", "Unit quaternion (w, x, y, z).")
        .def(py::init<>(),
             "Identity quaternion.")
        .def(py::init<float, float, float, float>(),
             py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"),
             "Construct directly from components (caller must ensure unit length).")
        .def_readonly("w", &glm::quat::w)
        .def_readonly("x", &glm::quat::x)
        .def_readonly("y", &glm::quat::y)
        .def_readonly("z", &glm::quat::z)
        .def("__getitem__",
             [](const glm::quat &q, int i)
             {
                 // glm::quat indexing convention is (x, y, z, w) for
                 // operator[]; we use the same so chaining with vec4 makes
                 // sense.
                 if (i < 0 || i >= 4)
                     throw py::index_error("quat index out of range");
                 return q[i];
             })
        .def("__repr__",
             [](const glm::quat &q)
             {
                 return "quat(w=" + fmtFloat(q.w) + ", x=" + fmtFloat(q.x) +
                        ", y=" + fmtFloat(q.y) + ", z=" + fmtFloat(q.z) + ")";
             });

    // ------------------------------------------------------------------
    // glm::mat4 — read-only matrix view
    // ------------------------------------------------------------------
    // Indexed access returns columns (vec4) to match glm's column-major
    // convention. This matches what users see when they print glm matrices
    // in C++.
    py::class_<glm::mat4>(m, "mat4", "4x4 column-major float matrix (read-only view).")
        .def(py::init<>(), "Identity matrix.")
        .def(py::init<float>(), py::arg("diagonal"),
             "Construct a diagonal matrix (e.g. mat4(1.0) = identity).")
        .def("__getitem__", [](const glm::mat4 &m, int col) -> glm::vec4
             {
                 if (col < 0 || col >= 4)
                     throw py::index_error("mat4 column index out of range");
                 return m[col]; }, "Return column `col` as a vec4 (column-major).")
        .def("get", [](const glm::mat4 &m, int col, int row)
             {
                 if (col < 0 || col >= 4 || row < 0 || row >= 4)
                     throw py::index_error("mat4 (col,row) out of range");
                 return m[col][row]; }, py::arg("col"), py::arg("row"), "Read element at (col, row).")
        .def("__repr__", [](const glm::mat4 &m)
             {
                 std::ostringstream oss;
                 oss << "mat4(\n";
                 for (int row = 0; row < 4; ++row)
                 {
                     oss << "  [";
                     for (int col = 0; col < 4; ++col)
                     {
                         if (col > 0)
                             oss << ", ";
                         oss << fmtFloat(m[col][row]);
                     }
                     oss << "]\n";
                 }
                 oss << ")";
                 return oss.str(); });
}
