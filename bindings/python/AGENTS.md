# AGENTS.md — bindings/python/

Python package, extension bindings, and Python-facing behavior.

- Source of truth: `bindings/python/pyproject.toml`, `setup.py`, `CMakeLists.txt`.
- Keep API compatibility by default.
- Preserve clear dtype/shape validation and error messages.
- Add tests for every user-visible behavior change.

## pybind11 integration
- **GIL release:** For blocking MKL calls or long-running C++ operations, release GIL with `py::call_guard<py::gil_scoped_release>()`.
- **Type validation:** Explicitly validate NumPy dtypes (float32, float64, uint8 for quantized) and shapes before passing to C++.
- **Memory ownership:** Document lifetime expectations for array views vs copies in docstrings.

## Common failure modes
- Forgetting to update pybind11 wrappers when C++ signatures change in `include/svs/`.
- Not testing edge cases: empty arrays, mismatched dtypes, out-of-bounds indices.
- Returning raw pointers without proper lifetime management — use `py::return_value_policy`.
