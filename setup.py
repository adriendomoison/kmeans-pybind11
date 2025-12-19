from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError as e:
    raise RuntimeError(
        "pybind11 is required to build the extension. Try: pip install pybind11"
    ) from e

ext_modules = [
    Pybind11Extension(
        "dbscan_cpp",
        ["dbscan_cpp.cpp"],
        cxx_std=17,
    ),
    Pybind11Extension(
        "kmeans_cpp",
        ["kmeans_cpp.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="dbscan_from_scratch",
    version="0.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
