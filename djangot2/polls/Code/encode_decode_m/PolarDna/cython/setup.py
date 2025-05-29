from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
                            "Polar_decode_cython.pyx",
                           ],
                          language_level=3, annotate=True)
)
