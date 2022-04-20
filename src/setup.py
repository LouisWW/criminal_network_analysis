"""This script's intention is to compile the pyx files to c.

To compile the file, the following command is used:

python3 setup.py build_ext --inplace

__author__ = Louis Weyland
__date__   = 18/03/2022
"""
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

ext_modules = [
    Extension("sim_mart_vaq_helper_c", ["simulators/sim_mart_vaq_helper_c.pyx"])
]

extensions = cythonize(ext_modules, compiler_directives={"language_level": "3"})
setup(
    name="MyProject",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
