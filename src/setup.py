"""This script's intention is to compile the pyx files to c.

To compile the file, the following command is used:

python3 setup.py build_ext --inplace

__author__ = Louis Weyland
__date__   = 18/03/2022
"""
import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

ext_modules = [
    Extension(
        "simulators.sim_mart_vaq_helper_c", ["simulators/sim_mart_vaq_helper_c.pyx"]
    ),
    Extension(
        "network_utils.network_combiner_helper_c",
        ["network_utils/network_combiner_helper_c.pyx"],
        include_dirs=[np.get_include()],
    ),
]

extensions = cythonize(ext_modules, compiler_directives={"language_level": "3"})
setup(
    name="MyProject",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
