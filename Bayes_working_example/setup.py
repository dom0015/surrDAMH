import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
#mpi_link_args    = os.popen("mpic++ --showme:link").read().strip().split(' ')

ext_modules=[
    Extension("pympi",
              sources            = ["pympi.pyx"],
              libraries=["mpi_stats"],
              library_dirs=["lib"],
              include_dirs=["lib", "/usr/include/mpi"],
              language           = 'c++',
              #extra_compile_args = mpi_compile_args,
              #extra_link_args    = mpi_link_args,
          )
]

setup(
  name = "pympi",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
