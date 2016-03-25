from distutils.core import setup, Extension
import numpy as np
print np.get_include()
# Put this and testmodule.cpp in the same directory
# Open Visual Studio Command Prompt, cd to dir and do:
# set DISTUTILS_USE_SDK=1
# set MSSdk=1
# python setup.py build
# python setup.py install
includes= [np.get_include(),'C:/opencv/build/include', 'C:/Anaconda2/include', 'C:/warmerda/bld/include','C:/Boost/include/boost-1_60']
lib_dir = ['C:/Anaconda2/libs', './libs']
libs = ['opencv_world310', 'gdal_i', 'python27','libboost_python-vc90-mt-1_60']
extra_args= ['/EHsc'] 

module1 = Extension('load_extension',
                    sources = ['load_extension.cpp'],
                    include_dirs = includes,
                    library_dirs = lib_dir,
                    libraries = libs,
                    extra_compile_args= extra_args
)

setup (name = 'load_extension',
       version = '1.0',
       description = 'This is a module that will load a numpy array with tif and jp2 raster data',
       ext_modules = [module1])
