from distutils.core import setup, Extension
import numpy as np
print np.get_include()
# Put this and testmodule.cpp in the same directory
# Open Visual Studio Command Prompt, cd to dir and do:
# set DISTUTILS_USE_SDK=1
# set MSSdk=1
# python setup.py build
# python setup.py install

module1 = Extension('load_extension',
                    sources = ['load_extension.cpp'],
                    include_dirs = [np.get_include(),'C:/opencv/build/include', 'C:/Anaconda2/include', 'C:/warmerda/bld/include','C:/Boost/include/boost-1_60'],
                    library_dirs = ['C:/Anaconda2/libs', 'C:/opencv/build/x64/vc14\lib', 'C:/warmerda/bld/lib', './'],
                    libraries = ['opencv_world310', 'gdal_i', 'python27','libboost_python-vc90-mt-1_60'],
                    extra_compile_args= ['/EHsc']                    

)

setup (name = 'load_extension',
       version = '1.0',
       description = 'This is a test module',
       ext_modules = [module1])