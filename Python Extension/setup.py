from distutils.core import setup, Extension
# Put this and testmodule.cpp in the same directory
# Open Visual Studio Command Prompt, cd to dir and do:
# set DISTUTILS_USE_SDK=1
# set MSSdk=1
# python setup.py build
# python setup.py install

module1 = Extension('testmodule',
                    sources = ['testmodule.cpp'],
                    include_dirs = ['C:\opencv\build\include', 'C:\Anaconda\include', 'C:GDAL\include'],
                    library_dirs = ['C:\Anaconda\libs', 'C:\opencv\build\x64\vc14\lib', 'C:\GDAL\lib'],
                    libraries = ['opencv_world310', 'gdal_i', 'python27'])

setup (name = 'testmodule',
       version = '1.0',
       description = 'This is a test module',
       ext_modules = [module1])