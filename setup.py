#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()


requires = ['numpy >=1.8', #'PyQt5','Pillow',
            'scipy >=0.14',
            'matplotlib>=3.1.1',
            'astropy >=1.3',
            'pyds9',
            'photutils',
            'tqdm',
            'pyvista',#==0.25.3
            'datetime',
            'pandas',
            'argparse',
            'PyQt5',
            'dataphile']

entry_points = {}
entry_points['console_scripts'] = ['DS9Utils = pyds9plugin.DS9Utils:main']

#data = { "pyds9plugin": ["QuickLookPlugIn.ds9.ans","config/*","Images/stack????????.fits","Images/stack.fits","Sextractor/*","dataphile/*","doc/ref/*/*","doc/ref/*"]}#,"doc/features_files/*","doc/img/*","doc/index_files/*"
#data = { "pyds9plugin": ["QuickLookPlugIn.ds9.ans","Macros/*","Macros/Macros_Header_catalog/*","filters/*","SEDs/*","config/*","Images/stack????????.fits","Images/m33_hi.fits","Images/stack.fits","Sextractor/*","dataphile/*","doc/ref/examples/*","doc/ref/img/*","doc/ref/index_files/*","doc/ref/*.html"]}
data = { "pyds9plugin": ["pyds9plugin/QuickLookPlugIn.ds9.ans","Macros/*","Macros/Macros_Header_catalog/*","Images/stack????????.fits","Images/m33_hi.fits","Images/stack.fits","Sextractor/*","doc/ref/examples/*","doc/ref/img/*","doc/ref/index_files/*","doc/ref/*.html"]}
#,"doc/ref/*/*","doc/features_files/*","doc/img/*","doc/index_files/*"
#pyds9plugin/QuickLookPlugIn.ds9.ans

MAJOR = '3'
MINOR = '0'
MICRO = '1dev5'
version = '%s.%s%s' % (MAJOR, MINOR, MICRO)




class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        # print("\033[32mPackage installed. \nRun 'DS9Utils' to  see the different functions and 'DS9Utils LoadDS9QuickLookPlugin' to load the analysis file in DS9! \nThen open DS9 and use the different functions.\x1b[0m")
        import os
        os.system('DS9Utils')
        os.system('DS9Utils LoadDS9QuickLookPlugin')

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        # print("\033[32mPackage installed. \nRun 'DS9Utils' to  see the different functions and 'DS9Utils LoadDS9QuickLookPlugin' to load the analysis file in DS9! \nThen open DS9 and use the different functions.\x1b[0m")
        import os
        os.system('DS9Utils')
        os.system('DS9Utils LoadDS9QuickLookPlugin')


def setup_package():
    setup(
    name='pyds9plugin',
    python_requires='>3.5.2',
    version=version,
    license='CeCILL-B',
    install_requires = requires,
    long_description= " A python DS9 extension for quicklook processing of astronomical images. This highly interactive extension can be generalized automatically to a set of images to turn the plug-in into a real multi-processing pipeline.",
    url='https://people.lam.fr/picouet.vincent/index.html',
    platforms=["Linux", "Mac OS-X", "Unix"],
    cmdclass={'install': PostInstallCommand,'develop': PostDevelopCommand},
    packages = find_packages(),
    package_data = data,
    include_package_data=True,
    entry_points = entry_points,
    author = 'Vincent Picouet',
    maintainer = 'Vincent Picouet',
    author_email = 'vincent.picouet@lam.fr',
    description  = 'DS9 Quick-Look plug-in')
    return

if __name__ == '__main__':
    setup_package()
    print("\033[32mPackage installed. \nRun 'DS9Utils' to  see the different functions and 'DS9Utils LoadDS9QuickLookPlugin' to load the analysis file in DS9! \nThen open DS9 and use the different functions.\x1b[0m")
