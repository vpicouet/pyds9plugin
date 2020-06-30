#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


requires = ['numpy >=1.8', #'PyQt5','Pillow',
            'scipy >=0.14', 
            'matplotlib>=3.1.1',
            'astropy >=1.3',
            'pyds9',
            'photutils',
            'tqdm',
            'pyvista', 
            'datetime',  
            'pandas',       
            'dataphile']

entry_points = {}
entry_points['console_scripts'] = ['DS9Utils = pyds9plugin.DS9Utils:main']

data = { "pyds9plugin": ["QuickLookPlugIn.ds9.ans","config/*","Images/stack????????.fits","Images/stack.fits","Sextractor/*","dataphile/*","doc/ref/*/*","doc/ref/*"]}#,"doc/features_files/*","doc/img/*","doc/index_files/*"


MAJOR = '2'
MINOR = '9'
MICRO = '3dev9'
version = '%s.%s%s' % (MAJOR, MINOR, MICRO)

def setup_package():
    setup(
    name='pyds9plugin',
    python_requires='>3.5.2',
    version=version,
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires = requires,
    long_description=long_description,
    url='https://people.lam.fr/picouet.vincent/index.html', 
    platforms=["Linux", "Mac OS-X", "Unix"],
    packages = find_packages(),
    package_data = data,
    include_package_data=True,
    entry_points = entry_points,
    author_email = 'vincent.picouet@lam.fr',
    description  = 'DS9 Quick-Look plug-in')
    return

if __name__ == '__main__':
    setup_package()
