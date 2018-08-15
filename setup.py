from distutils.core import setup
from setuptools import find_packages

requires = ['numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib',
            'astropy >=1.0',
            'pyds9',
	        'photutils'
            ]

setup(
    name='DS9functions',
    version='0.1dev',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires = requires,
    packages = find_packages(),
    author_email = 'vincent.picouet@lam.fr',
    description  = 'LAM AIT/quicklook functions proposal for FIREBall-2',
)
