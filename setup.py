#from distutils.core import setup
from setuptools import setup, find_packages
import sys

requires1 = ['numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib',
            'astropy >=1.3',
            'pyds9',
            'photutils',
            'PyQt5',
            'tqdm'
            ]


requires2 = ['numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib',
            'astropy >=1.3',
            'pyds9',
            'photutils',
            'tqdm'
            ]

entry_points = {}
entry_points['console_scripts'] = ['DS9Utils = DS9FireBall.DS9Utils:main']

data = { "DS9FireBall": ["FireBall.ds9.ans", "Slits/*","Targets/*","Mappings/*"]}


version = '1.52dev'

if sys.version_info.major == 3:
    setup(
        name='DS9FireBall',
        version=version,
        license='Creative Commons Attribution-Noncommercial-Share Alike license',
        install_requires = requires1,
        packages = find_packages(),
        package_data = data,
        include_package_data=True,
        entry_points = entry_points,
        author_email = 'vincent.picouet@lam.fr',
        description  = 'LAM AIT/quicklook functions proposal for FIREBall-2')
elif sys.version_info.major == 2:
    setup(
        name='DS9FireBall',
        version=version,
        license='Creative Commons Attribution-Noncommercial-Share Alike license',
        install_requires = requires2,
        packages = find_packages(),
        package_data = data,
        include_package_data=True,
        entry_points = entry_points,
        author_email = 'vincent.picouet@lam.fr',
        description  = 'LAM AIT/quicklook functions proposal for FIREBall-2')    




