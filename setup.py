#from distutils.core import setup
from setuptools import setup, find_packages


requires = ['numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib',
            'astropy >=1.0',
            'pyds9',
	        'photutils'
            ]

entry_points = {}
entry_points['console_scripts'] = ['DS9Utils = DS9FireBall.DS9Utils:main']

data = { "DS9FireBall": ["FireBall.ds9.ans", "Slits/*","Targets/*","Mappings/*"]}





setup(
    name='DS9FireBall',
    version='0.4dev',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires = requires,
    packages = find_packages(),
    package_data = data,
    include_package_data=True,
    entry_points = entry_points,
    author_email = 'vincent.picouet@lam.fr',
    description  = 'LAM AIT/quicklook functions proposal for FIREBall-2',
)





