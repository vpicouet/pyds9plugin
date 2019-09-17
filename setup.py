#from distutils.core import setup
from setuptools import setup, find_packages
import sys
#from pip.req import parse_requirements

requires1 = [#'https://github.com/ericmandel/pyds9.git#egg=pyds9',
	    'numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib==2.2.4',
            'astropy >=1.3',
            #'pyds9',
            'photutils',
            'PyQt5',
            'logalpha',
            'tqdm'
            ]


requires2 = [#'https://github.com/ericmandel/pyds9.git#egg=pyds9',
            'numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib==2.2.4',
            'astropy >=1.3',
            #'pyds9',
            'photutils',
            'logalpha',
            'tqdm'
            ]

entry_points = {}
entry_points['console_scripts'] = ['DS9Utils = DS9FireBall.DS9Utils:main']

data = { "DS9FireBall": ["FireBall.ds9.ans","dygraph-combined_new.js", "Slits/*","Targets/*","Mappings/*","Regions/*","CSVs/*","config/*","Sextractor/*"]}


version = '2.76dev'

if sys.version_info.major == 3:
    setup(
        name='DS9FireBall',
        version=version,
        license='Creative Commons Attribution-Noncommercial-Share Alike license',
        install_requires = requires1,
#	install_reqs = parse_requirements('requirements.txt', session='hack'),
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
#        install_reqs = parse_requirements('requirements.txt', session='hack'),
        packages = find_packages(),
        package_data = data,
        include_package_data=True,
        entry_points = entry_points,
        author_email = 'vincent.picouet@lam.fr',
        description  = 'LAM AIT/quicklook functions proposal for FIREBall-2')    




