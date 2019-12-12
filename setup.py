#from distutils.core import setup
from setuptools import setup, find_packages
#import sys
#from pip.req import parse_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()



requires1 = [#'https://github.com/ericmandel/pyds9.git#egg=pyds9',
	        'numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib>=3.1.1',
            'astropy >=1.3',
            'pyds9',
            'photutils',
            'PyQt5',#'mpl_toolkits',#tkinter
            'tqdm',#'shutil','os','glob','sys','time'
            'datetime',#pkgp_resources'#,'urllib','subprocess','collections','multiprocessing','errno',            
            'Pillow', 
            'pyswarm'
            #'importlib','pkgutil'
            ]

#
#requires2 = [#'https://github.com/ericmandel/pyds9.git#egg=pyds9',
#            'numpy >=1.8', 
#            'scipy >=0.14', 
#            'matplotlib==2.2.4',
#            'astropy >=1.3',
#            #'pyds9',
#            'photutils',
#            'logalpha',
#            'tqdm'
#            ]

entry_points = {}
entry_points['console_scripts'] = ['DS9Utils = DS9FireBall.DS9Utils:main']

data = { "DS9FireBall": ["QuickLookPlugIn.ds9.ans","dygraph-combined_new.js", "Slits/*","Targets/*","Mappings/*","Regions/*","CSVs/*","config/*","Sextractor/*"]}


#version = '2.87dev'


MAJOR = '2'
MINOR = '9'
MICRO = '0dev0'
version = '%s.%s%s' % (MAJOR, MINOR, MICRO)

def setup_package():
    setup(
    name='DS9FireBall',
    python_requires='>3.5.2',
    version=version,
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires = requires1,#	install_reqs = parse_requirements('requirements.txt', session='hack'),
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
