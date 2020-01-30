from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


requires = ['numpy >=1.8', 
            'scipy >=0.14', 
            'matplotlib>=3.1.1',
            'astropy >=1.3',
            'pyds9',
            'photutils',
            'PyQt5',
            'tqdm',
            'datetime',        
            'Pillow', 
            'pyswarm',
            'logalpha']

entry_points = {}
entry_points['console_scripts'] = ['DS9Utils = pyds9plugin.DS9Utils:main']

data = { "pyds9plugin": ["QuickLookPlugIn.ds9.ans","dygraph-combined_new.js", "Slits/*","Targets/*","Mappings/*","Regions/*","CSVs/*","config/*","Sextractor/*","doc/*"]}


MAJOR = '2'
MINOR = '9'
MICRO = '0dev5'
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
