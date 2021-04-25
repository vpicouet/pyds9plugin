# [DS9 Quick look Plug-in](https://people.lam.fr/picouet.vincent/index.html)
=========

pyDS9plugin is a high-level python package for both data analysis and data processing. 


[![GitHub License](http://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[//]: <> ([![PyPI Version](https://img.shields.io/pypi/v/dataphile.svg)](https://pypi.org/project/dataphile/))
[//]: <> ([![Docs Latest](https://readthedocs.org/projects/dataphile/badge/?version=latest)](https://dataphile.readthedocs.io))

---

<!-- Animated GIF of AutoGUI -->
<img src="https://people.lam.fr/picouet.vincent/pyds9plugin/gif/3d2d_new.mov.gif" width="80%"
style="display:block;margin: 0 auto;">

**Figure**: Demonstration of of one of pyds9plugin features.

Installation
------------

To install pyds9plugin for general purposes use Pip:

```
pip install pyds9plugin
```

If you are using Anaconda, install using the above call to pip _from inside your environment_.
There is not as of yet a separate conda package.

Documentation
-------------

Some documentation is availanle available at [https://people.lam.fr/picouet.vincent/pyds9plugin](https://people.lam.fr/picouet.vincent/index.html).


Contributions
-------------

Contributions are welcome in the form of  suggestions for additional features,  pull requests with
new features or  bug fixes, etc. If you find  bugs or have questions, open an  _Issue_ here. If and
when the project grows, a  code of conduct will be provided along side  a more comprehensive set of
guidelines for contributing; until then, just be nice.

Features
--------


SAOImage DS9 is an astronomical imaging and data visualization application. Its 30 years of development has made it very stable and easy to use. Because of this it became an essential tool in all fields of astronomy (observation, simulation, instrumentation). Much more profitable but pretty unsung feature, its extensibility makes it a limitless tool to interact with astronomical (or not) data. Unfortunately it appears that this extensibility did not inspire or generate a large collaborative and well organized effort to develop important extensions that would finally, years after years, converge towards a stable/rapid/configurable multi-extension package of DS9 which would make it as essential as Photoshop for photographers.

Because I am convince of its interest, this extension is a very naive attempt to try initiate this tendency. Besides joining the very user friendly but general DS9 interface with a series of more specific function this extension tries to gather a glimpse of all the possibilities that offers DS9 extensibility:  

-   Fill some gaps of DS9 with stable  [general functions](https://people.lam.fr/picouet.vincent/General.html): Radial/Energy profiles, stacking, fitting, region based functions
-   Link all essential  [astronomical image processing softwares](https://people.lam.fr/picouet.vincent/softwares.html)  (SExtractor, STIFF, SWARP, PSFex) and offer them a parameter GUI
-   Create  [images data base](https://people.lam.fr/picouet.vincent/General.html)  based on header information that can then be used to create data subset (symbolic links) verifying specific condditions
-   Link more complex but general  [processing functions](https://people.lam.fr/picouet.vincent/processing.html): Autocorrelation, FFT, smoothing, maskink/interpolation, noise measurements
-   Gain some time with  [key functions](https://people.lam.fr/picouet.vincent/General.html): Automatically change display/lock parameters
-   Create a  [all-in-one astronomical functions package](https://people.lam.fr/picouet.vincent/astronomy.html): converter, cosmological calculator, etc.
  
More important, it offers a base to [add your own functions](https://people.lam.fr/picouet.vincent/output.html) to this architecture. Follow this [procedure](https://people.lam.fr/picouet.vincent/Install.html) to install it. If, after some time playing with it, you believe in the potential of beginning a collaborative effort to develop the tool described above please [email me](mailto:vincent.picouet@lam.fr) and [join the project on gitlab](https://gitlab.lam.fr/vpicouet/DS9functions/tree/master)!
