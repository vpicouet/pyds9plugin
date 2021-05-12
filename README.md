# [DS9 Quick look Plug-in](https://people.lam.fr/picouet.vincent/index.html)

pyDS9plugin is the first open source pythonic [SAOImageDS9](https://sites.google.com/cfa.harvard.edu/saoimageds9) quick look plugin.

The documentation of the extension is available [here](https://nuage.osupytheas.fr/s/Z6D64wRpDkE7wTJ).

SAOImage DS9 is an astronomical imaging and data visualization application. Its 30 years of development has made it very stable and easy to use. This made it an essential tool in all fields of astronomy (observation, simulation, instrumentation). Much more profitable but pretty unsung feature, its extensibility makes it a limitless tool to interact with astronomical data. This extensibility did not generate a large collaborative and well organized effort to develop important extensions that could progressively converge towards a stable/rapid/configurable multi-extension DS9 package.

Because I am convince of its interest, this extension is a very naive attempt to try initiate this tendency and explore the different possibilities.

The goal is the pyds9plugin is then three-fold:
-   Boosting the way we interact with scientific images in a quantitative way to earn important time
-   Try to bring the visualization software DS9 a step further by combining it to image processing tools
-   Create a code collaboration catalyst by providing a first extensive package gathering a glimpse of all the possibilities that offers DS9 extensibility




---

<!-- Animated GIF of AutoGUI -->



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

-   Command line access: The package is totally accessible via command line. Run `DS9Utils` to see all the available function and `DS9Utils function -h` to see the help of each function. All the arguments of the functions are parsed through argparse module which makes the functions not only accessible from DS9, but from terminal and python using argv argument.
-   Multi-processing: PyDS9plugin is by essence a quicklook plug-in that is perfect to analyze and process images on the fly by changing parameters and so on. But it was important for us to make it suitable for more important pipelines as soon as you are ok with the parameters to use. To this end, most of the functions are compatible with multi-processing so that they can be run on a set of images.
-   Multi-operability: Command line access and python import allows to operate the plugin for other pipelines. The plugin for DS9 could actually pretty easily be operated by other visualization softwares like ginga or glueviz.

-   Python interpreter/macros:


<img src="https://people.lam.fr/picouet.vincent/pyds9plugin/gif/python.mov.gif" width="80%"
style="display:block;margin: 0 auto;">
-   VTK 3D rendering:


<img src="https://people.lam.fr/picouet.vincent/pyds9plugin/gif/3d2d_new.mov.gif" width="80%"
style="display:block;margin: 0 auto;">

-   Interactive profile fitting :


<img src="https://people.lam.fr/picouet.vincent/pyds9plugin/gif/fitinter.mov.gif" width="80%"
style="display:block;margin: 0 auto;">
