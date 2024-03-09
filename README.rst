==================
gfs-dynamical-core
==================

**gfs-dynamical-core** is home to the GFS dynamical core, which was previously a part of
CliMT_.

Installation
-------------

Python versions supported - 3.8, 3.9 and 3.10

gfs-dynamical-core can be installed directly from the python package index using pip. This should
also automatically install the correct version of climt.

    pip install gfs-dynamical-core

Working versions - 0.1.23 and 0.1.9

Specify the version if required

    pip install gfs-dynamical-core==0.1.23

This command should work on most systems and will install wheels for generic architecture. However,
this may result in a slightly slower code.

To optimise the package for your system architecture, build it from source. See the documentation_
for instructions.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _sympl: https://github.com/mcgibbon/sympl
.. _Pint: https://pint.readthedocs.io
.. _xarray: http://xarray.pydata.org
.. _documentation: https://gfs-dynamical-core.readthedocs.io
.. _CliMT: https://github.com/CliMT/climt