.. highlight:: shell

============
Installation
============

pip install
-----------

Python versions supported - 3.8, 3.9 and 3.10

You can install gfs_dynamical_core using the command. This should
also automatically install the correct version of climt.

.. code-block:: console

    $ pip install gfs_dynamical_core

Working versions - 0.1.23 and 0.1.9

Specify the version if required

.. code-block:: console

    $ pip install gfs-dynamical-core==0.1.23

On Ubuntu Linux, you might need to prefix the above command with :code:`sudo`. This command should
work on Linux and Mac. This package is currently not available on Windows.

.. NOTE::
    If you are not using Anaconda, please ensure you have the libpython library installed.

This is the simplest way to install the gfs_dynamical_core package and does not require a compiler
on your system. However, as pip installs wheels for a generic system architecture, installing via pip
will not build optimally for your particular architecture, resulting in a slower code. 

For this reason, it is recommended to build gfs_dynamical_core from source.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

Installing from source
----------------------

The sources for gfs_dynamical_core can be downloaded from the `Github repo`_.

To clone the public repository:

.. code-block:: console

    $ git clone https://github.com/Ai33L/gfs_dynamical_core.git

Once you have a copy of the source, you can install it from inside the 
gfs_dynamical_core directory with:

.. code-block:: console

    $ pip install -r requirements_dev.txt
    $ python setup.py develop

Both commands may require the use of :code:`sudo`.

Dependencies for source installations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gfs_dynamical_core depends on some easily installable libraries. For
an installation from source, it also requires that C and fortran
compilers be installed.

On Ubuntu Linux, for example, these dependencies can be
installed by:

.. code-block:: console

    $ sudo apt-get install gcc
    $ sudo apt-get install gfortran
    $ sudo apt-get install python-dev
    $ sudo pip install -U cython
    $ sudo pip install -U numpy

use :code:`pip3` and :code:`python3-dev` if you use Python 3.

On Mac OSX, it is recommended that you use `anaconda`_ as your python distribution.
This will eliminate the need to install cython, numpy and python-dev.
Once you have anaconda installed, you will need to do the following:

.. code-block:: console

    $ brew install gcc
    $ export CC=gcc-x
    $ export FC=gfortran-x

Where :code:`gcc-x,gfortran-x` are the names of the C,Fortran compilers that Homebrew installs.
Exporting the name of the compiler is essential on Mac since the
default compiler that ships with Mac (called :code:`gcc`, but is actually a
different compiler) cannot
compile OpenMP programs.

Common build issues
~~~~~~~~~~~~~~~~~~~

A frequent issue is OpenBLAS build failing. This happens when OpenBLAS fails to detect cpu architecture and/or if a particular cpu 
is not supported. If you face this issue, modify line 64 in gfs_dynamical_core/_lib/Makefile

.. code-block:: console

    $ if [ ! -d $(BLAS_DIR) ]; then mkdir $(BLAS_DIR); tar -xvzf $(BLAS_GZ) -C $(BLAS_DIR) --strip-components=1 > log 2>&1; fi; cd $(BLAS_DIR); CFLAGS=$(CLIMT_CFLAGS) make NO_SHARED=1 NO_LAPACK=0 > log 2>&1 ; make PREFIX=$(BASE_DIR) NO_SHARED=1 NO_LAPACK=0 install > log.install.blas 2>&1 ; cp ../lib/libopenblas.a $(LIB_DIR)

Modify this to

.. code-block:: console

    $ if [ ! -d $(BLAS_DIR) ]; then mkdir $(BLAS_DIR); tar -xvzf $(BLAS_GZ) -C $(BLAS_DIR) --strip-components=1 > log 2>&1; fi; cd $(BLAS_DIR); CFLAGS=$(CLIMT_CFLAGS) make TARGET='GENERIC' NO_SHARED=1 NO_LAPACK=0 > log 2>&1 ; make PREFIX=$(BASE_DIR) NO_SHARED=1 NO_LAPACK=0 install > log.install.blas 2>&1 ; cp ../lib/libopenblas.a $(LIB_DIR)

Speciying a GENERIC architecture will not optimise for the specific architecture, but should solve the build issue in most cases.

.. _Homebrew: https://brew.sh/
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github repo: https://github.com/Ai33L/gfs_dynamical_core
.. _tarball: https://github.com/CliMT/climt/tarball/master
.. _anaconda: https://www.continuum.io/downloads
