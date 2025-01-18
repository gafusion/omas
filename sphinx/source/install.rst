Installation
============

.. _install:

OMAS runs with *Python3.6+*

**PyPI**: To install `OMAS with pip <https://pypi.python.org/pypi/omas/>`_ (for users):

.. code-block:: none

    pip install --upgrade omas

where `upgrade` is used to update the omas installation to the latest version.

The development version of omas can also be installed with pip:

.. code-block:: none

    pip install --upgrade -e git+git@github.com:gafusion/omas#egg=omas

**Conda**: To install `OMAS with conda <https://anaconda.org/conda-forge/omas>`_ (for users):

.. code-block:: none

    conda install -c conda-forge omas
    conda update  -c conda-forge omas

**GitHub** To clone `OMAS from GitHub <https://github.com/gafusion/omas>`_ (for developers):

.. code-block:: none

    git clone git@github.com:gafusion/omas.git
    cd omas
    pip install --upgrade -e '.[build_structures,build_documentation]'   # Add this `omas` directory to your $PYTHONPATH
                                                                         # The [build_structures,build_documentation] options
                                                                         # install packages required for extra development purposes

List of `Python package requirements <_static/requirements.txt>`_.

Optional dependencies
---------------------

Some dependencies are not required for the core functionality of OMAS,
but are needed for some of the additional features.

Here is a list of optional dependencies:

.. Table of optional dependencies: mongo, s3, build_structures, build_documentation

+----------------------+----------------------+
| **Package**          | **Description**      |
+======================+======================+
| `mongo`              | Support for MongoDB  |
+----------------------+----------------------+
| `s3`                 | Support for AWS S3   |
+----------------------+----------------------+
| `build_structures`   | Support for building IMAS structures |
+----------------------+----------------------+
| `build_documentation`| Support for building Sphinx documentation |
+----------------------+----------------------+

Optional dependencies can be installed with the following command:

**PyPI**:

.. code-block:: none

    pip install --upgrade omas[optional,optional2,...]

**Conda**: All optional dependencies are installed by default. To suppress the installation of optional dependencies, use the following command:

.. code-block:: none

    conda install -c conda-forge omas --no-deps

**GitHub**:

.. code-block:: none

    pip install --upgrade -e '.[optional,optional2,...]'

**Note**: Skip the `--upgrade` option if you are installing the package for the first time.

Installation with IMAS
======================

Different IMAS versions require different Python installations, each of which may not have the Python packages that are needed to run OMAS.
One may ask the maintainers of the IMAS installation to
The simplest way to ensure that the `omas` dependencies are always available and up-to-date, is to setup all of the Python packages that `omas` depends on in a standalone folder:

.. code-block:: none

    cd path_to_omas_installation
    git clone git@github.com:gafusion/omas.git
    cd omas
    pip install --target ./site-packages -r requirements.txt

Then update the `omas` UNIX module to include the `omas` and the `omas/site-packages` folders to the $PYTHONPATH environmental variable:

.. code-block:: none

    prepend-path     PYTHONPATH path_to_omas_installation/omas
    prepend-path     PYTHONPATH path_to_omas_installation/omas/site-packages

Loading the `imas` module followed by the `omas` module should then give you a fully functional IMAS + OMAS environment:

.. code-block:: none

    module load imas # NOTE: the name of the IMAS module may change on different systems
    module load omas # NOTE: the name of the OMAS module may change on different systems

Testing installation
====================

The OMAS installation can be tested by running the regression tests:

.. code-block:: none

    cd omas
    make test
