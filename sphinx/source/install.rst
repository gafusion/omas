Installation
============

.. _install:

OMAS runs both with *Python2* and *Python3*.

**Pypi**: To install `OMAS with pip <https://pypi.python.org/pypi/omas/>`_ (for users):

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
    sudo pip install --upgrade -e .[build_structures, build_documentation]      # Add this `omas` directory to your $PYTHONPATH
                                                                                # The [build_structures,build_documentation] options
                                                                                # install packages required for extra development purposes

List of `Python 2 <_static/requirements_python2.txt>`_ or `Python 3 <_static/requirements_python3.txt>`_ package requirements.

Testing installation
====================

The OMAS installation can be tested by running the regression tests:

.. code-block:: none

    cd omas
    make tests2  # run tests witht the `python2` executable
    make tests3  # run tests witht the `python3` executable
