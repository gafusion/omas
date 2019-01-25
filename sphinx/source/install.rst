Installation
============
.. _install:

The OMAS library runs on **Python2** and **Python3**.

To install `OMAS from pip <https://pypi.python.org/pypi/omas/>`_ (for users):

.. code-block:: none

    sudo pip install --upgrade omas

where ``--upgrade`` is used to update the omas installation to the latest version.

To install `OMAS from GitHub <https://github.com/gafusion/omas>`_ (for developers):

.. code-block:: none

    git clone git@github.com:gafusion/omas.git
    cd omas
    sudo pip install --upgrade -e .[build_structures, build_documentation]      # Add this `omas` directory to your $PYTHONPATH
                                                                                # The [build_structures,build_documentation] options
                                                                                # install packages required for extra development purposes

Different `Python packages are required <_static/requirements.txt>`_ depending on the :ref:`data storage systems <omas_formats>`

--------------------
Testing installation
--------------------

The OMAS installation can be tested by running the regression tests:

.. code-block:: none

    cd omas
    make tests2  # run tests witht the `python2` executable
    make tests3  # run tests witht the `python3` executable
