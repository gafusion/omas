Installation
============
.. _install:

The OMAS library runs on **Python2** and **Python3**.

To install `OMAS from pip <https://pypi.python.org/pypi/omas/>`_ (for users):

.. code-block:: none

    sudo pip install omas

To install `OMAS from GitHub <https://github.com/gafusion/omas>`_ (for developers):

.. code-block:: none

    git clone git@github.com:gafusion/omas.git
    cd omas
    sudo pip install -e .[build_structures,build_documentation]       # add this `omas` directory to your $PYTHONPATH
                                                                      # [build_structures,build_documentation] options
                                                                      # install extra packages for development purposes

Different `Python packages are required <_static/requirements.txt>`_ depending on the :ref:`data storage systems <omas_formats>`

--------------------
Testing installation
--------------------

The OMAS installation can be tested by running the regression tests:

.. code-block:: none

    python -m unittest discover --pattern="*.py" -s omas/tests/ -v

