Installation
============

The OMAS library runs on **Python2** and **Python3**.

To install `OMAS from pip <https://pypi.python.org/pypi/omas/>`_::

        sudo pip install omas

To install `OMAS from GitHub <https://github.com/gafusion/omas>`_::

        git clone git@github.com:gafusion/omas.git
        cd omas
        sudo pip install -e .        # add this `omas` directory to your $PYTHONPATH

Different `Python packages are required <_static/requirements.txt>`_ depending on the :ref:`data storage systems <omas_formats>`

--------------------
Testing installation
--------------------

The OMAS installation can be tested by running the regression tests:

.. code-block:: none

    cd omas
    python omas/tests/test_omas_plot.py
    python omas/tests/test_omas_suite.py

A summary of the storage systems

.. code-block:: none

    cd omas
    python omas/examples/save_load_all.py
