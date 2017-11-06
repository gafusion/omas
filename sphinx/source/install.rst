Installation
============

The OMAS library runs on **Python2** and **Python3**

To install `OMAS from pip <https://pypi.python.org/pypi/omas/>`_::

        sudo pip install omas

To install `OMAS from GitHub <http://gafusion.github.io/omas/>`_::

        git clone git@github.com:gafusion/omas.git
        cd omas
        python samples/build_json_structures.py
        sudo pip install -e ".[build_structures]"

-------
Testing
-------

The OMAS installation can be tested by:

.. code-block:: python

    from omas import *
    test_omas_suite()

-------------------------
Installation requirements
-------------------------

Some Python packages are required :ref:`depending on the data formats used <omas_formats>`
