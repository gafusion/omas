Installation
============

The OMAS library runs on **Python2** and **Python3**

To install `OMAS from pip <https://pypi.python.org/pypi/omas/>`_::

        sudo pip install omas

To install `OMAS from GitHub <https://github.com/gafusion/omas>`_::

        git clone git@github.com:gafusion/omas.git
        cd omas
        python samples/build_json_structures.py
        sudo pip install -e ".[build_structures]"

-------------------------
Installation requirements
-------------------------

Some Python packages are required :ref:`depending on the data formats used <omas_formats>`

--------------------
Testing installation
--------------------

The OMAS installation can be tested by running:

.. code-block:: python

    from omas import *
    test_omas_suite()

This test saves and load a sample piece of data through the different OMAS save formats
and checks that information does not get lost or corrupted. If successful,
the test should output::

    ====================
    FROM  pkl  TO  pkl  : OK
    FROM  pkl  TO  json : OK
    FROM  pkl  TO   nc  : OK
    FROM  pkl  TO   s3  : OK
    FROM  pkl  TO  imas : OK
    FROM  json TO  pkl  : OK
    FROM  json TO  json : OK
    FROM  json TO   nc  : OK
    FROM  json TO   s3  : OK
    FROM  json TO  imas : OK
    FROM   nc  TO  pkl  : OK
    FROM   nc  TO  json : OK
    FROM   nc  TO   nc  : OK
    FROM   nc  TO   s3  : OK
    FROM   nc  TO  imas : OK
    FROM   s3  TO  pkl  : OK
    FROM   s3  TO  json : OK
    FROM   s3  TO   nc  : OK
    FROM   s3  TO   s3  : OK
    FROM   s3  TO  imas : OK
    FROM  imas TO  pkl  : OK
    FROM  imas TO  json : OK
    FROM  imas TO   nc  : OK
    FROM  imas TO   s3  : OK
    FROM  imas TO  imas : OK
    ====================
    [[1 1 1 1 1]
     [1 1 1 1 1]
     [1 1 1 1 1]
     [1 1 1 1 1]
     [1 1 1 1 1]]
    ====================

If for example `imas` is not installed on the system, the test output will look like::

    ====================
    FROM  pkl  TO  pkl  : OK
    FROM  pkl  TO  json : OK
    FROM  pkl  TO   nc  : OK
    FROM  pkl  TO   s3  : OK
    FROM  pkl  TO  imas : NO --> ImportError('No module named imas',)
    FROM  json TO  pkl  : OK
    FROM  json TO  json : OK
    FROM  json TO   nc  : OK
    FROM  json TO   s3  : OK
    FROM  json TO  imas : NO --> ImportError('No module named imas',)
    FROM   nc  TO  pkl  : OK
    FROM   nc  TO  json : OK
    FROM   nc  TO   nc  : OK
    FROM   nc  TO   s3  : OK
    FROM   nc  TO  imas : NO --> ImportError('No module named imas',)
    FROM   s3  TO  pkl  : OK
    FROM   s3  TO  json : OK
    FROM   s3  TO   nc  : OK
    FROM   s3  TO   s3  : OK
    FROM   s3  TO  imas : NO --> ImportError('No module named imas',)
    FROM  imas TO  pkl  : NO --> ImportError('No module named imas',)
    FROM  imas TO  json : NO --> ImportError('No module named imas',)
    FROM  imas TO   nc  : NO --> ImportError('No module named imas',)
    FROM  imas TO   s3  : NO --> ImportError('No module named imas',)
    FROM  imas TO  imas : NO --> ImportError('No module named imas',)
    ====================
    [[1 1 1 1 0]
     [1 1 1 1 0]
     [1 1 1 1 0]
     [1 1 1 1 0]
     [0 0 0 0 0]]
    ====================