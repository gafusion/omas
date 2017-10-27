Installation
============

The OMAS library runs on Python2 and Python3

To install OMAS::

        sudo pip install omas

To get the OMAS source::

        git clone git@github.com:gafusion/omas.git
        cd omas
        sudo pip install -e .[build_structures]

-------------------------
Installation requirements
-------------------------

Some Python dependencies are required depending on the data save formats used:

+---------------+-----------------------+
| OMAS format   |  Python Requirements  |
+===============+=======================+
| **omas**      |                       |
+---------------+-----------------------+
| pickle        |                       |
+---------------+-----------------------+
| Json          |                       |
+---------------+-----------------------+
| NetCDF        |           netCDF4     |
+---------------+-----------------------+
| S3            |           boto        |
+---------------+-----------------------+
| IMAS          |           imas        |
+---------------+-----------------------+