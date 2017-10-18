Installation
============

To get the OMAS source::

        git clone git@github.com:gafusion/omas.git

OMAS is written in pure Python, and (with limited functionality) can work out of the box without any additional Python dependencies.

Addidional Python dependencies are required depending on the data save formats used.

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