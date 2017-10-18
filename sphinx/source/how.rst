The hierarchical IMAS data model is represented in OMAS as a set of nested `omas` objects.

Techically speaking the Python `omas` class is a subclass of the standard Python dictionary class, and it inherits all of the functionalities, with the addition of:

1. on the fly checks for consistency with IMAS data model

2. graceful error handling with suggestions for navigating the hierarchical data structure

3. dynamic creation of the tree hierarchy as items they are accessed

4. support read/write/access using multiple formats
  * standard Python dictionary format: `ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']`
  * IMAS hierarchical tree format: `ods['equilibrium.time_slice[0].profiles_2d[0].psi']`
  * dot separated string format: `ods['equilibrium.time_slice.0.profiles_2d.0.psi']`
  * list of nodes format: `ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]`

5. save/load omas objects to/from:
  * Python pickle: files using native Python serialization tools
  * Json: files using Json ASCII format for representing hierarchical data
  * nc: files using binary NetCDF format
  * S3: the cloud using Amazon Simple Storage Service
  * IMAS: ITER data storage system

+---------------+------------------------+--------+
| OMAS format   | Storage type           | Remote |
+===============+========================+========+
| **omas**      | Python memory          |        |
+---------------+------------------------+--------+
| pickle        | Python binary file     |        |
+---------------+------------------------+--------+
| Json          | ASCII files            |        |
+---------------+------------------------+--------+
| NetCDF        | Binary files           |        |
+---------------+------------------------+--------+
| S3            | Binary files Database  | *      |
+---------------+------------------------+--------+
| IMAS          | IMAS Database          |        |
+---------------+------------------------+--------+