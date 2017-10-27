The hierarchical IMAS data model is represented in OMAS as a set of nested `omas` objects.

Techically speaking **the Python `omas` class is a subclass of the standard Python dictionary class**,
and it inherits all of the functionalities, with the addition of:

1. on-the-fly check for consistency with IMAS data model

2. graceful error handling with suggestions for navigating the hierarchical data structure

3. dynamic creation of the tree hierarchy as items they are accessed

4. support read/write/access using multiple formats::

    ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']  #standard Python dictionary format
    ods['equilibrium.time_slice[0].profiles_2d[0].psi']            #IMAS hierarchical tree format
    ods['equilibrium.time_slice.0.profiles_2d.0.psi']              #dot separated string format
    ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]     #list of nodes format

5. save/load omas objects to/from:

   +---------------+-------------------------------------------------------------+------------------------+--------+
   | OMAS format   | Description                                                 | Storage type           | Remote |
   +===============+=============================================================+========================+========+
   | **omas**      |                                                             | Python memory          |        |
   +---------------+-------------------------------------------------------------+------------------------+--------+
   | pickle        | files using native Python serialization tools               | Python binary file     |        |
   +---------------+-------------------------------------------------------------+------------------------+--------+
   | Json          | using Json ASCII format for representing hierarchical data  | ASCII files            |        |
   +---------------+-------------------------------------------------------------+------------------------+--------+
   | NetCDF        | files using binary NetCDF format                            | Binary files           |        |
   +---------------+-------------------------------------------------------------+------------------------+--------+
   | S3            | store data in the cloud using Amazon Simple Storage Service | Binary files Database  | *      |
   +---------------+-------------------------------------------------------------+------------------------+--------+
   | IMAS          | ITER data storage system                                    | IMAS Database          |        |
   +---------------+-------------------------------------------------------------+------------------------+--------+