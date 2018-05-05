Concept
=======

The hierarchical IMAS data model is represented in OMAS as a set of nested `omas` objects.

The Python `omas` class behaves like standard Python dictionary and list classes
and it inherits all of their functionalities, with the addition of:

1. on-the-fly check for consistency with IMAS data model

2. graceful error handling with suggestions for navigating the hierarchical data structure::

    LookupError: `equilibrium.time_slice.:.does_not_exist` is not a valid IMAS 3.17.2 location
                                           ^^^^^^^^^^^^^^
    Did you mean: ['coordinate_system', 'profiles_1d', 'profiles_2d', 'ggd', 'time', 'convergence', 'boundary', 'global_quantities', 'constraints']

3. dynamic creation of the tree hierarchy as items they are accessed with support for different syntaxes::

    ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']   # standard Python dictionary format
    ods['equilibrium.time_slice[0].profiles_2d[0].psi']            # IMAS hierarchical tree format
    ods['equilibrium.time_slice.0.profiles_2d.0.psi']              # dot separated string format
    ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]      # list of nodes format

4. data slicing with `:` format (not only in time)::

    ods['equilibrium.time_slice[:].global_quantities.ip']

5. automatic `COCOS <https://linkinghub.elsevier.com/retrieve/pii/S0010465512002962>`_ transform support::

    ods=ODS(cocosin=2, cocosout=11)
    ods['equilibrium.time_slice.0..profiles_1d.psi'] = linspace(0,1,10)

6. seamless support for `uncertain <https://github.com/lebigot/uncertainties>`_ quantities::

    ods['equilibrium.time_slice[0].profiles_1d.q'] = uarray(nom_value,std_dev)

7. save/load omas objects to/from:

   .. _omas_formats:

   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+
   | OMAS format   | Description                                                 | Storage type           | Remote storage |  Python Requirements  |
   +===============+=============================================================+========================+================+=======================+
   | **pickle**    | Files using native Python serialization tools               | Python binary file     |       no       |                       |
   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+
   | **Json**      | Using Json format for representing hierarchical data        | ASCII files            |       no       |                       |
   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+
   | **NetCDF**    | Files using binary NetCDF format                            | Binary files           |       no       |        netCDF4        |
   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+
   | **S3**        | Cloud storage using Amazon Simple Storage Service           | Object Store           |       yes      |         boto3         |
   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+
   | **HDC**       | Hierarchical Dynamic Containers                             | Memory                 |       no       |         pyhdc         |
   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+
   | **ITM**  (*)  | ITM data storage system                                     | ITM Database           |       no       |         itm           |
   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+
   | **IMAS**      | ITER data storage system                                    | IMAS Database          |       no       |         imas          |
   +---------------+-------------------------------------------------------------+------------------------+----------------+-----------------------+

(\*) NOTE: In addition to the IMAS data model, OMAS can support any other hierarchical data representation where the data is stored in the leafs of the data structure. For example, ITM is a hierarchical data organization that is used by the `European Integrated Modeling Tokamak <http://iopscience.iop.org/article/10.1088/0029-5515/54/4/043018/meta>`_ effort and shares many similarities with IMAS. Writing data to ITM is supported by OMAS.
