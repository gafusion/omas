Concept
=======



The hierarchical IMAS data model is represented in OMAS as a set of nested `ODS` objects (OMAS Data Structure).
The `ODS` class extends native Python dictionary and list classes with:

1. On-the-fly check for **consistency with IMAS data model**

2. **Graceful error handling** with suggestions for navigating the hierarchical data structure::

    LookupError: `equilibrium.time_slice.:.does_not_exist` is not a valid IMAS location
                                           ^^^^^^^^^^^^^^
    Did you mean: ['coordinate_system', 'profiles_1d', 'profiles_2d', 'ggd', 'time', 'convergence', 'boundary', 'global_quantities', 'constraints']

3. **Dynamic creation of the tree hierarchy** as items they are accessed with support for **different syntaxes**::

    ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']   # standard Python dictionary format
    ods['equilibrium.time_slice[0].profiles_2d[0].psi']            # IMAS hierarchical tree format
    ods['equilibrium.time_slice.0.profiles_2d.0.psi']              # dot separated string format
    ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]      # list of nodes format

4. **Extract sigle time slice** from whole data structure::

    ods.slice_at_time(1000.)

5. **Collect data across array of structures** (not only in time) with `:` construct::

    ods['equilibrium.time_slice.:.global_quantities.ip']

6. Automatic `COCOS <https://docs.google.com/document/d/1-efimTbI55SjxL_yE_GKSmV4GEvdzai7mAj5UYLLUXw/edit?usp=sharing>`_ **transformations**::

    with cocos_environment(ods, cocosin=2):
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = gEQDSK['psi']

7. Unified interface for **querying about time** dimension::

    ods.time('equilibrium')
    ods.time('equilibrium.time_slice')
    ods.time('equilibrium.time_slice.0.global_quantities.ip')

8. Seamless handling of `uncertain <https://github.com/lebigot/uncertainties>`_ **quantities**::

    ods['equilibrium.time_slice.0.profiles_1d.q'] = uarray(nom_value,std_dev)

9. Evaluate **derived quantities**::

    ods.physics_core_profiles_pressures()

10. **Predefined plot**::

    ods.plot_core_profiles_summary()

11. Save/load ODSs to/from **different storage systems**:

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
