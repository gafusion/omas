Concept
=======
.. _concept:

The hierarchical IMAS data model is represented in OMAS as a set of nested `ODS` objects (OMAS Data Structure).

The `ODS` class extends native Python dictionary and list classes with:

1. On-the-fly check for **compliance with IMAS data model**

   * enforce consistency with IMAS data structure

   * enforce correct data type

   * enforce correct number of data dimensions (1D, 2D, 3D, ...)

   * warn if obsolescent IMAS entries are used

2. **Graceful error handling** with suggestions for navigating the hierarchical data structure::

    LookupError: `equilibrium.time_slice.:.does_not_exist` is not a valid IMAS location
                                           ^^^^^^^^^^^^^^
    Did you mean: ['coordinate_system', 'profiles_1d', 'profiles_2d', 'ggd', 'time', 'convergence', 'boundary', 'global_quantities', 'constraints']

3. **Dynamic creation of the tree hierarchy** as items they are accessed with support for **different syntaxes**:

   .. code-block:: python

       ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']   # standard Python dictionary syntax
       ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]      # list of nodes syntax
       ods['equilibrium.time_slice[0].profiles_2d[0].psi']            # IMAS hierarchical tree syntax
       ods['equilibrium.time_slice.0.profiles_2d.0.psi']              # . separated string syntax

       NOTE: Python3.6+ f-strings can be very handy when looping over arrays of structures. For example:
       for time_index in range(len(ods[f'equilibrium.time_slice'])):
           for grid_index in range(len(ods[f'equilibrium.time_slice.{time_index}.profiles_2d'])):
               print(ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.psi'])

4. **Extract sigle time slice** from whole data structure:

   .. code-block:: python

       ods.slice_at_time(1000.)

5. **Simplified handling of array of structures**

   .. code-block:: python

       ods['wall.description_2d.3.limiter.type.name']       # use `#` to access entries as in a list
       ods['wall.description_2d.-1.limiter.type.name']      # use `-#` to access entries from the end of the list
       ods['equilibrium.time_slice.:.global_quantities.ip'] # use `:` to collect quantities across list of structures
       ods['wall.description_2d.+.limiter.type.name']       # use `+` to append entries to a list of structures

6. Automatic **handling of XML code.parameters** will show them as dictionaries and save them as XML into IMAS:

   .. code-block:: python

        ods['equilibrium.code.parameters'] = CodeParameters('./parameters.xml')

7. Automatic **COCOS transformations** [`read the COCOS cheatsheet <https://docs.google.com/document/d/1-efimTbI55SjxL_yE_GKSmV4GEvdzai7mAj5UYLLUXw/edit?usp=sharing>`_]:

   .. code-block:: python

       with omas_environment(ods, cocosio=2):
           ods['equilibrium.time_slice.0.profiles_1d.psi'] = gEQDSK['psi']

8. Automatic **coordinate interpolations**:

   .. code-block:: python

       with omas_environment(ods, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': gEQDSK['psi']}):
           ods['equilibrium.time_slice[0].profiles_1d.pressure'] = gEQDSK['pressure']

9. Automatic **units conversions** via `pint Python package <http://pint.readthedocs.io/en/latest/>`_:

   .. code-block:: python

       ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement'] = 8.0 * milliseconds
       ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement'] # will return 0.008

10. Unified interface for **querying about time** dimension:

    .. code-block:: python

        ods.time('equilibrium')                                     # will return an array of times
        ods.time('equilibrium.time_slice')                          # will return an array of times
        ods.time('equilibrium.time_slice.0.global_quantities.ip')   # will return a scalar time

11. Seamless handling of **uncertain quantities** via `uncertainties Python package <https://github.com/lebigot/uncertainties>`_:

    .. code-block:: python

        ods['equilibrium.time_slice.0.profiles_1d.q'] = uarray(nom_value, std_dev)

12. Evaluate **derived physics quantities** from more fundamental ones:

    .. code-block:: python

        ods.physics_core_profiles_pressures()

13. **Get data as multidimensional array structures** in `xarray <http://xarray.pydata.org/en/stable/>`_ format:

    .. code-block:: python

        ods['core_profiles.profiles_1d.0.electrons.densdensity_thermality'].xarray()

14. Conveniently **plot individual quantities**:

    .. code-block:: python

        ods.plot_quantity('core_profiles.profiles_1d.0.electrons.density_thermal')

15. **Use regular expressions** with ``@`` construct for accessing data and plotting:

    .. code-block:: python

        ods['@core.*0.elect.*dens.*th']
        ods.plot_quantity('@core.*0.elect.*dens.*th')

16. **Predefined set of plots** available:

    .. code-block:: python
    
        ods.plot_core_profiles_summary()

17. **On the fly data mapping** of experimental data `from native database <https://github.com/gafusion/omas/blob/master/omas/machine_mappings/d3d.json>`_:

    .. code-block:: python

        with ods1.open('machine', 'd3d', 168830, options={'EFIT_tree': 'EFIT01'}):
            plot(ods['equilibrium.time_slice.0.global_quantities.ip'])

18. Save/load ODSs to/from **different backend storage systems**:

.. _omas_formats:

+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| OMAS format   | Description                                                                          | Storage type                  | Remote storage |  Python Requirements  |
+===============+======================================================================================+===============================+================+=======================+
| **pickle**    | Files using native Python serialization tools                                        | Python binary file            |       no       |                       |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **Json**      | Using Json format for representing hierarchical data                                 | ASCII file                    |       no       |                       |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **NetCDF**    | Files using binary NetCDF format (flat data structure)                               | Binary file                   |       no       |        netCDF4        |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **HDF5**      | Files using binary HDF5 format (hierarchical data structure)                         | Binary file                   |       no       |          h5py         |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **S3**        | Cloud storage using Amazon Simple Storage Service                                    | Object Store                  |       yes      |         boto3         |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **MongoDB**   | Document-based distributed database                                                  | MongoDB Database              |       yes      |        pymongo        |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **HDC**       | Hierarchical Dynamic Containers                                                      | Memory                        |       no       |   pyhdc (optional)    |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **UDA**       | Universal Data Access                                                                | UDA Database                  |       yes      |   pyuda (optional)    |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **ASCII**     | Format used by the IMAS AL 4.0.x ASCII backend                                       | ASCII file                    |       no       |                       |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **MACHINE**   | Access tokamaks experimental data with imas syntax                                   | Native experimental databases |       yes      |    MDSplus            |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+
| **IMAS**      | `ITER data storage system <https://user.iter.org/?uid=YSQENW&action=get_document>`_  | IMAS Database                 |       no       |    imas (optional)    |
+---------------+--------------------------------------------------------------------------------------+-------------------------------+----------------+-----------------------+