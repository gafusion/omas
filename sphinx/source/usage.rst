The following code snipplet shows the usage of the Python `omas` class:

.. code-block:: python

    from omas import *
    import numpy

    # instantiate new omas data structure (ODS)
    ods=omas()

    # add some 0D data
    ods['equilibrium']['time_slice'][0]['time']=1000.
    ods['equilibrium']['time_slice'][0]['global_quantities.ip']=1.E6
    # add some 1D data
    ods['equilibrium']['time_slice'][0]['profiles_1d.psi']=[1,2,3]
    # add some 2D data
    ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['b_field_tor']=[[1,2,3],[4,5,6]]

    # Save to NetCDF file
    save_omas(ods,'test.nc')
    # Load from NetCDF file
    ods1=load_omas('test.nc')
    # Check data consistency
    if not different_ods(ods, ods1):
       print('OMAS data was saved to and loaded from file correctly')

    # Save to IMAS
    paths=save_omas_imas(ods, user='meneghini', tokamak='D3D', version='3.10.1', shot=133221, run=0, new=True)
    # Load from IMAS
    ods1=load_omas_imas(user='meneghini', tokamak='D3D', version='3.10.1', shot=133221, run=0, paths=paths)
    # Check data consistency
    if not different_ods(ods, ods1):
       print('OMAS data was saved to and loaded from IMAS correctly')