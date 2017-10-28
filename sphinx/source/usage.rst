
Basic usage
===========

.. code-block:: python

    from omas import *
    
    # Instantiate new OMAS Data Structure (ODS)
    ods=omas()
    
    # 0D data
    ods['equilibrium']['time_slice'][0]['time']=1000.
    ods['equilibrium']['time_slice'][0]['global_quantities.ip']=1.E6
    # 1D data
    ods['equilibrium']['time_slice'][0]['profiles_1d.psi']=[1,2,3]
    # 2D data
    ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['b_field_tor']=[[1,2,3],[4,5,6]]
    
    # Save to file
    save_omas(ods,'test.omas')
    # Load from file
    ods1=load_omas('test.omas')
    
    # Save to IMAS
    paths=save_omas_imas(ods, user='meneghini', tokamak='D3D',
                         version='3.10.1', shot=133221, run=0, new=True)
    # Load from IMAS
    ods1=load_omas_imas(user='meneghini', tokamak='D3D',
                        version='3.10.1', shot=133221, run=0, paths=paths)


Save and load OMAS data in different formats
============================================

.. code-block:: python

    from omas import *
    
    # load some sample data
    ods_start = ods_sample()
    
    # save/load Python pickle
    filename = 'test.pkl'
    save_omas_pkl(ods_start, filename)
    ods = load_omas_pkl(filename)
    
    # save/load ASCII Json
    filename = 'test.json'
    save_omas_json(ods, filename)
    ods = load_omas_json(filename)
    
    # save/load NetCDF
    filename = 'test.nc'
    save_omas_nc(ods, filename)
    ods = load_omas_nc(filename)
    
    # remote save/load S3
    filename = 'test.s3'
    save_omas_s3(ods, filename)
    ods = load_omas_s3(filename)
    
    # save/load IMAS
    user = os.environ['USER']
    tokamak = 'D3D'
    version = os.environ.get('IMAS_VERSION','3.10.1')
    shot = 1
    run = 0
    new = True
    paths = save_omas_imas(ods,  user, tokamak, version, shot, run, new)
    ods_end = load_omas_imas(user, tokamak, version, shot, run, paths)
    
    # check data
    if not different_ods(ods_start, ods_end):
       print('OMAS data got saved to and loaded correctly throughout')


Usage with OMFIT classes
========================
Some classes of the `OMFIT framework <http://gafusion.github.io/OMFIT-source/>`_ support OMAS

.. code-block:: python

    from omfit.classes.omfit_eqdsk import OMFITeqdsk
    from omas import *
    
    #read gEQDSK file in OMFIT
    eq=OMFITgeqdsk('133221.01000')
    
    #convert gEQDSK to OMAS data structure
    ods=eq.to_omas()
    
    # save OMAS data structure to IMAS
    user = os.environ['USER']
    tokamak = 'D3D'
    version = os.environ.get('IMAS_VERSION','3.10.1')
    shot = 1
    run = 0
    new = True
    paths = save_omas_imas(ods, user, tokamak, version,
                           shot, run, new)
    
    # load IMAS to OMAS data structure
    ods1 = load_omas_imas(user, tokamak, version, shot,
                          run, paths)
    
    #read from OMAS data structure
    eq1=OMFITgeqdsk('133221.02000').from_omas(ods1)
    
    #save gEQDSK file
    eq1.deploy()


