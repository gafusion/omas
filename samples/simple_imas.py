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
