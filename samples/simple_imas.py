from __future__ import print_function, division, unicode_literals

import os
# set OMAS debugging topic
os.environ['OMAS_DEBUG_TOPIC'] = 'imas_code_dump'

from omas import *

# Instantiate new OMAS Data Structure (ODS)
ods = ODS()

# 0D data
ods['equilibrium']['time_slice'][0]['time'] = 1000.
ods['equilibrium']['time_slice'][0]['global_quantities.ip'] = 1.E6
# 1D data
ods['equilibrium']['time_slice'][0]['profiles_1d.psi'] = [1, 2, 3]
# 2D data
ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['b_field_tor'] = [[1, 2, 3],
                                                                        [4, 5, 6]]
if False:
    #this raises an error
    ods['equilibrium.time_slice.0.profiles_2d.0.grid_type']=1

# Save to file
save_omas(ods, 'test.omas')
# Load from file
ods1 = load_omas('test.omas')

# Save to IMAS
paths = save_omas_imas(ods, machine='ITER', shot=1, new=True)
# Load from IMAS
ods1 = load_omas_imas(machine='ITER', shot=1)#, paths=paths)

# check data
check = different_ods(ods, ods1)
if not check:
    print('OMAS data got saved and loaded correctly')
else:
    print(check)
