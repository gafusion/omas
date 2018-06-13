# -*- coding: utf-8 -*-
"""
Simple IMAS
===========
Simple script showcasing OMAS writing minimal amount of data to IMAS.
Also, this script shows the use of 'imas_code_dump' as an `OMAS_DEBUG_TOPIC`,
which can be useful for debugging purposes.
"""

from __future__ import print_function, division, unicode_literals

import os
from omas import *

# set OMAS debugging topic
# NOTE: appending '_dump' to a debug topic will write a omas.dump file in the working directory
#       with the debug output in it. The 'imas_code_dump' is particularly useful since it generates
#       the exact list of Python commands that OMAS used to work with IMAS.
os.environ['OMAS_DEBUG_TOPIC'] = 'imas_code_dump'

ods = ODS()

# 0D data
ods['equilibrium']['time_slice'][0]['time'] = 1000.
ods['equilibrium']['time_slice'][0]['global_quantities.ip'] = 1.E6
# 1D data
ods['equilibrium']['time_slice'][0]['profiles_1d.psi'] = [1, 2, 3]
# 2D data
ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['b_field_tor'] = [[1, 2, 3], [4, 5, 6]]

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
