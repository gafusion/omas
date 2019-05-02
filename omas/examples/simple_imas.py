#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple IMAS
===========
Simple script showcasing OMAS writing minimal amount of data to IMAS.
Also, this script shows the use of 'imas_code_dump' as an `OMAS_DEBUG_TOPIC`,
which can be useful for debugging purposes.

Prior running this script, the following commands must be typed at the teriminal
> import IMAS OMAS
> imasdb ITER
"""

from __future__ import print_function, division  # , unicode_literals

import os
from pprint import pprint
from omas import *

# enable fake IMAS support in case IMAS is not present on current system
# omas_rcparams['allow_fake_imas_fallback'] = True

# set OMAS debugging topic
# NOTE: appending '_dump' to a debug topic will write a omas.dump file in the working directory
#       with the debug output in it. The 'imas_code_dump' is particularly useful since it generates
#       the exact list of Python commands that OMAS used to work with IMAS.
os.environ['OMAS_DEBUG_TOPIC'] = 'imas_code_dump'

ods = ODS()

# first time-slice
# 0D data
ods['equilibrium']['time_slice'][0]['time'] = 1000.
ods['equilibrium']['time_slice'][0]['global_quantities.ip'] = 1.E6
# 1D data
ods['equilibrium']['time_slice'][0]['profiles_1d.psi'] = [1, 2, 3]

# second time-slice
# 0D data
ods['equilibrium']['time_slice'][1]['time'] = 1200.
ods['equilibrium']['time_slice'][1]['global_quantities.ip'] = 1.E6
# 1D data
ods['equilibrium']['time_slice'][1]['profiles_1d.psi'] = [1, 2, 3]
# 2D data
ods['equilibrium']['time_slice'][1]['profiles_2d'][0]['b_field_tor'] = [[1, 2, 3], [4, 5, 6]]

# different ODS
# 0D data
ods['core_profiles']['time'] = [1000.]

# Save to IMAS
print('=' * 20)
print(' Writing data to IMAS')
print('=' * 20)
paths = save_omas_imas(ods, machine='ITER', pulse=1, new=True)
pprint(ods.pretty_paths())

# Load from IMAS
print('=' * 20)
print(' Reading data from IMAS')
print('=' * 20)

# explicitly specify paths to collect
ods1 = load_omas_imas(machine='ITER', pulse=1, paths=paths)
pprint(ods1.pretty_paths())

# check data
print('=' * 20)
print(' Compared saved/loaded data')
print('=' * 20)
check = different_ods(ods, ods1)
if not check:
    print('OMAS data got saved and loaded correctly')
else:
    pprint(check)

# automatic paths discovery
# ods1 = load_omas_imas(machine='ITER', pulse=1)
# pprint(ods1.pretty_paths())

# subpath selection
# ods1 = load_omas_imas(machine='ITER', pulse=1, paths=['equilibrium.time_slice.0.time', 'equilibrium.time_slice.:.global_quantities.ip'])
# pprint(ods1.pretty_paths())
