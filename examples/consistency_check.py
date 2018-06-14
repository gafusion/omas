# -*- coding: utf-8 -*-
"""
IMAS consistency check
======================
OMAS can enforce consistency with IMAS data structure.
There are three possible settings:

* **consistency_check = False**: no consistency check. `False` can be used to use OMAS for other purposes other than IMAS

* **consistency_check = 'warn'**: will print a wanrning if entry is outside of IMAS scope. `'warn'` can be useful when wanting to work with entries that are not yet supported by IMAS. Remember to open a JIRA issue (https://jira.iter.org) to start a conversation about making these entries officially part of IMAS.

* **consistency_check = True**: will raise an error if entry is outside of IMAS scope. Recommended when working with IMAS. The error raised will clearly say where the error is, and provide suggestions based on valid IMAS structure.
"""

from __future__ import print_function, division, unicode_literals

import os

from omas import *

print('*' * 20)
print('consistency_check = False')
print('*' * 20)
ods = ods_sample()
ods.consistency_check = False
ods['equilibrium.time_slice[2].does_not_exist.global_quantities.ip'] = 1
print(ods['equilibrium.time_slice[2].does_not_exist.global_quantities.ip'])

print('*' * 20)
print("consistency_check = 'warn'")
print('*' * 20)
ods = ods_sample()
ods.consistency_check = 'warn'
ods['equilibrium.time_slice[0].does_not_exist.global_quantities.ip'] = 1
print(ods['equilibrium.time_slice[0].does_not_exist.global_quantities.ip'])

print('*' * 20)
print("consistency_check = True")
print('*' * 20)
ods = ods_sample()
ods['equilibrium.time_slice[2].does_not_exist.global_quantities.ip']

###################################
# In the above example `consistency_check = True` result in the following error::
#
#     LookupError: `equilibrium.time_slice.:.does_not_exist` is not a valid IMAS 3.18.0 location
#                                            ^^^^^^^^^^^^^^
#     Did you mean: ['profiles_2d', 'ggd', 'boundary', 'profiles_1d', 'constraints', 'global_quantities', 'coordinate_system', 'boundary_separatrix', 'time', 'convergence']
