from __future__ import print_function, division, unicode_literals

import os

from omas import *

print('*'*20)
print('consistency_check=False')
print('*'*20)
ods=ods_sample()
ods.consistency_check=False
ods['equilibrium.time_slice[2].does_not_exist.global_quantities.ip']=1
print(ods['equilibrium.time_slice[2].does_not_exist.global_quantities.ip'])

print('*'*20)
print("consistency_check='warn'")
print('*'*20)
ods=ODS()
ods=ods_sample()
ods.consistency_check='warn'
ods['equilibrium.time_slice[0].does_not_exist.global_quantities.ip']=1
print(ods['equilibrium.time_slice[0].does_not_exist.global_quantities.ip'])

print('*'*20)
print("consistency_check=True")
print('*'*20)
ods=ods_sample()
ods['equilibrium.time_slice[2].does_not_exist.global_quantities.ip']