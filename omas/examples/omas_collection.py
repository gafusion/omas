#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMAS data collection
=====================
ODC class can be used to collate ODSs, which is useful for collection of data from:
* different experimental shots
* different simulations
"""

from omas import *

print('# collection identifier as a string')
odc = ODC()
odc['sim1.equilibrium.time_slice.0.global_quantities.ip'] = 1000.0
odc['sim3.equilibrium.time_slice.0.global_quantities.ip'] = 2000.0
odc['sim2.equilibrium.time_slice.0.global_quantities.ip'] = 3000.0
print(f"  keys  : {odc.keys()}")
print(f"  values: {odc[':.equilibrium.time_slice.0.global_quantities.ip']}")
print()

print('# collection identifier as consecutive integers')
odc = ODC()
odc['0.equilibrium.time_slice.0.global_quantities.ip'] = 1000.1
odc['1.equilibrium.time_slice.0.global_quantities.ip'] = 2000.1
odc['2.equilibrium.time_slice.0.global_quantities.ip'] = 3000.1
print(f"  keys  : {odc.keys()}")
print(f"  values: {odc[':.equilibrium.time_slice.0.global_quantities.ip']}")
print()

print('# collection identifier as non-consecutive integers')
odc = ODC()
odc['133221.equilibrium.time_slice.0.global_quantities.ip'] = 1000.2
odc['133229.equilibrium.time_slice.0.global_quantities.ip'] = 1000.2
odc['133230.equilibrium.time_slice.0.global_quantities.ip'] = 1000.2
print(f"  keys  : {odc.keys()}")
print(f"  values: {odc[':.equilibrium.time_slice.0.global_quantities.ip']}")
print()

print('# data slicing across collection as well as ODSs')
odc = ODC()
for k in range(5):
    odc[f'133221.equilibrium.time_slice.{k}.global_quantities.ip'] = 1000.0 + k + 1
    odc[f'133229.equilibrium.time_slice.{k}.global_quantities.ip'] = 2000.0 + k + 1
    odc[f'133230.equilibrium.time_slice.{k}.global_quantities.ip'] = 3000.0 + k + 1
print(f"  keys  : {odc.keys()}")
print(f"  values:\n{odc[':.equilibrium.time_slice.:.global_quantities.ip']}")
print()
