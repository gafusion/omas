#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic loading of IMAS data
============================
This example illustrates how OMAS can load in memory IMAS data only when it is first requested
"""

import os
from omas import *

# set OMAS_DEBUG_TOPIC to see when data is loaded dynamically
os.environ['OMAS_DEBUG_TOPIC'] = 'dynamic'

# generate some data and save it in IMAS
ods = ODS().sample(ntimes=2)
ods.save('imas', os.environ['USER'], 'DIII-D', 1000, 0, new=True, verbose=True)

# ODS.open() will keep the file descriptor open so that OMAS
# can load in memory only the data when it is first requested
# NOTE: one can use the `with` statement or open()/close()
ods = ODS()
with ods.open('imas', os.environ['USER'], 'DIII-D', 1000, 0):
    # data gets read from IMAS when first requested
    print(ods['equilibrium.time_slice.:.global_quantities.ip'])
    # then it is in memory
    print(ods['equilibrium.time_slice.0.global_quantities.ip'])

# save it as a pickle
ods.save('test.pkl')

# load the data back
ods = ODS()
ods.load('test.pkl')

# the data that was loaded is stored in the pickle
print(ods.flat().keys())

# continue loading more data
with ods.open('imas', os.environ['USER'], 'DIII-D', 1000, 0):
    print(ods['equilibrium.time'])

# tell us what IMAS elements have data
with ods.open('imas', os.environ['USER'], 'DIII-D', 1000, 0):
    print(ods.keys())
    print(ods['equilibrium.time_slice.0.profiles_1d'].keys())
