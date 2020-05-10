#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic loading of IMAS data
============================
This example illustrates how OMAS can load in memory IMAS data only when it is first requested
"""

import os,sys
from omas import *

# set OMAS_DEBUG_TOPIC to see when data is loaded dynamically
#os.environ['OMAS_DEBUG_TOPIC'] = 'dynamic'

# generate some data and save it as a netcdf file
#ods = ods_sample(ntimes=2)
#ods.save('imas',os.environ['USER'],'DIII-D',1000,0,new=True,verbose=True)

ods = ODS()
with ods.open('imas',os.environ['USER'],'DIII-D',1000,0):
    print(ods.keys(True))
    print(ods['equilibrium.time_slice.0.profiles_1d'].keys(True))

sys.exit()

# ODS.open() will keep the file descriptor open so that OMAS
# can load in memory only the data when it is first requested
# NOTE: one can use the `with` statement or open()/close()
ods = ODS()
with ods.open('imas',os.environ['USER'],'DIII-D',1000,0):
    # data gets read from NC file when first requested
    print(ods['equilibrium.time_slice.:.global_quantities.ip'])
    # then it is in memory
    print(ods['equilibrium.time_slice.0.global_quantities.ip'])

# save it as a pickle (which will keep memory of the dynamic nature of the ODS)
ods.save('test.pkl')

# load the data back
ods = ODS()
ods.load('test.pkl')

# the data that was loaded is stored in the pickle
print(ods.flat().keys())

# re-open the file descriptor to continue loading more data
with ods.open('imas',os.environ['USER'],'DIII-D',1000,0):
    print(ods['equilibrium.time'])

