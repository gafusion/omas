#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic loading of data
=======================
This example illustrates how OMAS can load in memory only the data when it is first requested.
This approach can also be used to transfer data on demand from/to a server where an OMAS service is running.
What is done here for NC file, also works also for IMAS.
"""

import os
from omas import *

# Local or remote data transfer
# If remote, we start an omas service locally to which we then connect
remote = False
if remote is True:
    import sys
    import subprocess

    print('=' * 20)
    print('OMAS service')
    print('=' * 20)
    print('Staring omas service...')
    p = subprocess.Popen(sys.executable + ' ' + omas_service_script, shell=True, stdout=subprocess.PIPE)
    stdout = p.stdout.readline().decode('utf-8').strip()
    remote = stdout.split()[-1]
    print(stdout + '\n')

    print('=' * 20)
    print('OMAS client')
    print('=' * 20)
else:
    remote = None


# set OMAS_DEBUG_TOPIC to see when data is loaded dynamically
os.environ['OMAS_DEBUG_TOPIC'] = 'dynamic'

# generate some data and save it as a netcdf file
ods = ODS().sample(ntimes=2)
ods.save('test.nc')

# ODS.open() will keep the file descriptor open so that OMAS
# can load in memory only the data when it is first requested
# NOTE: one can use the `with` statement or open()/close()
ods = ODS()
with ods.open('test.nc', remote=remote):
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

# continue loading more data
with ods.open('test.nc', remote=remote):
    print(ods['equilibrium.time'])
