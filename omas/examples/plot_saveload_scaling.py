#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Storage performance scaling
===========================
This example shows scaling performance of different
OMAS storage formats.
"""

import os
import time
import tempfile
from omas import *
import numpy
from pprint import pprint
from matplotlib import pyplot

ods = ODS()
ods.sample_equilibrium()

max_n = 3
samples = numpy.unique(list(map(int, numpy.logspace(0, max_n, 10 + 1)))).tolist()

times = {}
times['h5_save'] = []
times['h5_load'] = []
times['ds_save'] = []
times['ds_load'] = []
times['dx_load'] = []

print(samples)
for n in samples:
    print(n)

    # keep adding time slices
    for k in range(len(ods['equilibrium.time_slice']), n):
        ods.sample_equilibrium(time_index=k)

    # save load in HDF5
    if not len(times['h5_save']) or times['h5_save'][-1] < 1.0:
        t0 = time.time()
        save_omas_h5(ods, tempfile.gettempdir() + os.sep + 'tmp.h5')
        times['h5_save'].append(time.time() - t0)

        t0 = time.time()
        load_omas_h5(tempfile.gettempdir() + os.sep + 'tmp.h5')
        times['h5_load'].append(time.time() - t0)
    else:
        times['h5_save'].append(numpy.nan)
        times['h5_load'].append(numpy.nan)

    # save load in xarray dataset format
    if not len(times['ds_save']) or times['ds_save'][-1] < 1.0:
        t0 = time.time()
        save_omas_ds(ods, tempfile.gettempdir() + os.sep + 'tmp.ds')
        times['ds_save'].append(time.time() - t0)

        t0 = time.time()
        load_omas_ds(tempfile.gettempdir() + os.sep + 'tmp.ds')
        times['ds_load'].append(time.time() - t0)

        t0 = time.time()
        for k in range(10):
            load_omas_dx(tempfile.gettempdir() + os.sep + 'tmp.ds')
        times['dx_load'].append((time.time() - t0) / 10.)

    else:
        times['ds_save'].append(numpy.nan)
        times['ds_load'].append(numpy.nan)
        times['dx_load'].append(numpy.nan)

    if len(times['h5_save']) and times['h5_save'][-1] is numpy.nan and len(times['ds_save']) and times['ds_save'][-1] is numpy.nan:
        samples = samples[:samples.index(n) + 1]
        break

    pprint(times)

# plot scalings
for item in ['h5_save', 'h5_load', 'ds_save', 'ds_load', 'dx_load']:
    pyplot.loglog(samples, times[item], label='OMAS ' + item.replace('_', ' '), lw=1.5, ls=['-', '--']['save' in item])
pyplot.xlabel('# of Equilibrium Time Slices')
pyplot.ylabel('Time [s]')
pyplot.legend()
pyplot.show()
