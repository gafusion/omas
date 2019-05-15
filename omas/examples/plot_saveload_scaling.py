#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Storage performance scaling
===========================
This example shows a scaling performance study for
storaging data in hieararchical or tensor format
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

max_n = 100
max_samples = 11
stats_reps = 10
samples = numpy.unique(list(map(int, numpy.logspace(0, numpy.log10(max_n), max_samples)))).tolist()
max_samples = len(samples)

times = {}
for type in ['H', 'T']:  # hierarchical or tensor
    for action in ['R', 'W', 'M', 'A', 'S', 'B']:  # Read, Write, Mapping, Array access, Stripe access, Bulk access
        times[type + action] = []

for n in samples:
    print('%d/%d'%(n,samples[-1]))

    # keep adding time slices
    for k in range(len(ods['equilibrium.time_slice']), n):
        ods.sample_equilibrium(time_index=k)

    # save load in HDF5
    filename = tempfile.gettempdir() + os.sep + 'tmp.h5'
    t0 = time.time()
    save_omas_h5(ods, filename)
    times['HW'].append(time.time() - t0)

    t0 = time.time()
    load_omas_h5(filename)
    times['HR'].append(time.time() - t0)

    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            ods['equilibrium.time_slice.%d.profiles_1d.psi' % kk]
    times['HA'].append((time.time() - t0) / n / float(stats_reps))

    t0 = time.time()
    for kk in range(n):
        ods['equilibrium.time_slice.:.profiles_1d.psi'][:, 0]
    times['HS'].append((time.time() - t0) / n)

    t0 = time.time()
    for k in range(stats_reps):
        ods['equilibrium.time_slice.:.profiles_1d.psi']
    times['HB'].append((time.time() - t0) / float(stats_reps))

    filename = tempfile.gettempdir() + os.sep + 'tmp.ds'

    t0 = time.time()
    odx = ods_2_odx(ods)
    times['HM'].append(time.time() - t0)

    t0 = time.time()
    save_omas_dx(odx, filename)
    times['TW'].append(time.time() - t0)

    t0 = time.time()
    odx = load_omas_dx(filename)
    times['TR'].append(time.time() - t0)

    t0 = time.time()
    ods = odx_2_ods(odx)
    times['TM'].append(time.time() - t0)

    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            odx['equilibrium.time_slice.%d.profiles_1d.psi' % kk]
    times['TA'].append((time.time() - t0) / n / float(stats_reps))

    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            odx['equilibrium.time_slice.:.profiles_1d.psi'][:, 0]
    times['TS'].append((time.time() - t0) / n / float(stats_reps))

    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            odx['equilibrium.time_slice.:.profiles_1d.psi']
    times['TB'].append((time.time() - t0) / n / float(stats_reps))

print(samples)
pprint(times)

# plot read/write scaling
pyplot.figure()
for type in ['H', 'T']:  # hierarchical or tensor
    for action in ['R', 'W']:  # Read, Write
        pyplot.loglog(samples, times[type + action], label=type + action, lw=1.5, ls=['-', '--']['H' in type])
pyplot.xlabel('# of Equilibrium Time Slices')
pyplot.ylabel('Time [s]')
pyplot.legend(loc='upper left', frameon=False)
pyplot.title('Read/Write', y=0.85, x=0.7)

# plot mapping scaling
pyplot.figure()
for type in ['H', 'T']:  # hierarchical or tensor
    for action in ['M']:  # Mapping
        pyplot.loglog(samples, times[type + action], label=type + action, lw=1.5, ls=['-', '--']['H' in type])
pyplot.xlabel('# of Equilibrium Time Slices')
pyplot.ylabel('Time [s]')
pyplot.legend(loc='upper left', frameon=False)
pyplot.title('Mapping', y=0.85, x=0.7)

# plot access scaling
pyplot.figure()
for type in ['H', 'T']:  # hierarchical or tensor
    for action in ['A', 'S', 'B']:  # Array access, Stripe access, Bulk access
        pyplot.loglog(samples, times[type + action], label=type + action, lw=1.5, ls=['-', '--']['H' in type])
pyplot.xlabel('# of Equilibrium Time Slices')
pyplot.ylabel('Time [s]')
pyplot.legend(loc='upper left', frameon=False)
pyplot.title('Access', y=0.85, x=0.7)

pyplot.show()
