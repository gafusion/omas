#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scaling IMAS performance
========================
This example shows a scaling performance study for manipulating OMAS data in hierarchical or tensor format.

The **hierarchical organization of the IMAS data** structure can in some situations hinder IMAS's ability to efficiently manipulate large data sets.
This contrasts to the **multidimensional arrays (ie. tensors)** approach that is commonly used in computer science for high-performance numerical calculations.

Based on this observation OMAS implements a transformation that casts the data that is contained in the IMAS hierarchical structure as a list of tensors, by taking advantage of the homogeneity of grid sizes that is commonly found across arrays of structures.
Such transformation and a summary of the scaling results are illustrated here for an hypothetical IDS that has data organized as a series of time-slices:

.. figure:: ../images/odx_concept.png
  :align: center
  :width: 100%
  :alt: OMAS implements a transformation that casts the data that is contained in the IMAS hierarchical structure as a list of tensors
  :target: /.._images/odx_concept.png

The favorable scaling that is observed when representing IMAS data in tensor form makes a strong case for adopting it.
Implementing the same system as part of the IMAS backend storage of data and in memory representation would likely greatly benefit IMAS performance in many real-world applications.

The new tensors representation would also greatly simplify the integration of IMAS with a broad range of tools and numerical libraries that are commonly used across many fields of science.

Finally, the addition of an extra dimension to the tensors could be used to efficiently store multiple realizations of signals from a distribution function of uncertain quantities.
Such feature would enable support of uncertainty quantification workflows and Bayesian integrated data analyses within IMAS.

Scaling study in detail
-----------------------

OMAS can seamlessly use either hierarchical or tensor representations as the backend for storing data both in memory and on file, and transform from one format to the other.
The mapping function is generic and can handle nested hierarchical list of structures (not only in time).
Also OMAS can automatically determine which data can be collected across the hierarchical structure, which cannot, and seamlessly handle both at the same time.

The following diagram summarizes the tests performed in this scaling study.
Benchmarks show that most operations stemming from the hierarchical representation of the data scale linearly with with the number of time-slices in the sample IDS (**red markers** in the diagram), whereas operations that make only use of the tensor representation show little to no dependency on the dataset size (**green markers** in the diagram).
As a result the tensors representation can be several orders of magnitude faster than a hierarchical organization, even for datasets of modest size.

.. figure:: ../images/odx_flow.png
  :align: center
  :width: 50%
  :alt: OMAS can seamlessly use either hierarchical or tensor representations as backed for storing data both in memory and on file, and transform from one format to the other
  :target: /.._images/odx_flow.png

Scaling plots and code used for the benchmark follow:

"""

import os
import time
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

try:
    __file__
except NameError:
    import inspect

    __file__ = inspect.getfile(lambda: None)
for n in samples:
    print('%d/%d' % (n, samples[-1]))

    # keep adding time slices to the data structure
    for k in range(len(ods['equilibrium.time_slice']), n):
        ods.sample_equilibrium(time_index=k)

    # hierarchical write to HDF5
    filename = omas_testdir(__file__) + '/tmp.h5'
    t0 = time.time()
    save_omas_h5(ods, filename)
    times['HW'].append(time.time() - t0)

    # hierarchical read from HDF5
    t0 = time.time()
    load_omas_h5(filename)
    times['HR'].append(time.time() - t0)

    # hierarchical access to individual array
    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            ods['equilibrium.time_slice.%d.profiles_1d.psi' % kk]
    times['HA'].append((time.time() - t0) / n / float(stats_reps))

    # hierarchical slice across the data structure
    t0 = time.time()
    for kk in range(n):
        ods['equilibrium.time_slice.:.profiles_1d.psi'][:, 0]
    times['HS'].append((time.time() - t0) / n)

    # hierarchical bulk access to data
    t0 = time.time()
    for k in range(stats_reps):
        ods['equilibrium.time_slice.:.profiles_1d.psi']
    times['HB'].append((time.time() - t0) / float(stats_reps))

    # hierarchical mapping to tensor
    t0 = time.time()
    odx = ods_2_odx(ods)
    times['HM'].append(time.time() - t0)

    filename = omas_testdir(__file__) + '/tmp.ds'

    # tensor write to HDF5
    t0 = time.time()
    save_omas_dx(odx, filename)
    times['TW'].append(time.time() - t0)

    # tensor read from HDF5
    t0 = time.time()
    odx = load_omas_dx(filename)
    times['TR'].append(time.time() - t0)

    # tensor mapping to hierarchical
    t0 = time.time()
    ods = odx_2_ods(odx)
    times['TM'].append(time.time() - t0)

    # tensor access to individual array
    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            odx['equilibrium.time_slice.%d.profiles_1d.psi' % kk]
    times['TA'].append((time.time() - t0) / n / float(stats_reps))

    # tensor slice across the data structure
    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            odx['equilibrium.time_slice.:.profiles_1d.psi'][:, 0]
    times['TS'].append((time.time() - t0) / n / float(stats_reps))

    # tensor bulk access to data
    t0 = time.time()
    for k in range(stats_reps):
        for kk in range(n):
            odx['equilibrium.time_slice.:.profiles_1d.psi']
    times['TB'].append((time.time() - t0) / n / float(stats_reps))

# print numbers to screen
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
