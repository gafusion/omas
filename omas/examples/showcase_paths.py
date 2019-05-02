#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
showcase paths
==============
This example shows how OMAS supports dynamic path crection using different syntaxes.

.. figure:: ../images/dynamic_path_testimonial.png
  :align: center
  :width: 100%
  :alt: What people say about OMAS dynamic path creation
  :target: ../_images/dynamic_path_testimonial.png
"""

from __future__ import print_function, division, unicode_literals
import numpy

from omas import *

ods = ODS()

# without dynamic path creation one must use Python approach to create nested dictionaries
# this can be extremely tedious!
ods.dynamic_path_creation = False
ods['equilibrium'] = ODS()
ods['equilibrium']['time_slice'] = ODS()
ods['equilibrium']['time_slice'][0] = ODS()
ods['equilibrium']['time_slice'][0]['time'] = 1000
assert (ods['equilibrium']['time_slice'][0]['time'] == 1000.)

# Dynamic path creation (True by default) makes life easier.
# NOTE: OMAS supports different data access syntaxes
ods.dynamic_path_creation = True

# access data as dictionary
ods['equilibrium']['time_slice'][0]['time'] = 1000.
assert (ods['equilibrium']['time_slice'][0]['time'] == 1000.)

# access data as string
ods['equilibrium.time_slice.1.time'] = 2000.
assert (ods['equilibrium.time_slice.1.time'] == 2000.)

# access data as string (square brackets for arrays of structures)
ods['equilibrium.time_slice[2].time'] = 3000.
assert (ods['equilibrium.time_slice[2].time'] == 3000.)

# access data with path list
ods[['equilibrium', 'time_slice', 3, 'time']] = 4000.
assert (ods[['equilibrium', 'time_slice', 3, 'time']] == 4000.)

# access data with mix and match approach
ods['equilibrium']['time_slice.4.time'] = 5000.
assert (ods['equilibrium']['time_slice.4.time'] == 5000.)

# =============
# Data slicing
# =============

# classic ways to access data across an array of structures
data = []
for k in ods['equilibrium.time_slice'].keys():
    data.append(ods['equilibrium.time_slice'][k]['time'])
data = numpy.array(data)
assert (numpy.all(data == numpy.array([1000., 2000., 3000., 4000., 5000.])))

# access data across an array of structures via data slicing
data = ods['equilibrium.time_slice.:.time']
assert (numpy.all(data == numpy.array([1000., 2000., 3000., 4000., 5000.])))

# =========================
# .setdefault() and .get()
# =========================

# like for Python dictionaries .setdefault() will set an entry with its
# default value (second argument) only if that entry does not exists already
ods = ODS()
ods['equilibrium.time_slice.0.global_quantities.ip'] = 6
ods.setdefault('equilibrium.time_slice.0.global_quantities.ip', 5)
assert (ods['equilibrium.time_slice.0.global_quantities.ip'] == 6)
ods = ODS()
ods.setdefault('equilibrium.time_slice.0.global_quantities.ip', 5)
assert (ods['equilibrium.time_slice.0.global_quantities.ip'] == 5)

# like for Python dictionaries .get() return the value of an entry or its
# default value (second argument) if that does not exists
ods = ODS()
ods.get('equilibrium.time_slice.0.global_quantities.ip', 5)
assert ('equilibrium.time_slice.0.global_quantities.ip' not in ods)

# ========
# Cleanup
# ========

# Dynamic path creation can leave empty trees behind in case of bad IMAS location is entered.
# This is inevitable when splitting the path in individual pieces.
# These leftover are often innocuous, and occur only during the development stages.
# Use of .prune() to clean empty branches left behind from dynamic path creation
ods = ODS()
try:
    ods['equilibrium']['time_slice'][0]['global_quantities']['not_valid'] = 6
except LookupError:
    n = ods.prune()
    assert (n == 4)
    assert (len(ods) == 0)

# Note that single string access does not leave empty branches
ods = ODS()
try:
    ods['equilibrium.time_slice.0.global_quantities.asdasd'] = 6
except LookupError:
    assert (len(ods) == 0)
