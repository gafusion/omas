#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coordinates interpolation
=========================
This example illustrates how OMAS can automatically interpolate data
defined on coordinates that were already present in the data structure.
This feature is extremely useful when different codes that have different
computational grids need to read/write parts of the same data structure.
"""

from __future__ import print_function, division, unicode_literals

import numpy
from omas import *

# if a coordinate exists, then that is the coordinate that it is used
ods1 = ODS()
ods1['equilibrium.time_slice[0].profiles_1d.psi'] = numpy.linspace(0, 1, 10)
with coords_environment(ods1, {'equilibrium.time_slice[0].profiles_1d.psi': numpy.linspace(0, 1, 5)}):
    ods1['equilibrium.time_slice[0].profiles_1d.f'] = numpy.linspace(0, 1, 5)
print('ods1 f has length %d' % len(ods1['equilibrium.time_slice[0].profiles_1d.f']))
print()

# if a does not exists, then that coordinate is set
ods2 = ODS()
with coords_environment(ods2, {'equilibrium.time_slice[0].profiles_1d.psi': numpy.linspace(0, 1, 5)}):
    ods2['equilibrium.time_slice[0].profiles_1d.pressure'] = numpy.linspace(0, 1, 5)
print('ods2 p has length %d' % len(ods2['equilibrium.time_slice[0].profiles_1d.pressure']))
print()

# coordinates can be taken from existing ODSs
ods3 = ODS()
with coords_environment(ods3, ods1):
    ods3['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
with coords_environment(ods3, ods2):
    ods3['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2['equilibrium.time_slice[0].profiles_1d.pressure']
print('ods3 f has length %d' % len(ods3['equilibrium.time_slice[0].profiles_1d.f']))
print('ods3 p has length %d' % len(ods3['equilibrium.time_slice[0].profiles_1d.pressure']))
print()

# order matters
ods4 = ODS()
with coords_environment(ods4, ods2):
    ods4['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2['equilibrium.time_slice[0].profiles_1d.pressure']
with coords_environment(ods4, ods1):
    ods4['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
print('ods4 f has length %d' % len(ods4['equilibrium.time_slice[0].profiles_1d.f']))
print('ods4 p has length %d' % len(ods4['equilibrium.time_slice[0].profiles_1d.pressure']))
print()

###################################
# The above example prints the following::
#
#     ods1 f has length 10
#
#     ods2 p has length 5
#
#     ods3 f has length 10
#     ods3 p has length 10
#
#     ods4 f has length 5
#     ods4 p has length 5