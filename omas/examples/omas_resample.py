#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interpolate whole ODS
=====================
Seamless handling of coordinates within OMAS makes it easy to reinterpolate a whole ODS on a new grid
"""

import numpy
from omas import *

# original ODS
ods = ODS()
ods.sample_equilibrium()

# interpolated ODS
ods_interpolated = ODS()

# define new psi grid of ods_interpolated
new_psi = numpy.linspace(ods['equilibrium.time_slice.0.profiles_1d.psi'][0], ods['equilibrium.time_slice.0.profiles_1d.psi'][-1], 21)
ods_interpolated['equilibrium.time_slice.0.profiles_1d.psi'] = new_psi

# interpolate whole ods on new psi grid
with omas_environment(ods_interpolated, coordsio=ods):
    ods_interpolated.update(ods)

# print some quantity from interpolated ods
assert len(ods_interpolated['equilibrium.time_slice.0.profiles_1d.pressure']) == 21
print(ods_interpolated['equilibrium.time_slice.0.profiles_1d.pressure'])
