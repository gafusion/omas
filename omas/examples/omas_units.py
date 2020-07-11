#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Units conversions
=================
This example illustrates how OMAS can automatically translate units by leveraging the `pint` Python package
"""

from omas import *
import numpy
import pint

ureg = pint.UnitRegistry()

ods = ODS()

# populate ODS feeding data that has units
invalue = 8.0 * ureg.milliseconds
print('Data input: %s' % invalue)
ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement'] = invalue

# get data without units info
outvalue = ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement']
print('Data output without units support: %s   #(IMAS uses MKS)' % outvalue)

# get data with units information
with omas_environment(ods, unitsio=True):
    outvalue = ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement']
print('Data output with units support: %s' % outvalue)

# how to manipulate data that has units
ods.sample()
with omas_environment(ods, unitsio=True):
    ne = ods['core_profiles.profiles_1d.0.electrons.density_thermal']
    print(f'Mean density in  m^-3: {numpy.mean(ne.to("m^-3").magnitude):3.3g}')
    print(f'Mean density in cm^-3: {numpy.mean(ne.to("cm^-3").magnitude):3.3g}')

###################################
# .. code-block:: none
#
#     Data input: 8.0 millisecond
#     Data output without units support: 0.008 (IMAS uses MKS)
#     Data output with units support: 0.008 second
#     Mean density in  m^-3: 4.9e+19
#     Mean density in cm^-3: 4.9e+13
