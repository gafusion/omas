#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Units conversions
=================
This example illustrates how OMAS can automatically translate units by leveraging the `pint` Python package
"""

from __future__ import print_function, division, unicode_literals

from omas import *

import pint

ureg = pint.UnitRegistry()

ods = ODS()

invalue = 8.0 * ureg.milliseconds
print('Data input: %s' % invalue)
ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement'] = invalue

outvalue = ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement']
print('Data output without units support: %s (IMAS uses MKS)' % outvalue)

with omas_environment(ods, unitsio=True):
    outvalue = ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement']
print('Data output with units support: %s' % outvalue)

###################################
# .. code-block:: none
#
#     Data input: 8.0 millisecond
#     Data output without units support: 0.008 (IMAS uses MKS)
#     Data output with units support: 0.008 second
#
