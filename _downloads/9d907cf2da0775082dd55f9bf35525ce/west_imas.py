#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extracting selected data from WEST
==================================
This simple example loads selected data from the WEST IMAS database

To run this, make sure to first::

    module load omfit
    module load IMAS/3.30.0-4.8.4

"""

from omas import *

ods = ODS()
with ods.open('imas', 'public', 'west', 55866, 0):
    x = ods['equilibrium.time']
    y = ods['equilibrium.time_slice.:.global_quantities.ip']

from matplotlib import pyplot

pyplot.plot(x, y)
pyplot.show()
