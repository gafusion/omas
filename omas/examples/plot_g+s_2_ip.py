#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gEQDSK + statefile to input.profiles
====================================
This example shows how OMAS can be used to generate a GACODE input.profiles
file given a gEQDSK file and a ONETWO statefile.
"""

from __future__ import print_function, division, unicode_literals

from matplotlib import pyplot
from omas import *

from omfit.classes.omfit_eqdsk import OMFITgeqdsk
from omfit.classes.omfit_onetwo import OMFITstatefile
from omfit.classes.omfit_gacode import OMFITgacode

gfilename = '/Users/meneghini/tmp/ONETWO_files/g0.03000'
sfilename = '/Users/meneghini/tmp/ONETWO_files/statefile_3.000000E+00.nc'
ipfilename = '/Users/meneghini/tmp/ONETWO_files/input.profiles'

gEQDSK = OMFITgeqdsk(gfilename)
statefile = OMFITstatefile(sfilename)

ods = gEQDSK.to_omas()
ods = statefile.to_omas(ods)

ip2 = OMFITgacode(ipfilename).from_omas(ods)

ip2.plot()
pyplot.show()
