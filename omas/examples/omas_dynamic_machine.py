#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic loading of experimental data
====================================
This example illustrates how OMAS can load in memory only the data when it is first requested.
This approach can also be used to transfer data on demand.
"""

import os
from omas import *
from omfit.classes.omfit_eqdsk import OMFITgeqdsk
from matplotlib import pyplot
from pprint import pprint

os.environ['OMAS_DEBUG_TOPIC'] = 'dynamic'

ods = ODS()
with ods.open('machine', 'd3d', 168830):
    g = OMFITgeqdsk(None).from_omas(ods, 100)

pprint(list(ods.flat().keys()))

g.plot()
pyplot.show()
