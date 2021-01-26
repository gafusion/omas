#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic loading of experimental data
====================================
This example illustrates how OMAS can load experimental data on demand.
Only the data that is queried in the ODS will be loaded.
"""

import os
from omas import *
from omfit.classes.omfit_eqdsk import OMFITgeqdsk
from matplotlib import pyplot

os.environ['OMAS_DEBUG_TOPIC'] = 'dynamic'

# access some experimental data
pyplot.figure()
ods = ODS()
with ods.open('machine', 'd3d', 168830, options={'EFIT_tree': 'EFIT01'}):
    pyplot.plot(ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.ip'])
    pyplot.xlabel(f"[{ods.info('equilibrium.time')['units']}]")
    pyplot.ylabel(f"[{ods.info('equilibrium.time_slice.:.global_quantities.ip')['units']}]")
pyplot.show()

# generate a gEQDSK file from experimental data
ods1 = ODS()
with ods1.open('machine', 'd3d', 168830, options={'EFIT_tree': 'EFIT01'}):
    g0 = OMFITgeqdsk(None).from_omas(ods1, time=1.1)
    # notice that subsequent MDS+ calls for the same data are cached
    g1 = OMFITgeqdsk(None).from_omas(ods1, time=2.1)

# generate another one gEQDSK file from experimental data
ods2 = ODS()
with ods2.open('machine', 'd3d', 168830, options={'EFIT_tree': 'EFIT02'}):
    g2 = OMFITgeqdsk(None).from_omas(ods2, time=2.1)

pyplot.figure()
g0.plot()
g1.plot()
g2.plot()
pyplot.show()
