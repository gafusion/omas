#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic loading of experimental data
====================================
This example illustrates how OMAS can load experimental data on demand.
Only the data that is queried in the ODS will be loaded.
"""

import os
import logging
from omas import *
from matplotlib import pyplot
logging.basicConfig()
try:
    from omfit_classes.omfit_eqdsk import OMFITgeqdsk
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("If you want the full example, you need to install OMFIT.")
    OMFITgeqdsk = None
try:
    import MDSplus
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("If you want the full example, you need to install MDSplus.")
    MDSplus = None

os.environ['OMAS_DEBUG_TOPIC'] = 'dynamic'

if MDSplus is None:
    quit()

# load of some experiment quantities
ods = ODS()
with ods.open('machine', 'd3d', 168830, options={'EFIT_tree': 'EFIT01'}):
    ip = ods['equilibrium.time_slice.:.global_quantities.ip']
    print(f"Max Ip is {max(ip)} [A]")

# plot experiment equilibrium reconstruction w/ probes and PF coils
pyplot.figure()
ods = ODS()
with ods.open('machine', 'd3d', 168830, options={'EFIT_tree': 'EFIT02'}):
    ods.plot_equilibrium_CX(time=2.1)
    ods.plot_overlay(wall=True, magnetics=True, pf_active=True)
pyplot.show()

# plot magnetics
pyplot.figure()
ods = ODS()
with ods.open('machine', 'd3d', 168830):
    lines = pyplot.plot(ods[f'magnetics.b_field_pol_probe.:.field.time'].T, ods[f'magnetics.b_field_pol_probe.:.field.data'].T)
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Field [T]')
pyplot.show()

if OMFITgeqdsk is not None:
    # generate a D3D gEQDSK file from experimental data
    pyplot.figure()
    ods = ODS()
    with ods.open('machine', 'd3d', 168830, options={'EFIT_tree': 'EFIT02'}):
        gEQDSK = OMFITgeqdsk(None).from_omas(ods, time=1.1)
    gEQDSK.plot()
    pyplot.show()

    # generate a NSTX gEQDSK file from experimental data
    pyplot.figure()
    ods = ODS()
    with ods.open('machine', 'nstxu', 139047, options={'EFIT_tree': 'EFIT01'}):
        gEQDSK = OMFITgeqdsk(None).from_omas(ods, time=0.5)
    gEQDSK.plot()
    pyplot.show()
