#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMAS plot examples
==================
This example loads some data from S3, augments the ODS with pressure information, and generates some plots
"""

from matplotlib import pyplot
from omas import *

# load some data from S3
ods = load_omas_s3('OMFITprofiles_sample', user='omas_shared')

# augment ODS with pressure information
ods.physics_core_profiles_pressures()

# omas plot for pressures
ods.plot_core_profiles_pressures()
pyplot.show()

# omas plot for core profiles
ods.plot_core_profiles_summary()
pyplot.show()

# omas plot for equilibrium
omas_plot.equilibrium_summary(ods, linewidth=1, label='my equilibrium')
pyplot.show()

# omas plot for transport fluxes
ods = ODS().sample(5)
ods.plot_core_transport_fluxes()
pyplot.show()
