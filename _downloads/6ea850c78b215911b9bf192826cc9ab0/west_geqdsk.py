#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gEQDSK file from IMAS WEST data
===============================
This example loads WEST IMAS equilibrium data and generates a gEQDSK file from it.

NOTE: There is an OMFIT script that does this (works also remotely) and can be called with::

    omfit gEQDSK_west "shot=55866, run=0, occurrence=1, time=47, resolution=129"

To run this, make sure to first::

    module load omfit
    module load IMAS/3.30.0-4.8.4

"""

from omas import ODS
import numpy
from omfit_eqdsk import OMFITgeqdsk
from matplotlib import pyplot

shot = 55866
run = 0
occurrence = 1  # 0:magnetics / 1:MSE
time = 47  # in seconds
resolution = 129  # grid resolution of the gEQDKS file

# load the data from IMAS
ods = ODS().load(
    'imas',
    user='public',
    machine='west',
    pulse=shot,
    run=run,
    occurrence={'equilibrium': occurrence},
    paths=[['equilibrium']],
    skip_uncertainties=True,
)

# get the time info for this IDS
t = ods['equilibrium'].time('time_slice')
time_index = numpy.argmin(abs(t - time))  # nearest time_index in the ODS

# grid interpolation can be time consuming, so we do it only for the requested time index
ods.physics_equilibrium_ggd_to_rectangular(time_index=[time_index], resolution=resolution, method='extrapolate')

# much faster than querying the wall IDS
ods.physics_wall_add('west')

# generate the gEQDSK file
g = OMFITgeqdsk(None).from_omas(ods, time_index=time_index)
g.deploy()

# plot
g.plot()
pyplot.show()
