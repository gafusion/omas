#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMFIT gEQDSK and IMAS
=====================
This example loads a gEQDSK file using the OMFITgeqdsk class.
The gEQKDS file is then save to IMAS and loaded back.

Prior running this script, the following commands must be typed at the teriminal
> import IMAS OMAS
> imasdb ITER

.. figure:: ../images/eq_omas_omfit.png
  :align: center
  :width: 75%
  :alt: OMFIT+OMAS facilitate save/load gEQDSK to/from IMAS
  :target: /.._images/eq_omas_omfit.png

"""

import os
from matplotlib import pyplot

from omfit_classes.omfit_eqdsk import OMFITgeqdsk, OMFITsrc
from omas import ODS, fakeimas

# settings
os.environ['OMAS_DEBUG_TOPIC'] = 'imas'

# read gEQDSK file in OMFIT
eq = OMFITgeqdsk(OMFITsrc + '/../samples/g133221.01000')

# convert gEQDSK to OMAS data structure
ods = eq.to_omas()

# allow fallback on fake IMAS environment in OMAS in case real IMAS installation is not present
with fakeimas.fake_environment('fallback'):
    # save OMAS data structure to IMAS
    paths = ods.save('imas', machine='DIII-D', pulse=133221, new=True)

    # load OMAS data structure from IMAS
    ods1 = ODS().load('imas', machine='DIII-D', pulse=133221)

# generate gEQDSK file from OMAS data structure
eq1 = OMFITgeqdsk('g133221.02000').from_omas(ods1)

# save gEQDSK file
eq1.deploy('g133221.02000')

# plot
eq.plot()
eq1.plot()
pyplot.show()
