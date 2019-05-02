#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Save through data systems
=========================

This examples instantiates a sample ODS, and saves it and loads it trhough the different systems that OMAS supports.
Finally, a check is done to make sure that the final ODS is the same as the initial ODS.

.. figure:: ../images/omas_through_simple.png
  :align: center
  :width: 50%
  :alt: omas save data through different file formats
  :target: ../_images/omas_through_simple.png
"""

from __future__ import print_function, division, unicode_literals

import os

os.environ['OMAS_DEBUG_TOPIC'] = '*'

from omas import *

# load some sample data
ods_start = ods_sample()

# save/load Python pickle
filename = 'test.pkl'
save_omas_pkl(ods_start, filename)
ods = load_omas_pkl(filename)

# save/load ASCII json
filename = 'test.json'
save_omas_json(ods, filename)
ods = load_omas_json(filename)

# save/load NetCDF
filename = 'test.nc'
save_omas_nc(ods, filename)
ods = load_omas_nc(filename)

# save/load HDF5
filename = 'test.h5'
save_omas_h5(ods, filename)
ods = load_omas_h5(filename)

# remote save/load S3
filename = 'test.s3'
save_omas_s3(ods, filename)
ods = load_omas_s3(filename)

# save/load IMAS
omas_rcparams['allow_fake_imas_fallback'] = True
paths = save_omas_imas(ods, machine='ITER', pulse=1, new=True)
ods_end = load_omas_imas(machine='ITER', pulse=1, paths=paths)

# check data
check = different_ods(ods_start, ods_end)
if not check:
    print('OMAS data got saved and loaded correctly throughout')
else:
    pprint(check)
