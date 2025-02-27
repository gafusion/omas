#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Uncertain data
==============
This example shows how OMAS can seamlessly handle unceratain data leveraging the `uncertainties` Python package
"""

from omas import *
import os
import numpy
import uncertainties.unumpy as unumpy
from uncertainties import ufloat

# generate some uncertain data
ods = ODS()
ods['equilibrium.time'] = [0.0]
ods['equilibrium.time_slice[0].global_quantities.ip'] = ufloat(3, 0.1)
ods['thomson_scattering.channel[0].t_e.data'] = unumpy.uarray([1, 2, 3], [0.1, 0.2, 0.3])
ods['thomson_scattering.channel[0].n_e.data'] = numpy.array([1.0, 2.0, 3.0])
ods['thomson_scattering.time'] = numpy.linspace(0, 1, 3)
ods['thomson_scattering.ids_properties.homogeneous_time'] = 1

# save/load from pickle
print('== PKL ==')
try:
    __file__
except NameError:
    import inspect

    __file__ = inspect.getfile(lambda: None)
omas_testing_directory = omas_testdir()
save_omas_pkl(ods, omas_testdir(__file__) + '/test.pkl')
ods = load_omas_pkl(omas_testdir(__file__) + '/test.pkl')
print(ods)

# save/load from json
print('== JSON ==')
save_omas_json(ods, omas_testdir(__file__) + '/test.json')
ods = load_omas_json(omas_testdir(__file__) + '/test.json')
print(ods)

# save/load from nc
print('== NC ==')
save_omas_nc(ods, omas_testdir(__file__) + '/test.nc')
ods = load_omas_nc(omas_testdir(__file__) + '/test.nc')
print(ods)

# save/load from imas
print('== IMAS ==')
omas_rcparams['allow_fake_imas_fallback'] = True
save_omas_imas(ods, user=os.environ.get('USER', 'dummy_user'), machine='test', pulse=10, run=1, new=True)
ods = load_omas_imas(user=os.environ.get('USER', 'dummy_user'), machine='test', pulse=10, run=1, verbose=False)
print(ods)
