#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IDS to CPO mapping
==================
This is an example script translating some data that is stored in IMAS as an `ids`
to `cpo`, that is the EU-ITM data format. This example shows how OMAS can write
data to ITM.
"""

from __future__ import print_function, division, unicode_literals

from omas import *
import numpy
from pprint import pprint

# fill in with some test data
ids = ODS()
ids['equilibrium.code.name'] = 'test'
ids['equilibrium.code.version'] = 'v0.0'
ids['equilibrium.code.parameters'] = '<xml></xml>'
ids['equilibrium.time'] = numpy.linspace(0, 1, 3)
for itime in range(len(ids['equilibrium.time'])):
    indexes = {'itime': itime, 'iprof2d': 0}
    ids['equilibrium.time_slice[{itime}].profiles_1d.q'.format(**indexes)] = numpy.random.randn(5)
    ids['equilibrium.time_slice[{itime}].profiles_1d.rho_tor'.format(**indexes)] = numpy.random.randn(15)
    ids['equilibrium.time_slice[{itime}].profiles_2d[{iprof2d}].psi'.format(**indexes)] = numpy.reshape(numpy.random.randn(25), (5, 5))
    ids['core_profiles.profiles_1d[{itime}].electrons.temperature'.format(**indexes)] = numpy.random.randn(5)
    ids['core_profiles.profiles_1d[{itime}].electrons.density_thermal'.format(**indexes)] = numpy.random.randn(5)
    for iion in range(2):
        indexes['iion'] = iion
        ids['core_profiles.profiles_1d[{itime}].ion[{iion}].temperature'.format(**indexes)] = numpy.random.randn(5)
        ids['core_profiles.profiles_1d[{itime}].ion[{iion}].density'.format(**indexes)] = numpy.random.randn(5)

# map data
cpo = ids_cpo_mapper(ids)

# save CPO to ITM data system
with rcparams_environment(allow_fake_itm_fallback=True):
    save_omas_itm(cpo, machine='jet', pulse=1, new=True)
