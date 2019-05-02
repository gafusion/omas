#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCOS transformations
=====================
OMAS can seamlessly handle coordinates convention translations
"""

from __future__ import print_function, division, unicode_literals

import os

os.environ['OMAS_DEBUG_TOPIC'] = 'cocos'

from omas import *
import numpy

x = numpy.linspace(.1, 1, 10)

# use different COCOS storage conventions

ods = ODS(cocosio=11, cocos=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosio=11, cocos=2)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosio=2, cocos=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosio=2, cocos=2)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

# reassign the same value
ods = ODS(cocosio=2)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
ods['equilibrium.time_slice.0.profiles_1d.psi'] = ods['equilibrium.time_slice.0.profiles_1d.psi']
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

# use omas_environment
ods = ODS(cocosio=2)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
with omas_environment(ods, cocosio=11):
    assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x * (2 * numpy.pi)))
    print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])
