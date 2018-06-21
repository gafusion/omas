#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCOS
=====
OMAS can seamlessly handle coordinates convention translations
"""

from __future__ import print_function, division, unicode_literals

import os

os.environ['OMAS_DEBUG_TOPIC'] = 'cocos'

from omas import *
import numpy

x = numpy.linspace(.1, 1, 10)

# use different COCOS storage conventions

ods = ODS(cocosin=11, cocos=11, cocosout=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosin=11, cocos=2, cocosout=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

# use different COCOS in/out

ods = ODS(cocosin=2, cocosout=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x * 2 * numpy.pi))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosin=2, cocosout=2)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

# use cocos_environment

ods = ODS(cocosin=2)
with cocos_environment(ods, cocosin=11, cocosout=11):
    ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
    assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
    print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x * 2 * numpy.pi))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])
