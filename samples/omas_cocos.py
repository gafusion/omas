from __future__ import print_function, division, unicode_literals

import os
os.environ['OMAS_DEBUG_TOPIC'] = 'cocos'

from omas import *
import numpy

x=numpy.linspace(.1, 1, 10)

ods = ODS(cocosin=11, cocos=11, cocosout=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] =x
assert(numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'],x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosin=11, cocos=2, cocosout=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert(numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'],x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosin=2, cocosout=11)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert(numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'],-x*2*numpy.pi))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosin=2, cocosout=2)
ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert(numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'],x))
print(ods['equilibrium.time_slice.0.profiles_1d.psi'])

ods = ODS(cocosin=2)
with cocos_environment(ods, cocosin=11, cocosout=11) as ods1:
    ods1['equilibrium.time_slice.0.profiles_1d.psi'] = x
    assert(numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'],x))
    print(ods1['equilibrium.time_slice.0.profiles_1d.psi'])

ods1['equilibrium.time_slice.0.profiles_1d.psi'] = x
assert(numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'],-x*2*numpy.pi))
print(ods1['equilibrium.time_slice.0.profiles_1d.psi'])

# from omas.omas_utils import _structures as structures
# from omas.omas_utils import *
# for item in sorted(structures.values()[0].keys()):
#     if '_error_' not in item:
#         print("cocos_signals['%s']="%i2o(item))
