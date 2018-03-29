from __future__ import print_function, division, unicode_literals

from omas import *
import os
import numpy
import uncertainties.unumpy as unumpy

ods = ODS()
ods['thomson_scattering.channel[0].t_e.data'] = unumpy.uarray([1,2,3],[.1,.2,.3])
ods['thomson_scattering.channel[0].n_e.data'] = numpy.array([1.,2.,3.])
ods['thomson_scattering.time']=numpy.linspace(0,1,3)
ods['thomson_scattering.ids_properties.homogeneous_time']=1

print('== PKL ==')
save_omas_pkl(ods,'test.pkl')
ods=load_omas_pkl('test.pkl')
print(ods)

print('== JSON ==')
save_omas_json(ods,'test.json')
ods=load_omas_json('test.json')
print(ods)

print('== NC ==')
save_omas_nc(ods,'test.nc')
ods=load_omas_nc('test.nc')
print(ods)

print('== IMAS ==')
omas_rcparams['allow_fake_imas_fallback']=True
save_omas_imas(ods, user=os.environ['USER'],machine='test', shot=10, run=1, new=True)
ods = load_omas_imas(user=os.environ['USER'],machine='test', shot=10, run=1, verbose=False)
print(ods)
