from omas import *
import numpy
import uncertainties.unumpy as unumpy

ods=omas()
ods['thomson_scattering.channel[0].t_e.data'] = unumpy.uarray([1,2,3],[.1,.2,.3])

save_omas_pkl(ods,'test.pkl')
ods=load_omas_pkl('test.pkl')
print(ods)

save_omas_json(ods,'test.json')
ods=load_omas_json('test.json')
print(ods)

save_omas_nc(ods,'test.nc')
ods=load_omas_nc('test.nc')
print(ods)

omas_rcparams['allow_fake_imas_fallback']=True
save_omas_imas(ods, tokamak='ITER', shot=1, new=True)
ods = load_omas_imas(tokamak='ITER', shot=1)
print(ods)