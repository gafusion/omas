from __future__ import print_function, division, unicode_literals

from omas import *
import numpy
from pprint import pprint

# fill in with some test data
ids_in = omas()
ids_in['equilibrium.code.name'] = 'test'
ids_in['equilibrium.code.version'] = 'v0.0'
ids_in['equilibrium.code.parameters'] = '<xml></xml>'
ids_in['equilibrium.time'] = numpy.linspace(0, 1, 3)
for itime in range(len(ids_in['equilibrium.time'])):
    indexes={'itime':itime,'iprof2d':0}
    ids_in['equilibrium.time_slice[{itime}].profiles_1d.q'.format(**indexes)] = numpy.random.randn(5)
    ids_in['equilibrium.time_slice[{itime}].profiles_1d.rho_tor'.format(**indexes)] = numpy.random.randn(15)
    ids_in['equilibrium.time_slice[{itime}].profiles_2d[{iprof2d}].psi'.format(**indexes)] = numpy.reshape(numpy.random.randn(25), (5, 5))
    ids_in['core_profiles.profiles_1d[{itime}].electrons.temperature'.format(**indexes)] = numpy.random.randn(5)
    ids_in['core_profiles.profiles_1d[{itime}].electrons.density_thermal'.format(**indexes)] = numpy.random.randn(5)
    for iion in range(2):
        indexes['iion']=iion
        ids_in['core_profiles.profiles_1d[{itime}].ion[{iion}].temperature'.format(**indexes)] = numpy.random.randn(5)
        ids_in['core_profiles.profiles_1d[{itime}].ion[{iion}].density'.format(**indexes)] = numpy.random.randn(5)

# define mappings
translate = {}

translate['equilibrium.code.name'] = 'equilibrium[{itime}].codeparam.codename'
translate['equilibrium.code.version'] = 'equilibrium[{itime}].codeparam.codeversion'
translate['equilibrium.code.parameters'] = 'equilibrium[{itime}].codeparam.parameters'
# 0D
translate['equilibrium.time[{itime}]'] = 'equilibrium[{itime}].time'
# 1D
translate['equilibrium.time_slice[{itime}].profiles_1d.q'] = 'equilibrium[{itime}].profiles_1d.q'
translate['equilibrium.time_slice[{itime}].profiles_1d.rho_tor'] = 'equilibrium[{itime}].profiles_1d.rho_tor'
# 2D
translate['equilibrium.time_slice[{itime}].profiles_2d[{iprof2d}].psi'] = 'equilibrium[{itime}].profiles_2d[{iprof2d}].psi'

translate['core_profiles.profiles_1d[{itime}].electrons.temperature'] = 'coreprof[{itime}].te.value'
translate['core_profiles.profiles_1d[{itime}].electrons.density_thermal'] = 'coreprof(itime).ne.value'
translate['core_profiles.profiles_1d[{itime}].ion[{iion}].temperature'] = 'coreprof[{itime}].ti.value[{iion}]'
translate['core_profiles.profiles_1d[{itime}].ion[{iion}].density'] = 'coreprof[{itime}].ni.value[{iion}]'

# from IDS to CPO
cpo = omas_data_mapper(ids_in, translate, verbose=True)
pprint(cpo)

# from CPO to IDS
ids_out = omas_data_mapper(cpo, translate, flip_translate=True, verbose=True)

# check that data did not get lost
check = different_ods(ids_in, ids_out)
if not check:
    print('OMAS data got translated correctly')
else:
    print(check)

# save CPO to ITM data system
omas_rcparams['allow_fake_itm_fallback']=True
save_omas_itm(cpo, tokamak='jet', shot=1, new=True)
