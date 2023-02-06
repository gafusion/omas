import os, sys

sys.path.insert(0, os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])

import numpy
from matplotlib import pyplot
from pprint import pprint
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from omfit_classes.omfit_onetwo import OMFITstatefile
from omas.machine_mappings.d3d import magnetics_hardware
from omas import *

# settings
os.environ['OMAS_DEBUG_TOPIC'] = 'imas'
omas_rcparams['allow_fake_imas_fallback'] = True

# generate magnetics sample
ods = ODS()
magnetics_hardware(ods, 145419)
ods['magnetics.time'] = [0.0]
ods.save(omas_dir + 'samples/sample_magnetics_ods.json')

# generate equilibrium sample
eq = OMFITgeqdsk(omas_dir + 'samples/g145419.02100')
eq.resample(17)
eq['fluxSurfaces'].changeResolution(1)
ods = eq.to_omas()
ods.consistency_check = 'strict_drop_warn'
ods.save(omas_dir + 'samples/sample_equilibrium_ods.json')

# generate core_profiles and core_sources sample
state = OMFITstatefile(omas_dir + 'samples/state145419_02100.nc')
state.load()
ods = state.to_omas()

# subsample core_profiles and core_sources
coordsio = {}
coordsio['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = numpy.linspace(0, 1, 11)
for k in ods['core_sources.source']:
    coordsio['core_sources.source.%d.profiles_1d.0.grid.rho_tor_norm' % k] = numpy.linspace(0, 1, 11)

# write core_profiles and core_sources sample
for what in ['core_sources', 'core_profiles']:
    ods_subsampled = ODS()
    with omas_environment(ods, coordsio=coordsio):
        ods_subsampled[what].update(ods[what])
    ods_subsampled.save(omas_dir + f'samples/sample_{what}_ods.json')
