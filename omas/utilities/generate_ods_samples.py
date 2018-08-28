from __future__ import print_function, division, unicode_literals

import os
from matplotlib import pyplot
from pprint import pprint
from omfit.classes.omfit_eqdsk import OMFITgeqdsk, OMFITsrc
from omas import *

# settings
os.environ['OMAS_DEBUG_TOPIC'] = 'imas'
omas_rcparams['allow_fake_imas_fallback'] = True

# read gEQDSK file in OMFIT, subsample, and convert to ODS
eq = OMFITgeqdsk(imas_json_dir + '/../samples/g145419.02100')
eq.resample(17)
ods = eq.to_omas()
# update sample file
save_omas_json(ods, imas_json_dir + '/../samples/sample_eq_ods.json')
