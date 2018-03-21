from __future__ import print_function, division, unicode_literals

import os
os.environ['OMAS_DEBUG_TOPIC'] = '*'

from omas import *

ods=ods_sample()

ods['equilibrium.time_slice[2].does_not_exist.global_quantities.ip']