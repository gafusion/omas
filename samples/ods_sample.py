from __future__ import print_function, division, unicode_literals

import os
os.environ['OMAS_DEBUG_TOPIC'] = '*'

from omas import *

# test generation of a sample ods
ods=ods_sample()
print(ods)

# test getting information about a ids structure
print(omas_info('equilibrium'))