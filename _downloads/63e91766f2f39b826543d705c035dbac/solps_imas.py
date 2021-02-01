#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOLPS and IMAS
==============
Using OMAS to load data from a SOLPS simulation stored in the ITER scenario database

Prior running this script, the following commands must be typed at the teriminal
> import IMAS OMAS
"""

import os
from pprint import pprint
from numpy import *
from omas import *

# load an ITER scenario that has SOLPS data in an ODS
# NOTE: loading the ggd data structure can take some time
ods = load_omas_iter_scenario(pulse=102292, run=1)

# print paths
pprint(ods.pretty_paths())

# add some arbitrary data to an entry in the tree
ods['edge_profiles.ggd[0].electrons.density[9].values'] = linspace(0, 1, 10)

# save updated ODS to our personal IMAS database
save_omas_imas(ods, machine='ITER', pulse=102292, run=11, new=True, imas_version=omas_rcparams['default_imas_version'])

# re-load data from our personal IMAS database
ods1 = load_omas_imas(machine='ITER', pulse=102292, run=11, imas_version=omas_rcparams['default_imas_version'])

# convince ourselves that the data has indeed been written and read back
print(ods1['edge_profiles.ggd[0].electrons.density[9].values'])
