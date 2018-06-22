#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IDS info
========
This is an example showing how to query OMAS for structural information about IMAS IDSs
"""

from __future__ import print_function, division, unicode_literals

from pprint import pprint
from omas import *

# test getting information about ids structures
infos = omas_info(['equilibrium', 'core_profiles'])

pprint(infos['equilibrium']['time_slice'][0]['global_quantities']['ip'])

###################################
# The above example prints the following::
#
#     {'data_type': 'FLT_0D',
#      'documentation': 'Plasma current. Positive sign means anti-clockwise when viewed from above.',
#      'full_path': 'equilibrium/time_slice(itime)/global_quantities/ip',
#      'type': 'dynamic',
#      'units': 'A'}
