#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IDS info
========
This is an example showing how to query OMAS for structural information about IMAS IDSs
"""

from pprint import pprint
from omas import *

# Get information about whole IDS structures
infos = omas_info(['equilibrium', 'core_profiles'])
pprint(infos['equilibrium']['time_slice'][0]['global_quantities']['ip'])

###################################
# The above example prints the following::
#
#     {'cocos_label_transformation': 'ip_like',
#      'cocos_leaf_name_aos_indices': 'equilibrium.time_slice{i}.global_quantities.ip',
#      'cocos_transformation_expression': '.sigma_ip_eff',
#      'data_type': 'FLT_0D',
#      'documentation': 'Plasma current. Positive sign means anti-clockwise when '
#                       'viewed from above.',
#      'full_path': 'equilibrium/time_slice(itime)/global_quantities/ip',
#      'lifecycle_status': 'active',
#      'type': 'dynamic',
#      'units': 'A'}

# Get information about a single node
info_single_node = omas_info_node('equilibrium.time_slice.:.global_quantities.ip')
pprint(info_single_node)

# Generate the RST documentation for an existing ODS
docs = ODS().sample_equilibrium().document()
print(docs)
