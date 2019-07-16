#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Access data across multiple ODSs
================================
ODS with consistency_check=False can be used to traverse data across multiple ODSs
"""

from omas import *

# define a ods to collect multiple ODSs
master_ods = ODS(consistency_check=False)

# populate master_ods with a 5x3 grid of ODSs each having their data
for k in range(5):
    for j in range(3):
        ods = ODS(consistency_check=False)
        ods['data'] = k
        master_ods[k][j] = ods

# print data aggregating across ODSs
print(master_ods[':.:.data'])

###################################
# This will return::
#
#     [[0 0 0]
#      [1 1 1]
#      [2 2 2]
#      [3 3 3]
#      [4 4 4]]
