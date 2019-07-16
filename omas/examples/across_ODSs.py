#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Access data across multiple ODSs
================================
ODS with consistency_check=False can be used to traverse data across multiple ODSs
"""

from omas import *

master_ods = ODS(consistency_check=False)
for k in range(5):
    for j in range(3):
        ods = ODS(consistency_check=False)
        ods['data'] = k
        master_ods[k][j] = ods

print(master_ods[':.:.data'])
