#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample OMAS data
================
OMAS provides a way to populate ODSs with sample data.
This is often useful for testing/debugging purposes.
"""

from __future__ import print_function, division, unicode_literals

import os

os.environ['OMAS_DEBUG_TOPIC'] = '*'

from omas import *

# list functions that fill ODS with sample data
ods_sample_methods = []
for item in dir(ODS):
    if item.startswith('sample_'):
        ods_sample_methods.append(item)
print("ods_sample_methods = %s" % ods_sample_methods)

ods = ODS()

# add sample equilibrium data to an ODS
ods.sample_equilibrium()

# alternatively, ods_sample() generates a sample ODS that is populated by all `ODS.sample_` methods
ods = ods_sample()
