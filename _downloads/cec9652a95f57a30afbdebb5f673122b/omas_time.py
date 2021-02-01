#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time in OMAS
============
This example illustrates how OMAS can handle the time dimension
"""

from omas import *

# test generation of a sample ods
ods = ODS()
ods['equilibrium.time_slice'][0]['time'] = 100
ods['equilibrium.time_slice.0.global_quantities.ip'] = 0.0
ods['equilibrium.time_slice'][1]['time'] = 200
ods['equilibrium.time_slice.1.global_quantities.ip'] = 1.0
ods['equilibrium.time_slice'][2]['time'] = 300
ods['equilibrium.time_slice.1.global_quantities.ip'] = 2.0

# get time information from children
extra_info = {}
print(ods.time('equilibrium', extra_info=extra_info))
# >>  [100 200 300]
print(extra_info)
# >>  {'location': 'equilibrium.time_slice.:.time', 'homogeneous_time': True}

# show use of .homogeneous_time property
print(ods.homogeneous_time('equilibrium'))
# >>  True

# time arrays can be set using `set_time_array` function
# this simplifies the logic in the code since one does not
# have to check if the array was already there or not
ods.set_time_array('equilibrium.time', 0, 101)
ods.set_time_array('equilibrium.time', 1, 201)
ods.set_time_array('equilibrium.time', 2, 302)

# the make the timeslices consistent
ods['equilibrium.time_slice'][0]['time'] = 101
ods['equilibrium.time_slice'][1]['time'] = 201
ods['equilibrium.time_slice'][2]['time'] = 302

# get time information from explicitly set time array
extra_info = {}
print(ods.time('equilibrium', extra_info=extra_info))
# >>  [101 201 301]
print(extra_info)
# >>  {'location': '.time', 'homogeneous_time': True}

# get time value from item in array of structures
extra_info = {}
print(ods['equilibrium.time_slice'][0].time(extra_info=extra_info))
# >>  101
print(extra_info)
# >>  {'location': 'equilibrium.time_slice.0.time', 'homogeneous_time': True}

# get time array from array of structures
extra_info = {}
print(ods['equilibrium.time_slice'].time(extra_info=extra_info))
# >>  [101 201 302]
print(extra_info)
# >>  {'location': 'equilibrium.time_slice.:.time', 'homogeneous_time': True}

# get time from parent
extra_info = {}
print(ods.time('equilibrium.time_slice.0.global_quantities.ip', extra_info=extra_info))
# >>  101
print(extra_info)
# >>  {'location': '.time', 'homogeneous_time': None}

# slice whole ODS at given time
ods1 = ods['equilibrium'].slice_at_time(101)
print(ods.time('equilibrium'))
# >>  [101]
