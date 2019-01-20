#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
Work with ITER IMAS scenario database
=====================================
This example loads an ITER scenarios from the ITER IMAS scenario database

To browse what is in the ITER scenario database::

    >> module load IMAS
    >> pip install --user --upgrade pyyaml # (this needs to be done only once)
    >> scenario_summary

The script is meant to be run on the ITER workstations,
since it requires access to the `scenario_summary` utility,
as well as the data that is stored in the ITER IMAS database.
"""

from __future__ import print_function, division, unicode_literals

from omas import *

# load data from a shot chosen from the ITER scenario database
ods = load_omas_iter_scenario(shot=131034, run=0)

# print nodes with data
from pprint import pprint
pprint(ods.pretty_paths())

# save data in different format (eg. pickle file)
save_omas_pkl(ods, 'iter_scenario_131034.pk')
