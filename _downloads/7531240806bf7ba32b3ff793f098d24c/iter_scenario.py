#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
Work with ITER IMAS scenario database
=====================================

Please refer to the :ref:`ITER page: <iter>`.

"""

# load OMAS package
from omas import *

# load data from a pulse chosen from the ITER scenario database
ods = load_omas_iter_scenario(pulse=131034, run=0)

# print nodes with data
from pprint import pprint

pprint(ods.pretty_paths())

# save data in different format (eg. pickle file)
save_omas_pkl(ods, 'iter_scenario_131034.pk')
