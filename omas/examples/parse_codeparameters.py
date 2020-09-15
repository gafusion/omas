#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMAS handling of code parameters
==================

This example shows how XML code.parameters are handled withing OMAS

NOTE: OMAS will save code.parameters as XML only when saving to IMAS
All other save methods retain the code.parameters tree structure
"""

from pprint import pprint
from omas import *
import os

# Assign code parameters using dictionary approach
ods = ODS()
ods['equilibrium.code.parameters'] = CodeParameters()
ods['equilibrium.code.parameters']['test'] = {}
ods['equilibrium.code.parameters']['test']['parameter1'] = 1
ods['equilibrium.code.parameters']['test']['parameter2'] = 'hello'
with omas_environment(ods, xmlcodeparams=True):
    pprint(ods)

# Load code parameters from XML file
ods = ODS()
ods['equilibrium.time'] = [0.0]
ods['equilibrium.code.parameters'] = CodeParameters(imas_json_dir + '/../samples/input_gray.xml')

# In OMAS code.parameters are represented as dicts
pprint(ods)
# the XML representation can be set with an environment
with omas_environment(ods, xmlcodeparams=True):
    pprint(ods)
pprint(ods)

# The IMAS save is the only one that forces saving code.parameters as XML
# All other save methods maintain the dictionary structure
omas_rcparams['allow_fake_imas_fallback'] = True

# Save to IMAS
# code.parameters are saved as XML
print('=' * 20)
print(' Writing data to IMAS')
print('=' * 20)
save_omas_imas(ods, machine='ITER', pulse=1, new=True)

# Load from IMAS
# code.parameters are loaded as XML
print('=' * 20)
print(' Reading data from IMAS')
print('=' * 20)
ods1 = load_omas_imas(machine='ITER', pulse=1)

# code.parameters are dictionaries in the ODS
pprint(ods['equilibrium.code.parameters'])

# Handle code.parameters that are not in XML format as per IMAS specifications
ods = ODS()
ods['equilibrium.code.parameters'] = 'not in XML format'
with omas_environment(ods, xmlcodeparams=True):
    print(ods['equilibrium.code.parameters'])  # This will not raise an error
