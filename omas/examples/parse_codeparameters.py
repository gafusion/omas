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
    print(ods['equilibrium.code.parameters'])
    assert isinstance(ods['equilibrium.code.parameters'], str) and '<?xml' in ods['equilibrium.code.parameters']

# Load code parameters from XML file
ods = ODS()
ods['equilibrium.time'] = [0.0]
ods['equilibrium.code.parameters'] = CodeParameters(imas_json_dir + '/../samples/input_gray.xml')

# In OMAS code.parameters are represented as dicts
print(ods['equilibrium.code.parameters'])
assert isinstance(ods['equilibrium.code.parameters'], dict)

# the XML representation can be set with an environment
with omas_environment(ods, xmlcodeparams=True):
    print(ods['equilibrium.code.parameters'])
    assert isinstance(ods['equilibrium.code.parameters'], str) and '<?xml' in ods['equilibrium.code.parameters']

# allow fallback on fake IMAS environment in OMAS in case real IMAS installation is not present
with fakeimas.fake_environment('fallback'):
    # Save to IMAS
    # code.parameters are loaded from OMAS to IMAS XML
    print('=' * 20)
    print(' Writing data to IMAS')
    print('=' * 20)
    ods.save('imas', machine='ITER', pulse=2, new=True)

    # Load from IMAS
    # code.parameters are loaded from IMAS XML to OMAS dict
    print('=' * 20)
    print(' Reading data from IMAS')
    print('=' * 20)
    ods1 = ODS().load('imas', machine='ITER', pulse=2)
    assert isinstance(ods1['equilibrium.code.parameters'], dict)

# code.parameters are dictionaries in the ODS
print(ods['equilibrium.code.parameters'])

# Handle code.parameters that are not in XML format as per IMAS specifications
ods = ODS()
ods['equilibrium.code.parameters'] = 'not in XML format'
with omas_environment(ods, xmlcodeparams=True):  # This will print a warning
    assert ods['equilibrium.code.parameters'] == 'not in XML format'
