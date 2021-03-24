#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extend IMAS data dictionary
===========================
This example extends the IMAS data dictionary with user defined structures

"""

from omas import ODS
from omas.omas_structure import add_extra_structures

# OMAS extra_structures
_extra_structures = {
    'core_profiles': {
        "core_profiles.profiles_1d[:].does_not_exist": {
            "coordinates": ["core_profiles.profiles_1d[:].grid.rho_tor_norm"],
            "documentation": "testing adding entries to IMAS data dictionary",
            "data_type": "FLT_1D",
            "units": "rad/s",
            "cocos_signal": "TOR",
        }
    }
}


ods = ODS()

try:
    ods['core_profiles.profiles_1d[0].does_not_exist'] = [1.0]
except LookupError:
    pass
else:
    raise RuntimeError('core_profiles.profiles_1d[0].does_not_exist should not be a valid IMAS location')

add_extra_structures(_extra_structures)
ods['core_profiles.profiles_1d[:].does_not_exist'] = [1.0]
