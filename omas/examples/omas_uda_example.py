#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UDA Universal Data Access
=========================
This example illustrates how OMAS can load machines
experimental data stored in IMAS format via UDA
"""

from omas import *
from pprint import pprint

# MAST UDA server (restricted access)
server = 'idam0.mast.ccfe.ac.uk'
port = 56563

pulse = 30420

ods = load_omas_uda(server=server,
                    port=port,
                    pulse=pulse,
                    paths=['magnetics'])

pprint(ods.pretty_paths())
