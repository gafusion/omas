#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
Interface with GKDB
===================
Use OMAS to interface with Gyro-Kinetic DataBase (GKDB) https://gitlab.com/gkdb/gkdb
GKDB is a publicly accessible database of delta-f flux-tube gyro-kinetic simulations of tokamak plasmas
which stores its data according to the `gyrokinetic` IMAS IDS https://gafusion.github.io/omas/schema/schema_gyrokinetics.html
"""

from omas import ODS, imas_json_dir
from pprint import pprint
import tempfile
import sys

# load a sample GKDB sample json file
sample_filename = imas_json_dir + '/../samples/gkdb_linear_eigenvalue.json'
ods = ODS()
ods['gyrokinetics'].load(sample_filename)

# show content
pprint(ods.pretty_paths())

# save a copy
filename = tempfile.gettempdir() + '/gkdb_linear_initialvalue.json'
ods['gyrokinetics'].save(filename)

# load the newly saved copy
ods1 = ODS()
ods1['gyrokinetics'].load(filename)

# look for differences between original GKDB json and OMAS json
differences = ods.diff(ods1, ignore_type=True)
if not differences:
    print('\nPrint no differences found: save/load of GKDB json file worked\n')
else:
    pprint(differences)
    raise RuntimeError('Save/Load of GKDB on json file failed')

# raise error if trying to run GKDB under Python2x
try:
    if sys.version_info < (3, 0):
        raise ImportError('gkdb library is only Python 3 compatible')
    import gkdb.core.model
except ImportError as _excp:
    print('Could not import gkdb library: %s' % repr(_excp))
else:
    # Check that GKDB file written by OMAS is valid also according to GKDB
    if gkdb.core.ids_checks.check_json(filename, only_input=False):
        print('json file saved via OMAS is valid for gkdb')

    # This requires an account on the GKDB server
    if False:
        gkdb.core.model.connect_to_gkdb()
        gkdb.core.model.Ids_properties.from_json(filename)
