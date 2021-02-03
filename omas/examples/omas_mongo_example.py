#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
MongoDB storage and discovery
=============================
OMAS can store/load its data in a MongoDB server.
MongoDB is a document-oriented database program that uses JSON-like documents with schema
and supports field, range query, and regular expression searches.

As an example we illustrate storage of GKDB data in this format.
"""

from pprint import *
from omas import *
from random import random

ods = ODS().sample_equilibrium()
ods['equilibrium.time_slice.0.global_quantities.ip'] *= 0.9 + random() * 0.2
ods['equilibrium.code.name'] = 'test_code'

print('write entry to the database')
_id = save_omas_mongo(ods, collection='test', database='test')

print('retrieve exact entry using `_id`')
ods = ODS().load('mongo', {'_id': _id}, collection='test', database='test')

print('find at most 5 entries that satisfy the query by matching strings')
odss = load_omas_mongo({'equilibrium.code.name': 'test_code'}, collection='test', database='test', limit=5)
print(f' - found {len(odss)} entries')

print('find at most 5 entries based on scalar condition')
# https://docs.mongodb.com/manual/tutorial/query-embedded-documents/
odss = load_omas_mongo({'equilibrium.time_slice.0.global_quantities.ip': {'$gt': 0}}, collection='test', database='test', limit=5)
print(f' - found {len(odss)} entries')

print('find at most 5 entries based on conditions on array elements')
# https://docs.mongodb.com/manual/tutorial/query-arrays/
odss = load_omas_mongo({'equilibrium.vacuum_toroidal_field.b0': {'$size': 1}}, collection='test', database='test', limit=5)
print(f' - found {len(odss)} entries')

# =============================================
# showcase use of MongoDB storage for GKDB data
# =============================================
print('load a sample GKDB sample json file')
sample_filename = imas_json_dir + '/../samples/gkdb_linear_eigenvalue.json'
ods = ODS()
# warn about `gyrokinetics.fluxes_integrated_norm = []` and drop it
ods['gyrokinetics'].load(sample_filename, consistency_check='warn_drop')

print('write GKDB entry to the database')
_id = ods.save('mongo', collection='gkdb', database='test')

print('reload GKDB entry')
ods1 = ODS()
ods1.load('mongo', {'_id': _id}, collection='gkdb', database='test')

print('look for differences between original GKDB json and MongoDB entry')
differences = ods.diff(ods1, ignore_type=True)
if not differences:
    print('\nPrint no differences found: save/load of GKDB on MongoDB worked\n')
else:
    pprint(differences)
    raise RuntimeError('Save/Load of GKDB  on MongoDB failed')
