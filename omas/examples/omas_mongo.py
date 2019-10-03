from pprint import *
from omas import *

ods = ODS().sample_equilibrium()
ods['equilibrium.code.name'] = 'test_code'

# write entry to the database
_id = save_omas_mongo(ods, table='test')

# retrieve exact entry using `_id`
ods = ODS().load('mongo', {'_id': _id}, table='test')

# find entries that satisfy the query by matching strings
odss = load_omas_mongo({'equilibrium.code.name': 'test_code'}, table='test')

# navigate through dictionaries and arrays of structures
# https://docs.mongodb.com/manual/tutorial/query-embedded-documents/
odss = load_omas_mongo({'equilibrium.time_slice.0.global_quantities.ip': {'$gt': 0}}, table='test')

# find entries based on conditions on array elements
# https://docs.mongodb.com/manual/tutorial/query-arrays/
odss = load_omas_mongo({'equilibrium.vacuum_toroidal_field.b0': {'$size': 1}}, table='test')
