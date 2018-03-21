from __future__ import print_function, division, unicode_literals

import os
os.environ['OMAS_DEBUG_TOPIC'] = '*'

from omas import *

# load some sample data
ods_start = ods_sample()

# save/load Python pickle
filename = 'test.pkl'
save_omas_pkl(ods_start, filename)
ods = load_omas_pkl(filename)

# save/load ASCII Json
filename = 'test.json'
save_omas_json(ods, filename)
ods = load_omas_json(filename)

# save/load NetCDF
filename = 'test.nc'
save_omas_nc(ods, filename)
ods = load_omas_nc(filename)

# remote save/load S3
filename = 'test.s3'
save_omas_s3(ods, filename)
ods = load_omas_s3(filename)

# save/load IMAS
paths = save_omas_imas(ods, tokamak='ITER', shot=1, new=True)
ods1 = load_omas_imas(tokamak='ITER', shot=1, paths=paths)

# check data
check = different_ods(ods, ods1)
if not check:
    print('OMAS data got saved and loaded correctly throughout')
else:
    print(check)
