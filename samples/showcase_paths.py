from __future__ import print_function, division, unicode_literals
import numpy
from omas import *

ods = omas()

# access data as dictionary
ods['equilibrium']['time_slice'][0]['time'] = 1000.
print(ods['equilibrium']['time_slice'][0]['time'])

# access data as string
ods['equilibrium.time_slice.1.time'] = 2000.
print(ods['equilibrium.time_slice.1.time'])

# access data as string (square brackets for arrays of structures)
ods['equilibrium.time_slice[2].time'] = 3000.
print(ods['equilibrium.time_slice[2].time'])

# access data with path list
ods[['equilibrium', 'time_slice', 3, 'time']] = 4000.
print(ods[['equilibrium', 'time_slice', 3, 'time']])

# access data with mix and match approach
ods['equilibrium]']['time_slice.4.time'] = 5000.
print(ods['equilibrium.time_slice]']['4.time'])

# classic ways to access data across an array of structures
data = []
for k in ods['equilibrium.time_slice'].keys():
    data.append(ods['equilibrium.time_slice'][k]['time'])
print(numpy.array(data))

# access data across an array of structures via data slicing
print(ods['equilibrium.time_slice.:.time'])
