from __future__ import print_function, division, unicode_literals
import numpy

from omas import *

ods = ODS()

# access data as dictionary
ods['equilibrium']['time_slice'][0]['time'] = 1000.
assert (ods['equilibrium']['time_slice'][0]['time'] == 1000.)

# access data as string
ods['equilibrium.time_slice.1.time'] = 2000.
assert (ods['equilibrium.time_slice.1.time'] == 2000.)

# access data as string (square brackets for arrays of structures)
ods['equilibrium.time_slice[2].time'] = 3000.
assert (ods['equilibrium.time_slice[2].time'] == 3000.)

# access data with path list
ods[['equilibrium', 'time_slice', 3, 'time']] = 4000.
assert (ods[['equilibrium', 'time_slice', 3, 'time']] == 4000.)

# access data with mix and match approach
ods['equilibrium]']['time_slice.4.time'] = 5000.
assert (ods['equilibrium]']['time_slice.4.time'] == 5000.)

# classic ways to access data across an array of structures
data = []
for k in ods['equilibrium.time_slice'].keys():
    data.append(ods['equilibrium.time_slice'][k]['time'])
data = numpy.array(data)
assert (numpy.all(data == numpy.array([1000., 2000., 3000., 4000., 5000.])))

# access data across an array of structures via data slicing
data = ods['equilibrium.time_slice.:.time']
assert (numpy.all(data == numpy.array([1000., 2000., 3000., 4000., 5000.])))

# use of .prune() to clean empty branches left behind from dynamic path creation
ods = ODS()
try:
    ods['equilibrium']['time_slice'][0]['global_quantities']['asdasd'] = 6
except LookupError:
    n = ods.prune()
    assert (n == 4)
    assert (len(ods) == 0)

# single string access does not leave empty branches
ods = ODS()
try:
    ods['equilibrium.time_slice.0.global_quantities.asdasd'] = 6
except LookupError:
    assert (len(ods) == 0)

# examples using .setdefault()
ods = ODS()
ods['equilibrium.time_slice.0.global_quantities.ip'] = 6
ods.setdefault('equilibrium.time_slice.0.global_quantities.ip', 5)
assert (ods['equilibrium.time_slice.0.global_quantities.ip'] == 6)
ods = ODS()
ods.setdefault('equilibrium.time_slice.0.global_quantities.ip', 5)
assert (ods['equilibrium.time_slice.0.global_quantities.ip'] == 5)

# example using .get()
ods = ODS()
ods.get('equilibrium.time_slice.0.global_quantities.ip', 5)
assert ('equilibrium.time_slice.0.global_quantities.ip' not in ods)
