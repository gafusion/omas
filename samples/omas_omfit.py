#only works with python 2.7 since OMFIT is only Python 2.7 compatible
from __future__ import print_function, division, unicode_literals

from matplotlib.pyplot import show

import os
os.environ['OMAS_DEBUG_TOPIC'] = 'imas'

from omfit.classes.omfit_eqdsk import OMFITgeqdsk, OMFITsrc
from omfit.classes.omfit_gacode import OMFITgacode, OMFITsrc
from omas import *

#=====================
# plotting equilibrium
#=====================
# read gEQDSK file in OMFIT
geq = OMFITgeqdsk(OMFITsrc + '/../samples/g133221.01000')

# convert gEQDSK to OMAS data structure
ods = geq.to_omas()

omas_plot.equilibrium_summary(ods,linewidth=1)
#ods.plot_equilibrium_summary(linewidth=1)

#=====================
# plotting profiles
#=====================
# read input.profiles file in OMFIT
ip = OMFITgacode(OMFITsrc + '/../samples/input.profiles')

# convert input.profiles to OMAS data structure
ods = ip.to_omas()

#omas_plot.core_profiles_summary(ods)
ods.plot_core_profiles_summary()
show()
