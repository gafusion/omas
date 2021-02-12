#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMFIT classes and plotting
==========================
How to generate ODSs from OMFITclasses, and use OMAS plot methods
"""

from matplotlib.pyplot import show

from omfit_classes.omfit_eqdsk import OMFITgeqdsk, OMFITsrc
from omfit_classes.omfit_gacode import OMFITgacode
from omas import *

# =====================
# plotting equilibrium
# =====================
# read gEQDSK file in OMFIT
geq = OMFITgeqdsk(OMFITsrc + '/../samples/g133221.01000')
geq['fluxSurfaces'].load()

# convert gEQDSK to OMAS data structure
ods = geq.to_omas()

# omas plots using functional approach
omas_plot.equilibrium_summary(ods, linewidth=1)

# omas plots using object-oriented approach
# ods.plot_equilibrium_summary(linewidth=1)

# =====================
# plotting profiles
# =====================
# read input.profiles file in OMFIT
ip = OMFITgacode(OMFITsrc + '/../samples/input.profiles')

# convert input.profiles to OMAS data structure
ods = ip.to_omas()

# omas plots using functional approach
# omas_plot.core_profiles_summary(ods)

# omas plots using object-oriented approach
ods.plot_core_profiles_summary()

show()
