from omfit.classes.omfit_eqdsk import OMFITgeqdsk, OMFITsrc
from omas import *
import os

# set OMAS debugging topic
os.environ['OMAS_DEBUG_TOPIC'] = 'imas'

# read gEQDSK file in OMFIT
eq = OMFITgeqdsk(OMFITsrc+'/../samples/g133221.01000')

# convert gEQDSK to OMAS data structure
ods = eq.to_omas()

# save OMAS data structure to IMAS
paths = save_omas_imas(ods, tokamak='DIII-D', new=True)

# load IMAS to OMAS data structure
ods1 = load_omas_imas(user, tokamak='DIII-D', paths=paths)

# read from OMAS data structure
eq1 = OMFITgeqdsk('g133221.02000').from_omas(ods1)

# save gEQDSK file
eq1.deploy()
