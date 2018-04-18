from __future__ import print_function, division, unicode_literals

from omas import *

# test getting information about ids structures
infos=omas_info(['equilibrium','core_profiles'])

print(infos['equilibrium']['time_slice'][0]['global_quantities']['ip'])