from __future__ import print_function, division, unicode_literals

from omas import *

print( omas_scenario_database() )

# 130010 will take a while to download the first time
ods = omas_scenario_database(machine='ITER', shot=130010, run=1)
print( ods['core_profiles']['profiles_1d'][100]['electrons']['pressure'] )