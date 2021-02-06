import numpy as np
import inspect
from omas import *
from omas.omas_utils import printd
from omas.machine_mappings._common import *

__all__ = []


# Use the @machine_mapping_function to automatically fill out the .json mapping file
# All these functions must accept `ods` as their first argument
# All other arguments should have a default value defined, as this serves two purposes:
#  1. run the run_machine_mapping_functions
#  2. automatically fill the __options__ entry in the .json mapping file


@machine_mapping_function(__all__)
def MDS_gEQDSK_psi_nstx(ods, pulse=139047, EFIT_tree='EFIT01'):
    return MDS_gEQDSK_psi(ods, 'nstx-u', pulse, EFIT_tree)


# =====================
if __name__ == '__main__':
    ods = ODS()
    with ods.open('machine', 'nstx-u', 139047):
        print(ods['equilibrium.time_slice.:.profiles_1d.psi'])
    # run_machine_mapping_functions(['MDS_gEQDSK_psi_nstx'], globals(), locals())
