import numpy as np
import inspect
from omas import *
from omas.omas_utils import printd
from omas.machine_mappings._common import *

__all__ = []


@machine_mapping_function(__all__)
def MDS_gEQDSK_psi_nstx(ods, pulse=139047, EFIT_tree='EFIT01'):
    return MDS_gEQDSK_psi(ods, 'nstx-u', pulse, EFIT_tree)


@machine_mapping_function(__all__)
def MDS_gEQDSK_bbbs_nstx(ods, pulse=139047, EFIT_tree='EFIT01'):
    TDIs = {
        'r': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS',
        'z': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.ZBBBS',
        'n': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.NBBBS',
    }
    res = mdsvalue('nstx-u', pulse=pulse, treename=EFIT_tree, TDI=TDIs).raw()
    res['n'] = res['n'].astype(int)
    for k in range(len(res['n'])):
        ods[f'equilibrium.time_slice.{k}.boundary.outline.r'] = res['r'][k, : res['n'][k]]
        ods[f'equilibrium.time_slice.{k}.boundary.outline.z'] = res['z'][k, : res['n'][k]]


# =====================
if __name__ == '__main__':
    run_machine_mapping_functions(__all__, globals(), locals())
