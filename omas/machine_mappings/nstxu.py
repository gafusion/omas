import numpy as np
import inspect
from omas import *
from omas.omas_utils import printd
from omas.machine_mappings._common import *

__all__ = []


@machine_mapping_function(__all__)
def pf_active_hardware(ods):
    r"""
    Loads NSTX-U tokamak poloidal field coil hardware geometry

    :param ods: ODS instance

    :return: dict
        Information or instructions for follow up in central hardware description setup
    """
    # R        Z       dR      dZ    tilt1  tilt2
    # 0 in the last column really means 90 degrees
    # fmt: off
    fc_dat = np.array(
        [[0.324600011, 1.59060001, 0.0625, 0.463400006, 0.0, 0.0],
         [0.400299996, 1.80420005, 0.0337999985, 0.181400001, 0.0, 0.0],
         [0.550400019, 1.81560004, 0.0375000015, 0.1664, 0.0, 0.0],
         [0.799170017, 1.85264003, 0.162711993, 0.0679700002, 0.0, 0.0],
         [0.799170017, 1.93350995, 0.162711993, 0.0679700002, 0.0, 0.0],
         [1.49445999, 1.55263996, 0.186435997, 0.0679700002, 0.0, 0.0],
         [1.49445999, 1.63350999, 0.186435997, 0.0679700002, 0.0, 0.0],
         [1.80649996, 0.888100028, 0.115264997, 0.0679700002, 0.0, 0.0],
         [1.79460001, 0.807200015, 0.0915419981, 0.0679700002, 0.0, 0.0],
         [2.01180005, 0.648899972, 0.135900006, 0.0684999973, 0.0, 0.0],
         [2.01180005, 0.575100005, 0.135900006, 0.0684999973, 0.0, 0.0],
         [2.01180005, -0.648899972, 0.135900006, 0.0684999973, 0.0, 0.0],
         [2.01180005, -0.575100005, 0.135900006, 0.0684999973, 0.0, 0.0],
         [1.80649996, -0.888100028, 0.115264997, 0.0679700002, 0.0, 0.0],
         [1.79460001, -0.807200015, 0.0915419981, 0.0679700002, 0.0, 0.0],
         [1.49445999, -1.55263996, 0.186435997, 0.0679700002, 0.0, 0.0],
         [1.49445999, -1.63350999, 0.186435997, 0.0679700002, 0.0, 0.0],
         [0.799170017, -1.85264003, 0.162711993, 0.0679700002, 0.0, 0.0],
         [0.799170017, -1.93350995, 0.162711993, 0.0679700002, 0.0, 0.0],
         [0.550400019, -1.82959998, 0.0375000015, 0.1664, 0.0, 0.0],
         [0.400299996, -1.80420005, 0.0337999985, 0.181400001, 0.0, 0.0],
         [0.324600011, -1.59060001, 0.0625, 0.463400006, 0.0, 0.0]]
    )

    names = {'PF1AU':'PF1AU', 'PF1BU':'PF1BU', 'PF1CU':'PF1CU',
             'PF2U1':'PF2U', 'PF2U2':'PF2U',
             'PF3U1':'PF3U', 'PF3U2':'PF3U',
             'PF4U1':'PF4', 'PF4U2':'PF4',
             'PF5U1':'PF5', 'PF5U2':'PF5',
             'PF5L1':'PF5', 'PF5L2':'PF5',
             'PF4L1':'PF4', 'PF4L2':'PF4',
             'PF3L1':'PF3L', 'PF3L2':'PF3L',
             'PF2L1':'PF2L', 'PF2L2':'PF2L',
             'PF1CL':'PF1CL', 'PF1BL':'PF1BL', 'PF1AL':'PF1AL'}
    # fmt: on

    ods = pf_coils_to_ods(ods, fc_dat)

    for i, (name, fcid) in enumerate(names.items()):
        ods['pf_active.coil'][i]['name'] = ods['pf_active.coil'][i]['identifier'] = name
        ods['pf_active.coil'][i]['element.0.identifier'] = fcid


@machine_mapping_function(__all__)
def pf_active_coil_current_data(ods, pulse=203679):
    ods1 = ODS()
    inspect.unwrap(pf_active_hardware)(ods1)
    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels='pf_active.coil',
            identifier='pf_active.coil.{channel}.element.0.identifier',
            time='pf_active.coil.{channel}.current.time',
            data='pf_active.coil.{channel}.current.data',
            validity=None,
            mds_server='nstxu',
            mds_tree='ENGINEERING',
            tdi_expression='\\ENGINEERING::TOP.ANALYSIS.I{signal}',
            time_norm=1.0,
            data_norm=1.0,
        )


@machine_mapping_function(__all__)
def MDS_gEQDSK_psi_nstx(ods, pulse=139047, EFIT_tree='EFIT01'):
    return MDS_gEQDSK_psi(ods, 'nstxu', pulse, EFIT_tree)


@machine_mapping_function(__all__)
def MDS_gEQDSK_bbbs_nstx(ods, pulse=139047, EFIT_tree='EFIT01'):
    TDIs = {
        'r': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS',
        'z': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.ZBBBS',
        'n': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.NBBBS',
    }
    res = mdsvalue('nstxu', pulse=pulse, treename=EFIT_tree, TDI=TDIs).raw()
    res['n'] = res['n'].astype(int)
    for k in range(len(res['n'])):
        ods[f'equilibrium.time_slice.{k}.boundary.outline.r'] = res['r'][k, : res['n'][k]]
        ods[f'equilibrium.time_slice.{k}.boundary.outline.z'] = res['z'][k, : res['n'][k]]


# =====================
if __name__ == '__main__':
    run_machine_mapping_functions(__all__, globals(), locals())
