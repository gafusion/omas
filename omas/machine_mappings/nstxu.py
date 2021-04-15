import os
import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd
from omas.machine_mappings._common import *

# NOTES:
# List of MDS+ signals
# https://nstx.pppl.gov/nstx/Software/FAQ/signallabels.html

__all__ = []
__regression_arguments__ = {'__all__': __all__}


@machine_mapping_function(__regression_arguments__)
def pf_active_hardware(ods):
    r"""
    Loads NSTX-U tokamak poloidal field coil hardware geometry

    :param ods: ODS instance
    """
    from omfit_classes.omfit_efund import OMFITmhdin

    mhdin_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'mhdin.dat'])
    mhdin = OMFITmhdin(mhdin_dat_filename)
    mhdin.to_omas(ods, update='pf_active')

    names = [
        ['PF1AU', 'PF1AU'],
        ['PF1BU', 'PF1BU'],
        ['PF1CU', 'PF1CU'],
        ['PF2U1', 'PF2U'],
        ['PF2U2', 'PF2U'],
        ['PF3U1', 'PF3U'],
        ['PF3U2', 'PF3U'],
        ['PF4U1', 'PF4'],
        ['PF4U2', 'PF4'],
        ['PF5U1', 'PF5'],
        ['PF5U2', 'PF5'],
        ['PF5L1', 'PF5'],
        ['PF5L2', 'PF5'],
        ['PF4L1', 'PF4'],
        ['PF4L2', 'PF4'],
        ['PF3L1', 'PF3L'],
        ['PF3L2', 'PF3L'],
        ['PF2L1', 'PF2L'],
        ['PF2L2', 'PF2L'],
        ['PF1CL', 'PF1CL'],
        ['PF1BL', 'PF1BL'],
        ['PF1AL', 'PF1AL'],
    ]

    k = 0
    for c in ods[f'pf_active.coil']:
        for e in ods[f'pf_active.coil'][c]['element']:
            if k < len(names):
                ename, cname = names[k]
                ods[f'pf_active.coil'][c]['name'] = cname
                ods[f'pf_active.coil'][c]['identifier'] = cname
                ods[f'pf_active.coil'][c]['element'][e]['name'] = ename
                ods[f'pf_active.coil'][c]['element'][e]['identifier'] = ename
            k += 1


@machine_mapping_function(__regression_arguments__, pulse=204202)
def pf_active_coil_current_data(ods, pulse):
    ods1 = ODS()
    unwrap(pf_active_hardware)(ods1)
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
    # IMAS stores the current in the coil not multiplied by the number of turns
    for channel in ods1['pf_active.coil']:
        ods[f'pf_active.coil.{channel}.current.data'] /= ods1[f'pf_active.coil.{channel}.element.0.turns_with_sign']


@machine_mapping_function(__regression_arguments__)
def magnetics_hardware(ods):
    r"""
    Load NSTX-U tokamak flux loops and magnetic probes hardware geometry

    :param ods: ODS instance
    """
    # magnetics signals from
    #  OMFITnstxMHD('/p/spitfire/s1/common/plasma/phoenix/cdata/signals_020916_PF4.dat' ,serverPicker='portal')
    #  OMFITnstxMHD('/p/spitfire/s1/common/Greens/NSTX/Jan2015/01152015Av1.0/diagSpec01152015.dat' ,serverPicker='portal')

    from omfit_classes.omfit_efund import OMFITmhdin

    mhdin_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'mhdin.dat'])
    mhdin = OMFITmhdin(mhdin_dat_filename)
    mhdin.to_omas(ods, update='magnetics')

    for k in ods[f'magnetics.flux_loop']:
        ods[f'magnetics.flux_loop.{k}.identifier'] = 'F_' + ods[f'magnetics.flux_loop.{k}.identifier']

    for k in ods[f'magnetics.b_field_pol_probe']:
        ods[f'magnetics.b_field_pol_probe.{k}.identifier'] = 'B_' + ods[f'magnetics.b_field_pol_probe.{k}.identifier']


@machine_mapping_function(__regression_arguments__, pulse=204202)
def magnetics_floops_data(ods, pulse):
    ods1 = ODS()
    unwrap(magnetics_hardware)(ods1)
    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels='magnetics.flux_loop',
            identifier='magnetics.flux_loop.{channel}.identifier',
            time='magnetics.flux_loop.{channel}.flux.time',
            data='magnetics.flux_loop.{channel}.flux.data',
            validity='magnetics.flux_loop.{channel}.flux.validity',
            mds_server='nstxu',
            mds_tree='NSTX',
            tdi_expression='\\{signal}',
            time_norm=1.0,
            data_norm=1.0 / 2 / np.pi,
        )


@machine_mapping_function(__regression_arguments__, pulse=204202)
def magnetics_probes_data(ods, pulse):
    ods1 = ODS()
    unwrap(magnetics_hardware)(ods1)
    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels='magnetics.b_field_pol_probe',
            identifier='magnetics.b_field_pol_probe.{channel}.identifier',
            time='magnetics.b_field_pol_probe.{channel}.field.time',
            data='magnetics.b_field_pol_probe.{channel}.field.data',
            validity='magnetics.b_field_pol_probe.{channel}.field.validity',
            mds_server='nstxu',
            mds_tree='NSTX',
            tdi_expression='\\{signal}',
            time_norm=1.0,
            data_norm=1.0,
        )


@machine_mapping_function(__regression_arguments__, pulse=204202)
def MDS_gEQDSK_psi_nstx(ods, pulse, EFIT_tree='EFIT01'):
    return MDS_gEQDSK_psi(ods, 'nstxu', pulse, EFIT_tree)


@machine_mapping_function(__regression_arguments__, pulse=204202)
def MDS_gEQDSK_bbbs_nstx(ods, pulse, EFIT_tree='EFIT01'):
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
    test_machine_mapping_functions(__all__, globals(), locals())
