import os
import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd, printe, unumpy
from omas.machine_mappings._common import *

# NOTES:
# List of MDS+ signals
# https://nstx.pppl.gov/nstx/Software/FAQ/signallabels.html
# magnetics:
# https://nstx.pppl.gov/DragNDrop/Operations/Physics_Operations_Course/11%20OpsCourse_EquilibriumMagnetics_Rev1.pdf

__all__ = []
__regression_arguments__ = {'__all__': __all__}


@machine_mapping_function(__regression_arguments__)
def pf_active_hardware(ods):
    r"""
    Loads NSTX-U tokamak poloidal field coil hardware geometry

    :param ods: ODS instance
    """
    from omfit_classes.omfit_efund import OMFITmhdin, OMFITnstxMHD

    mhdin_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'mhdin.dat'])
    mhdin = get_support_file(OMFITmhdin, mhdin_dat_filename)
    mhdin.to_omas(ods, update='pf_active')

    signals_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'signals.dat'])
    signals = get_support_file(OMFITnstxMHD, signals_dat_filename)
    icoil_signals = signals['mappings']['icoil']

    for c in ods[f'pf_active.coil']:
        for e in ods[f'pf_active.coil'][c]['element']:
            cname = icoil_signals[c + 1]['name']
            cid = icoil_signals[c + 1]['mds_name_resolved'].strip('\\')
            ename = icoil_signals[c + 1]['mds_name_resolved'].strip('\\') + f'_element_{e}'
            eid = ename
            ods[f'pf_active.coil'][c]['name'] = cname
            ods[f'pf_active.coil'][c]['identifier'] = cid
            ods[f'pf_active.coil'][c]['element'][e]['name'] = ename
            ods[f'pf_active.coil'][c]['element'][e]['identifier'] = eid


@machine_mapping_function(__regression_arguments__, pulse=204202)
def pf_active_coil_current_data(ods, pulse):
    r"""
    Load NSTX-U tokamak pf_active coil current data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD

    ods1 = ODS()
    unwrap(pf_active_hardware)(ods1)
    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels=range(14),
            identifier='pf_active.coil.{channel}.identifier',
            time='pf_active.coil.{channel}.current.time',
            data='pf_active.coil.{channel}.current.data',
            validity=None,
            mds_server='nstxu',
            mds_tree='ENGINEERING',
            tdi_expression='\\{signal}',
            time_norm=1.0,
            data_norm=1.0,
        )

    with omas_environment(ods, cocosio=1):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels=range(14, 54),
            identifier='pf_active.coil.{channel}.identifier',
            time='pf_active.coil.{channel}.current.time',
            data='pf_active.coil.{channel}.current.data',
            validity=None,
            mds_server='nstxu',
            mds_tree='OPERATIONS',
            tdi_expression='\\{signal}',
            time_norm=1.0,
            data_norm=1.0,
        )

    # handle uncertainties
    signals_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'signals.dat'])
    signals = get_support_file(OMFITnstxMHD, signals_dat_filename)
    icoil_signals = signals['mappings']['icoil']
    for channel in ods1['pf_active.coil']:
        if f'pf_active.coil.{channel}.current.data' in ods:
            printd(f'Assigning uncertainty to pf_active.coil.{channel}', topic='machine')
            error = ods[f'pf_active.coil.{channel}.current.data'] * icoil_signals[channel + 1]['rel_error']
            error[error < icoil_signals[channel + 1]['abs_error']] = icoil_signals[channel + 1]['abs_error']
            ods[f'pf_active.coil.{channel}.current.data'] = unumpy.uarray(ods[f'pf_active.coil.{channel}.current.data'], error)

    # IMAS stores the current in the coil not multiplied by the number of turns
    for channel in ods1['pf_active.coil']:
        if f'pf_active.coil.{channel}.current.data' in ods:
            ods[f'pf_active.coil.{channel}.current.data'] /= ods1[f'pf_active.coil.{channel}.element.0.turns_with_sign']
        else:
            print(f'WARNING: pf_active.coil[{channel}].current.data is missing')


@machine_mapping_function(__regression_arguments__)
def magnetics_hardware(ods):
    r"""
    Load NSTX-U tokamak flux loops and magnetic probes hardware geometry

    :param ods: ODS instance
    """
    # magnetics signals from
    #  OMFITnstxMHD('/p/spitfire/s1/common/plasma/phoenix/cdata/signals_020916_PF4.dat' ,serverPicker='portal')
    #  OMFITnstxMHD('/p/spitfire/s1/common/Greens/NSTX/Jan2015/01152015Av1.0/diagSpec01152015.dat' ,serverPicker='portal')

    from omfit_classes.omfit_efund import OMFITmhdin, OMFITnstxMHD

    mhdin_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'mhdin.dat'])
    mhdin = get_support_file(OMFITmhdin, mhdin_dat_filename)
    mhdin.to_omas(ods, update='magnetics')

    signals_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'signals.dat'])
    signals = get_support_file(OMFITnstxMHD, signals_dat_filename)

    for k in ods[f'magnetics.flux_loop']:
        ods[f'magnetics.flux_loop.{k}.identifier'] = signals['mappings']['tfl'][k + 1]['mds_name'].strip('\\')

    for k in ods[f'magnetics.b_field_pol_probe']:
        ods[f'magnetics.b_field_pol_probe.{k}.identifier'] = signals['mappings']['bmc'][k + 1]['mds_name'].strip('\\')


@machine_mapping_function(__regression_arguments__, pulse=204202)
def magnetics_floops_data(ods, pulse):
    r"""
    Load NSTX-U tokamak flux loops flux data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD

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
            mds_tree='OPERATIONS',
            tdi_expression='\\{signal}',
            time_norm=1.0,
            data_norm=1.0 / 2 / np.pi,
        )

    # handle uncertainties
    signals_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'signals.dat'])
    signals = get_support_file(OMFITnstxMHD, signals_dat_filename)
    tfl_signals = signals['mappings']['tfl']
    for channel in ods1['magnetics.flux_loop']:
        if f'magnetics.flux_loop.{channel}.flux.data' in ods:
            printd(f'Assigning uncertainty to magnetics.flux_loop.{channel}', topic='machine')
            error = ods[f'magnetics.flux_loop.{channel}.flux.data'] * tfl_signals[channel + 1]['rel_error']
            error[error < tfl_signals[channel + 1]['abs_error']] = tfl_signals[channel + 1]['abs_error']
            ods[f'magnetics.flux_loop.{channel}.flux.data'] = unumpy.uarray(ods[f'magnetics.flux_loop.{channel}.flux.data'], error)


@machine_mapping_function(__regression_arguments__, pulse=204202)
def magnetics_probes_data(ods, pulse):
    r"""
    Load NSTX-U tokamak magnetic probes field data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD

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
            mds_tree='OPERATIONS',
            tdi_expression='\\{signal}',
            time_norm=1.0,
            data_norm=1.0,
        )

    # handle uncertainties
    signals_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'signals.dat'])
    signals = get_support_file(OMFITnstxMHD, signals_dat_filename)
    bmc_signals = signals['mappings']['bmc']
    for channel in ods1['magnetics.b_field_pol_probe']:
        if f'magnetics.b_field_pol_probe.{channel}.field.data' in ods:
            printd(f'Assigning uncertainty to magnetics.b_field_pol_probe.{channel}', topic='machine')
            error = ods[f'magnetics.b_field_pol_probe.{channel}.field.data'] * bmc_signals[channel + 1]['rel_error']
            error[error < bmc_signals[channel + 1]['abs_error']] = bmc_signals[channel + 1]['abs_error']
            ods[f'magnetics.b_field_pol_probe.{channel}.field.data'] = unumpy.uarray(
                ods[f'magnetics.b_field_pol_probe.{channel}.field.data'], error
            )


@machine_mapping_function(__regression_arguments__, pulse=204202)
def MDS_gEQDSK_psi_nstx(ods, pulse, EFIT_tree='EFIT01'):
    return MDS_gEQDSK_psi(ods, 'nstxu', pulse, EFIT_tree)


@machine_mapping_function(__regression_arguments__, pulse=204202)
def MDS_gEQDSK_bbbs_nstx(ods, pulse, EFIT_tree='EFIT01'):
    r"""
    Load NSTX-U EFIT boundary data

    :param ods: ODS instance

    :param pulse: shot number

    :param EFIT_tree: MDS+ EFIT tree
    """
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


@machine_mapping_function(__regression_arguments__, pulse=204202)
def ip_bt_dflux_data(ods, pulse):
    r"""
    Load NSTX-U tokamak Ip, Bt, and diamagnetic flux data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD

    signals_dat_filename = os.sep.join([omas_dir, 'machine_mappings', 'support_files', 'nstxu', 'signals.dat'])
    signals = get_support_file(OMFITnstxMHD, signals_dat_filename)

    mappings = {'PR': 'magnetics.ip.0', 'TF': 'tf.b_field_tor_vacuum_r', 'DL': 'magnetics.diamagnetic_flux.0'}

    TDIs = {}
    for item in ['PR', 'TF', 'DL']:
        TDIs[item + '_data'] = '\\' + signals[item][0]['mds_name'].strip('\\')
        TDIs[item + '_time'] = 'dim_of(\\' + signals[item][0]['mds_name'].strip('\\') + ')'
    res = mdsvalue('nstxu', pulse=pulse, treename='ENGINEERING', TDI=TDIs).raw()

    for item in ['PR', 'TF', 'DL']:
        if not isinstance(res[item + '_data'], Exception) and not isinstance(res[item + '_time'], Exception):
            ods[mappings[item] + '.data'] = res[item + '_data'] * signals[item][0]['scale']
            ods[mappings[item] + '.time'] = res[item + '_time']
        else:
            printe(f'No data for {mappings[item]}')
            ods[mappings[item] + '.data'] = []
            ods[mappings[item] + '.time'] = []

    # handle uncertainties
    for item in ['PR', 'TF', 'DL']:
        if mappings[item] + '.data' in ods:
            printd(f'Assigning uncertainty to {mappings[item]}', topic='machine')
            error = ods[mappings[item] + '.data'] * signals[item][0]['rel_error']
            error[error < signals[item][0]['abs_error'] * signals[item][0]['scale']] = (
                signals[item][0]['abs_error'] * signals[item][0]['scale']
            )
            ods[mappings[item] + '.data'] = unumpy.uarray(ods[mappings[item] + '.data'], error)


# =====================
if __name__ == '__main__':
    test_machine_mapping_functions(__all__, globals(), locals())
