import os
import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd, printe, unumpy
from omas.machine_mappings._common import *
import glob
import pyuda

__all__ = []
__regression_arguments__ = {'__all__': __all__}

def get_pyuda_client(server=None, port=None):
    if server is not None:
        pyuda.Client.server = server
    elif not os.environ['UDA_HOST']:
        raise pyuda.UDAException('Must set UDA_HOST environmental variable')

    if port is not None:
        pyuda.Client.port = port
    elif not os.environ['UDA_PORT']:
        raise pyuda.UDAException('Must set UDA_PORT environmental variable')

    return pyuda.Client()

def nstx_filenames(filename, pulse):

    return support_filenames('mast', filename, pulse)



@machine_mapping_function(__regression_arguments__, pulse=44653)
def pf_active_hardware(ods, pulse):
    r"""
    Loads MAST tokamak poloidal field coil hardware geometry

    :param ods: ODS instance
    """
    from omfit_classes.omfit_efund import OMFITmhdin, OMFITnstxMHD

    mhdin = get_support_file(OMFITmhdin, nstx_filenames('mhdin', pulse))
    mhdin.to_omas(ods, update=['oh', 'pf_active'])

    signals = get_support_file(OMFITnstxMHD, nstx_filenames('signals', pulse))
    icoil_signals = signals['mappings']['icoil']
    oh_signals = signals['mappings']['ioh']

    c_pf = 0
    c_oh = 0
    for c in ods[f'pf_active.coil']:
        if 'OH' in ods[f'pf_active.coil'][c]['name']:
            c_oh += 1
            cname = oh_signals[c_oh]['name']
            if oh_signals[c_oh]['mds_name'] is None:
                cid = 'None'
            else:
                cid = oh_signals[c_oh]['mds_name'].strip('\\').replace("~", " ")
        else:
            c_pf += 1
            cname = icoil_signals[c_pf]['name']
            if icoil_signals[c_pf]['mds_name'] is None:
                cid = 'None'
            else:
                cid = icoil_signals[c_pf]['mds_name'].strip('\\').replace("~", " ")
        for e in ods[f'pf_active.coil'][c]['element']:
            # if 'OH' in ods[f'pf_active.coil'][c]['name']:
            #    ename = oh_signals[c_oh]['mds_name_resolved'].strip('\\') + f'_element_{e}'
            # else:
            #    ename = icoil_signals[c_pf]['mds_name_resolved'].strip('\\') + f'_element_{e}'
            ename = cid + f'_element_{e}'
            eid = ename
            #ods[f'pf_active.coil'][c]['name'] = cname
            ods[f'pf_active.coil'][c]['identifier'] = cid
            ods[f'pf_active.coil'][c]['element'][e]['name'] = ename
            ods[f'pf_active.coil'][c]['element'][e]['identifier'] = eid


@machine_mapping_function(__regression_arguments__, pulse=44653)
def pf_active_coil_current_data(ods, pulse, server=None, port=None):
    r"""
    Load MAST tokamak pf_active coil current data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD
    from omfit_classes.utils_math import firFilter
    ods1 = ODS()
    unwrap(pf_active_hardware)(ods1, pulse)

    signals = get_support_file(OMFITnstxMHD, nstx_filenames('signals', pulse))
    icoil_signals = signals['mappings']['icoil']
    oh_signals = signals['mappings']['ioh']
    print(len(icoil_signals),len(oh_signals))
    # filter data with default smoothing & handle uncertainties
    oh_channel = 0
    pf_channel = 0

    client = get_pyuda_client(server=server, port=port)
    for channel in ods1['pf_active.coil']:
        if 'OH' in ods1[f'pf_active.coil.{channel}.name']:
            oh_channel += 1
            sig = oh_signals[oh_channel]
        else:
            pf_channel += 1
            sig = icoil_signals[pf_channel]
        try:
            sig_name = sig['mds_name_resolved'].replace('~',' ')
            if '/' in sig_name:
               tmp = ''
               for i in sig_name.split('/')[:-1]:
                   tmp +=i+'/'
               sig_name=tmp[:-1]
            print(sig_name)
            sig_scale = sig['scale']
            tmp = client.get(sig_name, pulse)
            printd(f'Smooth PF coil {channel} data', topic='machine')
            data = tmp.data * sig_scale
            ods[f'pf_active.coil.{channel}.current.time'] = time = tmp.time.data
            ods[f'pf_active.coil.{channel}.current.data'] = firFilter(time, data, [0, 300])

            rel_error = data * sig['rel_error']
            abs_error = sig['abs_error']
            error = np.sqrt(rel_error**2 + abs_error**2)
            error[np.abs(data) < sig['sig_thresh']] = sig['sig_thresh']
            ods[f'pf_active.coil.{channel}.current.data_error_upper'] = error

        except pyuda.UDAException:
            tmp = None
            ods[f'pf_active.coil.{channel}.current.data'] = np.array([])
            ods[f'pf_active.coil.{channel}.current.data_error_upper'] = np.array([])
            ods[f'pf_active.coil.{channel}.current.time'] = np.array([])   


    # IMAS stores the current in the coil not multiplied by the number of turns
    for channel in ods1['pf_active.coil']:
        if f'pf_active.coil.{channel}.current.data' in ods:
            ods[f'pf_active.coil.{channel}.current.data'] /= ods1[f'pf_active.coil.{channel}.element.0.turns_with_sign']
            ods[f'pf_active.coil.{channel}.current.data_error_upper'] /= ods1[f'pf_active.coil.{channel}.element.0.turns_with_sign']
        else:
            print(f'WARNING: pf_active.coil[{channel}].current.data is missing')


@machine_mapping_function(__regression_arguments__, pulse=44653)
def magnetics_hardware(ods, pulse):
    r"""
    Load MAST tokamak flux loops and magnetic probes hardware geometry

    :param ods: ODS instance
    """
    from omfit_classes.omfit_efund import OMFITmhdin, OMFITnstxMHD

    mhdin = get_support_file(OMFITmhdin, nstx_filenames('mhdin', pulse))
    mhdin.to_omas(ods, update='magnetics')

    signals = get_support_file(OMFITnstxMHD, nstx_filenames('signals', pulse))

    for k in ods[f'magnetics.flux_loop']:
        ods[f'magnetics.flux_loop.{k}.identifier'] = str(signals['mappings']['tfl'][k + 1]['mds_name'])

    for k in ods[f'magnetics.b_field_pol_probe']:
        ods[f'magnetics.b_field_pol_probe.{k}.identifier'] = str(signals['mappings']['bmc'][k + 1]['mds_name'])


@machine_mapping_function(__regression_arguments__, pulse=44653)
def magnetics_floops_data(ods, pulse, server=None, port=None):
    r"""
    Load MAST tokamak flux loops flux data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD
    
    signals = get_support_file(OMFITnstxMHD, nstx_filenames('signals', pulse))
    tfl_signals = signals['mappings']['tfl']

    client = get_pyuda_client(server=server, port=port)

    for channel in signals['FL']:
        sig = signals['FL'][channel]['mds_name']

        scale = signals['FL'][channel]['scale']     
        try:
            tmp = client.get(sig,pulse)
            ods[f'magnetics.flux_loop.{channel}.flux.data'] = tmp.data * scale
            ods[f'magnetics.flux_loop.{channel}.flux.time'] = tmp.time.data

        except pyuda.UDAException:
            ods[f'magnetics.flux_loop.{channel}.flux.validity'] = -2


    # handle uncertainties
    tfl_signals = signals['mappings']['tfl']
    for channel in range(len(ods['magnetics.flux_loop']) - 1):
        if f'magnetics.flux_loop.{channel}.flux.data' in ods:
            data = ods[f'magnetics.flux_loop.{channel}.flux.data']
            rel_error = data * tfl_signals[channel + 1]['rel_error']
            abs_error = tfl_signals[channel + 1]['abs_error']
            error = np.sqrt(rel_error**2 + abs_error**2)
            error[np.abs(data) < tfl_signals[channel + 1]['sig_thresh']] = tfl_signals[channel + 1]['sig_thresh']
            ods[f'magnetics.flux_loop.{channel}.flux.data_error_upper'] = error
            # 2*pi normalization is done at this stage so that rel_error, abs_error, sig_thresh are consistent with data
            ods[f'magnetics.flux_loop.{channel}.flux.data'] /= 2.0 * np.pi
            ods[f'magnetics.flux_loop.{channel}.flux.data_error_upper'] /= 2.0 * np.pi


@machine_mapping_function(__regression_arguments__, pulse=44653)
def magnetics_probes_data(ods, pulse, server=None, port=None):
    r"""
    Load NSTX-U tokamak magnetic probes field data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD

    signals = get_support_file(OMFITnstxMHD, nstx_filenames('signals', pulse))

    client = get_pyuda_client(server=server, port=port)
    for channel in signals['MC']:
        sig = signals['MC'][channel]['mds_name']
        scale = signals['MC'][channel]['scale']     
        try:
            tmp = client.get(sig,pulse)
            ods[f'magnetics.b_field_pol_probe.{channel}.field.data'] = tmp.data * scale
            ods[f'magnetics.b_field_pol_probe.{channel}.field.time'] = tmp.time.data

        except pyuda.UDAException:
            ods[f'magnetics.b_field_pol_probe.{channel}.field.validity'] = -2

    # handle uncertainties
    bmc_signals = signals['mappings']['bmc']
    for channel in ods['magnetics.b_field_pol_probe']:
        if f'magnetics.b_field_pol_probe.{channel}.field.data' in ods:
            data = ods[f'magnetics.b_field_pol_probe.{channel}.field.data']
            rel_error = data * bmc_signals[channel + 1]['rel_error']
            abs_error = bmc_signals[channel + 1]['abs_error']
            error = np.sqrt(rel_error**2 + abs_error**2)
            error[np.abs(data) < bmc_signals[channel + 1]['sig_thresh']] = bmc_signals[channel + 1]['sig_thresh']
            ods[f'magnetics.b_field_pol_probe.{channel}.field.data_error_upper'] = error



@machine_mapping_function(__regression_arguments__, pulse=44653)
def ip_bt_dflux_data(ods, pulse, server=None, port=None):
    r"""
    Load MAST tokamak Ip, Bt, and diamagnetic flux data

    :param ods: ODS instance

    :param pulse: shot number
    """
    from omfit_classes.omfit_efund import OMFITnstxMHD

    signals = get_support_file(OMFITnstxMHD, nstx_filenames('signals', pulse))

    mappings = {'PR': 'magnetics.ip.0', 'TF': 'tf.b_field_tor_vacuum_r'}

    client = get_pyuda_client(server=server, port=port)

    for item in ['PR', 'TF']:
        sig = signals[item][0]['mds_name'].replace('~',' ')
        scale = signals[item][0]['scale']
        try:
            tmp = client.get(sig,pulse)

            ods[mappings[item] + '.data'] = tmp.data * scale
            ods[mappings[item] + '.time'] = tmp.time.data
        except pyuda.UDAException:
            printe(f'No data for {mappings[item]}')
            ods[mappings[item] + '.data'] = []
            ods[mappings[item] + '.time'] = []

    # handle uncertainties
    for item in ['PR', 'TF']:
        if mappings[item] + '.data' in ods:
            data = ods[mappings[item] + '.data']
            rel_error = data * signals[item][0]['rel_error']
            abs_error = signals[item][0]['abs_error'] * signals[item][0]['scale']
            error = np.sqrt(rel_error**2 + abs_error**2)
            error[np.abs(data) < signals[item][0]['sig_thresh'] * signals[item][0]['scale']] = (
                signals[item][0]['sig_thresh'] * signals[item][0]['scale']
            )
            ods[mappings[item] + '.data_error_upper'] = error


# ================================
@machine_mapping_function(__regression_arguments__, pulse=44653)
def thomson_scattering_hardware(ods, pulse):
    """
    Gathers MAST(-U) Thomson measurement locations

    :param pulse: int

    """
    unwrap(thomson_scattering_data)(ods, pulse)


@machine_mapping_function(__regression_arguments__, pulse=44653)
def thomson_scattering_data(ods, pulse):
    """
    Loads MAST Thomson measurement data

    :param pulse: int
    """

    return


# =====================
if __name__ == '__main__':
    test_machine_mapping_functions(__all__, globals(), locals())
