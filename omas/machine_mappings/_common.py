import numpy as np
from omas import *
from omas.omas_utils import printd
import os
import glob
from omas.omas_setup import omas_dir
from omas.utilities.omas_mds import mdsvalue, get_pulse_id

__support_files_cache__ = {}


class D3DFitweight(dict):
    """
    OMFIT class to read DIII-D fitweight file
    """

    def __init__(self, filename):
        r"""
        OMFIT class to parse DIII-D device files

        :param filename: filename

        :param \**kw: arguments passed to __init__ of OMFITascii
        """
        self.filename = filename

    def load(self):
        self.clear()

        magpri67 = 29
        magpri322 = 31
        magprirdp = 8
        magudom = 5
        maglds = 3
        nsilds = 3
        nsilol = 41

        with open(self.filename, 'r') as f:
            data = f.read()

        data = data.strip().split()

        for i in data:
            ifloat = float(i)
            if ifloat > 100:
                ishot = int(ifloat)
                self[ifloat] = []
            else:
                self[ishot].append(ifloat)

        for irshot in self:
            if irshot < 124985:
                mloop = nsilol
            else:
                mloop = nsilol + nsilds

            if irshot < 59350:
                mprobe = magpri67
            elif irshot < 91000:
                mprobe = magpri67 + magpri322
            elif irshot < 100771:
                mprobe = magpri67 + magpri322 + magprirdp
            elif irshot < 124985:
                mprobe = magpri67 + magpri322 + magprirdp + magudom
            else:
                mprobe = magpri67 + magpri322 + magprirdp + magudom + maglds
            fwtmp2 = self[irshot][mloop : mloop + mprobe]
            fwtsi = self[irshot][0:mloop]
            self[irshot] = {}
            self[irshot]['fwtmp2'] = fwtmp2
            self[irshot]['fwtsi'] = fwtsi

        return self

class D3DCompfile(dict):
    """
    OMFIT class to read DIII-D compensation files such as btcomp.dat ccomp.dat and icomp.dat
    """

    def __init__(self, filename):
        r"""
        OMFIT class to parse DIII-D MHD device files

        :param filename: filename

        :param \**kw: arguments passed to __init__ of OMFITascii
        """
        self.filename=filename

    def load(self):
        self.clear()
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            linesplit = line.split()

            try:
                compshot = int(eval(linesplit[0]))
                self[compshot] = {}
                for compsig in linesplit[1:]:
                    self[compshot][compsig.strip("'")] = {}

            except Exception:
                sig = linesplit[0][1:].strip()
                comps = [float(i) for i in linesplit[2:]]
                for comp, compsig in zip(comps, self[compshot]):
                    self[compshot][compsig][sig] = comp

        return self

def support_filenames(device, filename, pulse):
    topdir = os.sep.join([omas_dir, 'machine_mappings', 'support_files', device])
    for rangefile in glob.glob(os.sep.join([topdir, '*', 'ranges.dat'])):
        with open(rangefile) as f:
            start, stop = map(int, f.read().split())
            if start < pulse < stop:
                dir = os.path.dirname(rangefile)
                filenames = glob.glob(os.sep.join([dir, filename]) + '*')
                if len(filenames):
                    filename = filenames[0]
                    printd(f'Reading {filename}', topic='machine')
                    return filename

    filenames = glob.glob(os.sep.join([topdir, filename]) + '*')
    if len(filenames):
        filename = filenames[0]
        printd(f'Reading {filename}', topic='machine')
        return filename
    raise FileNotFoundError(f"Could not find `{filename}` in {topdir} or any of its subdirectories.")


def get_support_file(object_type, filename):
    """
    Cached loading of support files

    :param object_type: Typically a OMFIT class

    :param filename: filename of the support file to load
    """
    if filename not in __support_files_cache__:
        __support_files_cache__[filename] = object_type(filename)
        __support_files_cache__[filename].load()
    return __support_files_cache__[filename]


__MDS_gEQDSK_COCOS_identify_cache__ = {}


def D3Dmagnetics_weights(pulse, name=None):
    r"""
    Load DIII-D tokamak magnetics equilibrium weights

    :param pulse: pulse number

    :param name: name of the type of weights to return

    :return: dictionary with the requested weights or both if name=None
    """

    fitweight = get_support_file(D3DFitweight, support_filenames('d3d', 'fitweight', pulse))
    if len(fitweight) == 0:
        raise ValueError(f"Could not find d3d fitweight for shot {pulse}")
    weight_ishot = -1
    for ishot in fitweight:
        if pulse > ishot and ishot > weight_ishot:
            weight_ishot = ishot

    if name is None:
        return fitweight[weight_ishot]['fwtmp2'], fitweight[weight_ishot]['fwtsi']
    elif name in fitweight[weight_ishot]:
        return fitweight[weight_ishot][name]
    else:
        raise ValueError(f"{name} is part of the d3d fitweight")


def MDS_gEQDSK_COCOS_identify(machine, pulse, EFIT_tree, EFIT_run_id):
    """
    Python function that queries MDSplus EFIT tree to figure
    out COCOS convention used for a particular reconstruction

    :param machine: machine name

    :param pulse: pulse number

    :param EFIT_tree: MDSplus EFIT tree name

    :param EFIT_run_id:  with id extension for non-standard shot numbers. E.g. 19484401 for EFIT tree

    :return: integer cocos convention
    """
    pulse_id =  get_pulse_id(pulse, EFIT_run_id)
    if (machine, pulse_id, EFIT_tree) in __MDS_gEQDSK_COCOS_identify_cache__:
        return __MDS_gEQDSK_COCOS_identify_cache__[(machine, pulse_id, EFIT_tree)]
    TDIs = {'bt': f'mean(\\{EFIT_tree}::TOP.RESULTS.GEQDSK.BCENTR)', 'ip': f'mean(\\{EFIT_tree}::TOP.RESULTS.GEQDSK.CPASMA)'}
    res = mdsvalue(machine, EFIT_tree, pulse_id, TDIs).raw()
    bt = res['bt']
    ip = res['ip']
    g_cocos = {(+1, +1): 1, (+1, -1): 3, (-1, +1): 5, (-1, -1): 7, (+1, 0): 1, (-1, 0): 3}
    sign_Bt = int(np.sign(bt))
    sign_Ip = int(np.sign(ip))
    cocosio = g_cocos.get((sign_Bt, sign_Ip), None)
    __MDS_gEQDSK_COCOS_identify_cache__[(machine, pulse_id, EFIT_tree)] = cocosio
    return cocosio


def MDS_gEQDSK_psi(ods, machine, pulse, EFIT_tree):
    """
    evaluate EFIT psi

    :param ODS: input ODS

    :param machine: machine name

    :param pulse: pulse

    :param EFIT_tree: MDSplus EFIT tree name

    :return: integer cocos convention
    """
    cocosio = MDS_gEQDSK_COCOS_identify(machine, pulse, EFIT_tree)
    with omas_environment(ods, cocosio=cocosio):
        TDIs = {
            'psi_axis': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIMAG',
            'psi_boundary': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.SSIBRY',
            'rho_tor_norm': f'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.PSIN',
        }
        res = mdsvalue(machine, EFIT_tree, pulse, TDIs).raw()
        n = res['rho_tor_norm'].shape[1]
        for k in range(len(res['psi_axis'])):
            ods[f'equilibrium.time_slice.{k}.global_quantities.psi_axis'] = res['psi_axis'][k]
            ods[f'equilibrium.time_slice.{k}.global_quantities.psi_boundary'] = res['psi_boundary'][k]
            ods[f'equilibrium.time_slice.{k}.profiles_1d.rho_tor_norm'] = res['rho_tor_norm'][k]
            ods[f'equilibrium.time_slice.{k}.profiles_1d.psi'] = res['psi_axis'][k] + np.linspace(0, 1, n) * (
                res['psi_boundary'][k] - res['psi_axis'][k]
            )


def pf_coils_to_ods(ods, coil_data):
    """
    Transfers poloidal field coil geometry data from a standard EFIT mhdin.dat format to ODS.

    WARNING: only rudimentary identifies are assigned.
    You should assign your own identifiers and only rely on this function to assign numerical geometry data.

    :param ods: ODS instance
        Data will be added in-place

    :param coil_data: 2d array
        coil_data[i] corresponds to coil i. The columns are R (m), Z (m), dR (m), dZ (m), tilt1 (deg), and tilt2 (deg)
        This should work if you just copy data from iris:/fusion/usc/src/idl/efitview/diagnoses/<device>/*coils*.dat
        (the filenames for the coils vary)

    :return: ODS instance
    """

    from omas.omas_plot import geo_type_lookup

    rect_code = geo_type_lookup('rectangle', 'pf_active', ods.imas_version, reverse=True)
    outline_code = geo_type_lookup('outline', 'pf_active', ods.imas_version, reverse=True)

    nc = len(coil_data[:, 0])

    for i in range(nc):
        ods['pf_active.coil'][i]['name'] = ods['pf_active.coil'][i]['identifier'] = 'PF{}'.format(i)
        if (coil_data[i, 4] == 0) and (coil_data[i, 5] == 0):
            rect = ods['pf_active.coil'][i]['element.0.geometry.rectangle']
            rect['r'] = coil_data[i, 0]
            rect['z'] = coil_data[i, 1]
            rect['width'] = coil_data[i, 2]  # Or width in R
            rect['height'] = coil_data[i, 3]  # Or height in Z
            ods['pf_active.coil'][i]['element.0.geometry.geometry_type'] = rect_code
        else:
            outline = ods['pf_active.coil'][i]['element.0.geometry.outline']
            fdat = coil_data[i]
            fdat[4] = -coil_data[i, 4] * np.pi / 180.0
            fdat[5] = -(coil_data[i, 5] * np.pi / 180.0 if coil_data[i, 5] != 0 else np.pi / 2)
            outline['r'] = [
                fdat[0] - fdat[2] / 2.0 - fdat[3] / 2.0 * np.tan((np.pi / 2.0 + fdat[5])),
                fdat[0] - fdat[2] / 2.0 + fdat[3] / 2.0 * np.tan((np.pi / 2.0 + fdat[5])),
                fdat[0] + fdat[2] / 2.0 + fdat[3] / 2.0 * np.tan((np.pi / 2.0 + fdat[5])),
                fdat[0] + fdat[2] / 2.0 - fdat[3] / 2.0 * np.tan((np.pi / 2.0 + fdat[5])),
            ]
            outline['z'] = [
                fdat[1] - fdat[3] / 2.0 - fdat[2] / 2.0 * np.tan(-fdat[4]),
                fdat[1] + fdat[3] / 2.0 - fdat[2] / 2.0 * np.tan(-fdat[4]),
                fdat[1] + fdat[3] / 2.0 + fdat[2] / 2.0 * np.tan(-fdat[4]),
                fdat[1] - fdat[3] / 2.0 + fdat[2] / 2.0 * np.tan(-fdat[4]),
            ]
            ods['pf_active.coil'][i]['element.0.geometry.geometry_type'] = outline_code

    return ods


def fetch_assign(
    ods,
    ods1,
    pulse,
    channels,
    identifier,
    time,
    data,
    validity,
    mds_server,
    mds_tree,
    tdi_expression,
    time_norm,
    data_norm,
    homogeneous_time=True,
):
    """
    Utility function to get data from a list of TDI signals which all share the same time basis

    :param ods: ODS that will hold the data

    :param ods1: ODS that contains the channels information

    :param pulse: pulse number

    :param channels: location in `ods1` where the channels are defined

    :param identifier: location in `ods1` with the name of the signal to be retrieved

    :param time: location in `ods` where to set the time info

    :param data: location in `ods` where to set the data

    :param validity: location in `ods` where to set the validity flag

    :param mds_server: MDSplus server to connect to

    :param mds_tree: MDSplus tree from where to get the data

    :param tdi_expression: string with tdi_expression to use

    :param time_norm: time normalization

    :param data_norm: data normalization

    :param homogeneous_time: data has homogeneous time basis

    :return: ODS instance
    """
    t = None
    TDIs = []

    if isinstance(channels, str):
        channels = ods1[channels]

    for stage in ['fetch', 'assign']:
        for channel in channels:
            signal = ods1[identifier.format(**locals())]
            TDI = tdi_expression.format(**locals())
            TDIs.append(TDI)
            if not homogeneous_time:
                TDIs.append(f'dim_of({TDI},0)')
            elif stage == 'fetch' and t is None:
                try:
                    t = mdsvalue(mds_server, mds_tree, pulse, TDI=TDI).dim_of(0)
                    if len(t) <= 1:
                        t = None
                except Exception:
                    pass
            if stage == 'assign':
                if homogeneous_time and t is None:
                    raise RuntimeError(f'Could not determine time info from {TDI} signals')
                time_loc = str(time.format(**locals()))
                if 'None' in TDI and validity is None:
                    time_loc = time.format(**locals())
                    if not homogeneous_time:
                        ods[time_loc] = [0.0, 1e10]
                    else:
                        ods[time.format(**locals())] = t * time_norm
                    ods[data.format(**locals())] = np.zeros(len(ods[time.format(**locals())]))
                    ods[data.format(**locals())][:] = np.nan
                elif not isinstance(tmp[TDI], Exception):
                    if not homogeneous_time:
                        ods[time.format(**locals())] = tmp[f'dim_of({TDI},0)'] * time_norm
                    else:
                        ods[time.format(**locals())] = t * time_norm
                    ods[data.format(**locals())] = tmp[TDI] * data_norm
                    if validity is not None:
                        if len(ods[time.format(**locals())]) == len(ods[data.format(**locals())]) and len(ods[data.format(**locals())]) > 1:
                            ods[validity.format(**locals())] = 0
                        else:
                            ods[validity.format(**locals())] = -2
                elif validity is not None:
                    ods[validity.format(**locals())] = -2
        if stage == 'fetch':
            tmp = mdsvalue(mds_server, mds_tree, pulse, TDI=TDIs).raw()
    return ods
