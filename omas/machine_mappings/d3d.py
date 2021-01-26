import numpy as np
import copy
from omas import *
from omas.omas_utils import printd

__all__ = []


@machine_mapping_function(__all__)
def setup_gas_injection_hardware_description_d3d(ods, pulse=133221):
    """
    Sets up DIII-D gas injector data.

    R and Z are from the tips of the arrows in puff_loc.pro; phi from angle listed in labels in puff_loc.pro .
    I recorded the directions of the arrows on the EFITviewer overlay, but I don't know how to include them in IMAS, so
    I commented them out.

    Warning: changes to gas injector configuration with time are not yet included. This is just the best picture I could
    make of the 2018 configuration.

    Data sources:
    EFITVIEWER: iris:/fusion/usc/src/idl/efitview/diagnoses/DIII-D/puff_loc.pro accessed 2018 June 05, revised 20090317
    DIII-D webpage: https://diii-d.gat.com/diii-d/Gas_Schematic accessed 2018 June 05
    DIII-D wegpage: https://diii-d.gat.com/diii-d/Gas_PuffLocations accessed 2018 June 05

    Updated 2018 June 05 by David Eldon

    :return: dict
        Information or instructions for follow up in central hardware description setup
    """
    if pulse < 100775:
        warnings.warn('DIII-D Gas valve locations not applicable for shots earlier than 100775 (2000 JAN 17)')

    i = 0

    def pipe_copy(pipe_in):
        pipe_out = ods['gas_injection']['pipe'][i]
        for field in ['name', 'exit_position.r', 'exit_position.z', 'exit_position.phi']:
            pipe_out[field] = copy.copy(pipe_in[field])
        vvv = 0
        while 'valve.{}.identifier'.format(vvv) in pipe_in:
            pipe_out = copy.copy(pipe_in['valve.{}.identifier'.format(vvv)])
            vvv += 1
        return pipe_out

    # PFX1
    for angle in [12, 139, 259]:  # degrees, DIII-D hardware left handed coords
        pipe_pfx1 = ods['gas_injection']['pipe'][i]
        pipe_pfx1['name'] = 'PFX1_{:03d}'.format(angle)
        pipe_pfx1['exit_position']['r'] = 1.286  # m
        pipe_pfx1['exit_position']['z'] = 1.279  # m
        pipe_pfx1['exit_position']['phi'] = -np.pi / 180.0 * angle  # radians, right handed
        pipe_pfx1['valve'][0]['identifier'] = 'PFX1'
        dr = -1.116 + 1.286
        dz = -1.38 + 1.279
        # pipea['exit_position']['direction'] = 180/np.pi * tan(dz/dr) if dr != 0 else 90 * sign(dz)
        pipe_pfx1['second_point']['phi'] = pipe_pfx1['exit_position']['phi']
        pipe_pfx1['second_point']['r'] = pipe_pfx1['exit_position']['r'] + dr
        pipe_pfx1['second_point']['z'] = pipe_pfx1['exit_position']['z'] + dz
        i += 1

    # PFX2 injects at the same poloidal locations as PFX1, but at different toroidal angles
    for angle in [79, 199, 319]:  # degrees, DIII-D hardware left handed coords
        pipe_copy(pipe_pfx1)
        pipe_pfx2 = ods['gas_injection']['pipe'][i]
        pipe_pfx2['name'] = 'PFX2_{:03d}'.format(angle)
        pipe_pfx2['exit_position']['phi'] = -np.pi / 180.0 * angle  # rad
        pipe_pfx2['valve'][0]['identifier'] = 'PFX2'
        pipe_pfx2['second_point']['phi'] = pipe_pfx2['exit_position']['phi']
        i += 1

    # GAS A
    pipea = ods['gas_injection']['pipe'][i]
    pipea['name'] = 'GASA_300'
    pipea['exit_position']['r'] = 1.941  # m
    pipea['exit_position']['z'] = 1.01  # m
    pipea['exit_position']['phi'] = -np.pi / 180.0 * 300  # rad
    pipea['valve'][0]['identifier'] = 'GASA'
    # pipea['exit_position']['direction'] = 270.  # degrees, giving dir of pipe leading towards injector, up is 90
    pipea['second_point']['phi'] = pipea['exit_position']['phi']
    pipea['second_point']['r'] = pipea['exit_position']['r']
    pipea['second_point']['z'] = pipea['exit_position']['z'] - 0.01
    i += 1

    # GAS B injects in the same place as GAS A
    pipe_copy(pipea)
    pipeb = ods['gas_injection']['pipe'][i]
    pipeb['name'] = 'GASB_300'
    pipeb['valve'][0]['identifier'] = 'GASB'
    i += 1

    # GAS C
    pipec = ods['gas_injection']['pipe'][i]
    pipec['name'] = 'GASC_000'
    pipec['exit_position']['r'] = 1.481  # m
    pipec['exit_position']['z'] = -1.33  # m
    pipec['exit_position']['phi'] = -np.pi / 180.0 * 0
    pipec['valve'][0]['identifier'] = 'GASC'
    pipec['valve'][1]['identifier'] = 'GASE'
    # pipec['exit_position']['direction'] = 90.  # degrees, giving direction of pipe leading towards injector
    pipec['second_point']['phi'] = pipec['exit_position']['phi']
    pipec['second_point']['r'] = pipec['exit_position']['r']
    pipec['second_point']['z'] = pipec['exit_position']['z'] + 0.01
    i += 1

    # GAS D injects at the same poloidal location as GAS A, but at a different toroidal angle.
    # There is a GASD piezo valve that splits into four injectors, all of which have their own gate valves and can be
    # turned on/off independently. Normally, only one would be used at at a time.
    pipe_copy(pipea)
    piped = ods['gas_injection']['pipe'][i]
    piped['name'] = 'GASD_225'  # This is the injector name
    piped['exit_position']['phi'] = -np.pi / 180.0 * 225
    piped['valve'][0]['identifier'] = 'GASD'  # This is the piezo name
    piped['second_point']['phi'] = piped['exit_position']['phi']
    i += 1

    # Spare 225 is an extra branch of the GASD line after the GASD piezo
    pipe_copy(piped)
    pipes225 = ods['gas_injection']['pipe'][i]
    pipes225['name'] = 'Spare_225'  # This is the injector name
    i += 1

    # RF_170 and RF_190: gas ports near the 180 degree antenna, on the GASD line
    for angle in [170, 190]:
        pipe_rf = ods['gas_injection']['pipe'][i]
        pipe_rf['name'] = 'RF_{:03d}'.format(angle)
        pipe_rf['exit_position']['r'] = 2.38  # m
        pipe_rf['exit_position']['z'] = -0.13  # m
        pipe_rf['exit_position']['phi'] = -np.pi / 180.0 * angle  # rad
        pipe_rf['valve'][0]['identifier'] = 'GASD'
        i += 1

    # DRDP
    pipe_copy(piped)
    piped = ods['gas_injection']['pipe'][i]
    piped['name'] = 'DRDP_225'
    piped['valve'][0]['identifier'] = 'DRDP'
    i += 1

    # UOB
    for angle in [45, 165, 285]:
        pipe_uob = ods['gas_injection']['pipe'][i]
        pipe_uob['name'] = 'UOB_{:03d}'.format(angle)
        pipe_uob['exit_position']['r'] = 1.517  # m
        pipe_uob['exit_position']['z'] = 1.267  # m
        pipe_uob['exit_position']['phi'] = -np.pi / 180.0 * angle
        pipe_uob['valve'][0]['identifier'] = 'UOB'
        # pipe_uob['exit_position']['direction'] = 270.  # degrees, giving dir of pipe leading to injector, up is 90
        i += 1

    # LOB1
    for angle in [30, 120]:
        pipe_lob1 = ods['gas_injection']['pipe'][i]
        pipe_lob1['name'] = 'LOB1_{:03d}'.format(angle)
        pipe_lob1['exit_position']['r'] = 1.941  # m
        pipe_lob1['exit_position']['z'] = -1.202  # m
        pipe_lob1['exit_position']['phi'] = -np.pi / 180.0 * angle
        pipe_lob1['valve'][0]['identifier'] = 'LOB1'
        # pipe_lob1['exit_position']['direction'] = 180.  # degrees, giving dir of pipe leading to injector; up is 90
        i += 1

    # Spare 75 is an extra branch of the GASC line after the LOB1 piezo
    pipes75 = ods['gas_injection']['pipe'][i]
    pipes75['name'] = 'Spare_075'
    pipes75['exit_position']['r'] = 2.249  # m (approximate / estimated from still image)
    pipes75['exit_position']['z'] = -0.797  # m (approximate / estimated from still image)
    pipes75['exit_position']['phi'] = 75  # degrees, DIII-D hardware left handed coords
    pipes75['valve'][0]['identifier'] = 'LOB1'
    # pipes75['exit_position']['direction'] = 180.  # degrees, giving direction of pipe leading towards injector
    i += 1

    # RF_010 & 350
    for angle in [10, 350]:
        pipe_rf_lob1 = ods['gas_injection']['pipe'][i]
        pipe_rf_lob1['name'] = 'RF_{:03d}'.format(angle)
        pipe_rf_lob1['exit_position']['r'] = 2.38  # m
        pipe_rf_lob1['exit_position']['z'] = -0.13  # m
        pipe_rf_lob1['exit_position']['phi'] = -np.pi / 180.0 * angle
        pipe_rf_lob1['valve'][0]['identifier'] = 'LOB1'
        # pipe_rf10['exit_position']['direction'] = 180.  # degrees, giving dir of pipe leading to injector; up is 90
        i += 1

    # DiMES chimney
    pipe_dimesc = ods['gas_injection']['pipe'][i]
    pipe_dimesc['name'] = 'DiMES_Chimney_165'
    pipe_dimesc['exit_position']['r'] = 1.481  # m
    pipe_dimesc['exit_position']['z'] = -1.33  # m
    pipe_dimesc['exit_position']['phi'] = -np.pi / 180.0 * 165
    pipe_dimesc['valve'][0]['identifier'] = '240R-2'
    pipe_dimesc['valve'][0]['name'] = '240R-2 (PCS use GASD)'
    # pipe_dimesc['exit_position']['direction'] = 90.  # degrees, giving dir of pipe leading towards injector, up is 90
    i += 1

    # CPBOT
    pipe_cpbot = ods['gas_injection']['pipe'][i]
    pipe_cpbot['name'] = 'CPBOT_150'
    pipe_cpbot['exit_position']['r'] = 1.11  # m
    pipe_cpbot['exit_position']['z'] = -1.33  # m
    pipe_cpbot['exit_position']['phi'] = -np.pi / 180.0 * 150
    pipe_cpbot['valve'][0]['identifier'] = '240R-2'
    pipe_cpbot['valve'][0]['name'] = '240R-2 (PCS use GASD)'
    # pipe_cpbot['exit_position']['direction'] = 0.  # degrees, giving dir of pipe leading towards injector, up is 90
    i += 1

    # LOB2 injects at the same poloidal locations as LOB1, but at different toroidal angles
    for angle in [210, 300]:
        pipe_copy(pipe_lob1)
        pipe_lob2 = ods['gas_injection']['pipe'][i]
        pipe_lob2['name'] = 'LOB2_{:03d}'.format(angle)
        pipe_lob2['exit_position']['phi'] = -np.pi / 180.0 * angle  # degrees, DIII-D hardware left handed coords
        pipe_lob2['valve'][0]['identifier'] = 'LOB2'
        i += 1

    # Dimes floor tile 165
    pipe_copy(pipec)
    pipe_dimesf = ods['gas_injection']['pipe'][i]
    pipe_dimesf['name'] = 'DiMES_Tile_160'
    pipe_dimesf['exit_position']['phi'] = -np.pi / 180.0 * 165
    pipe_dimesf['valve'][0]['identifier'] = 'LOB2'
    i += 1

    # RF COMB
    pipe_rfcomb = ods['gas_injection']['pipe'][i]
    pipe_rfcomb['name'] = 'RF_COMB_'
    pipe_rfcomb['exit_position']['r'] = 2.38  # m
    pipe_rfcomb['exit_position']['z'] = -0.13  # m
    # pipe_rfcomb['exit_position']['phi'] = Unknown, sorry
    pipe_rfcomb['valve'][0]['identifier'] = 'LOB2'
    # pipe_rf307['exit_position']['direction'] = 180.  # degrees, giving dir of pipe leading towards injector, up is 90
    i += 1

    # RF307
    pipe_rf307 = ods['gas_injection']['pipe'][i]
    pipe_rf307['name'] = 'RF_307'
    pipe_rf307['exit_position']['r'] = 2.38  # m
    pipe_rf307['exit_position']['z'] = -0.13  # m
    pipe_rf307['exit_position']['phi'] = -np.pi / 180.0 * 307
    pipe_rf307['valve'][0]['identifier'] = 'LOB2'
    # pipe_rf307['exit_position']['direction'] = 180.  # degrees, giving dir of pipe leading towards injector, up is 90
    i += 1

    # GAS H injects in the same place as GAS C
    pipe_copy(pipec)
    pipeh = ods['gas_injection']['pipe'][i]
    pipeh['name'] = 'GASH_000'
    pipeh['valve'][0]['identifier'] = '???'  # This one's not on the manifold schematic
    i += 1

    # GAS I injects in the same place as GAS C
    pipe_copy(pipec)
    pipei = ods['gas_injection']['pipe'][i]
    pipei['name'] = 'GASI_000'
    pipei['valve'][0]['identifier'] = '???'  # This one's not on the manifold schematic
    i += 1

    # GAS J injects in the same place as GAS D
    pipe_copy(piped)
    pipej = ods['gas_injection']['pipe'][i]
    pipej['name'] = 'GASJ_225'
    pipej['valve'][0]['identifier'] = '???'  # This one's not on the manifold schematic
    i += 1

    # RF260
    pipe_rf260 = ods['gas_injection']['pipe'][i]
    pipe_rf260['name'] = 'RF_260'
    pipe_rf260['exit_position']['r'] = 2.38  # m
    pipe_rf260['exit_position']['z'] = 0.14  # m
    pipe_rf260['exit_position']['phi'] = -np.pi / 180.0 * 260
    pipe_rf260['valve'][0]['identifier'] = 'LOB2?'  # Seems to have been removed. May have been on LOB2, though.
    # pipe_rf260['exit_position']['direction'] = 180.  # degrees, giving dir of pipe leading towards injector, up is 90
    i += 1

    # CPMID
    pipe_cpmid = ods['gas_injection']['pipe'][i]
    pipe_cpmid['name'] = 'CPMID'
    pipe_cpmid['exit_position']['r'] = 0.9  # m
    pipe_cpmid['exit_position']['z'] = -0.2  # m
    pipe_cpmid['valve'][0]['identifier'] = '???'  # Seems to have been removed. Not on schematic.
    # pipe_cpmid['exit_position']['direction'] = 0.  # degrees, giving dir of pipe leading towards injector, up is 90
    i += 1

    return {}


def pf_coils_to_ods(ods, coil_data):
    """
    Transfers poloidal field coil geometry data from a standard format used by efitviewer to ODS.

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


@machine_mapping_function(__all__)
def setup_pf_active_hardware_description_d3d(ods):
    r"""
    Adds DIII-D tokamak poloidal field coil hardware geometry to ODS
    :param ods: ODS instance

    :param pulse: int

    :return: dict
        Information or instructions for follow up in central hardware description setup
    """
    # From  iris:/fusion/usc/src/idl/efitview/diagnoses/DIII-D/coils.dat , accessed 2018 June 08  D. Eldon
    # fmt: off
    fc_dat = np.array(
        [  # R        Z       dR      dZ    tilt1  tilt2
            [0.8608, 0.16830, 0.0508, 0.32106, 0.0, 0.0],  # 0 in the last column really means 90 degrees.
            [0.8614, 0.50810, 0.0508, 0.32106, 0.0, 0.0],
            [0.8628, 0.84910, 0.0508, 0.32106, 0.0, 0.0],
            [0.8611, 1.1899, 0.0508, 0.32106, 0.0, 0.0],
            [1.0041, 1.5169, 0.13920, 0.11940, 45.0, 0.0],
            [2.6124, 0.4376, 0.17320, 0.1946, 0.0, 92.40],
            [2.3733, 1.1171, 0.1880, 0.16920, 0.0, 108.06],
            [1.2518, 1.6019, 0.23490, 0.08510, 0.0, 0.0],
            [1.6890, 1.5874, 0.16940, 0.13310, 0.0, 0.0],
            [0.8608, -0.17370, 0.0508, 0.32106, 0.0, 0.0],
            [0.8607, -0.51350, 0.0508, 0.32106, 0.0, 0.0],
            [0.8611, -0.85430, 0.0508, 0.32106, 0.0, 0.0],
            [0.8630, -1.1957, 0.0508, 0.32106, 0.0, 0.0],
            [1.0025, -1.5169, 0.13920, 0.11940, -45.0, 0.0],
            [2.6124, -0.4376, 0.17320, 0.1946, 0.0, -92.40],
            [2.3834, -1.1171, 0.1880, 0.16920, 0.0, -108.06],
            [1.2524, -1.6027, 0.23490, 0.08510, 0.0, 0.0],
            [1.6889, -1.5780, 0.16940, 0.13310, 0.0, 0.0],
        ]
    )
    # fmt: on

    ods = pf_coils_to_ods(ods, fc_dat)

    for i in range(len(fc_dat[:, 0])):
        fcid = 'F{}{}'.format((i % 9) + 1, 'AB'[int(fc_dat[i, 1] < 0)])
        ods['pf_active.coil'][i]['name'] = ods['pf_active.coil'][i]['identifier'] = fcid
        ods['pf_active.coil'][i]['element.0.identifier'] = fcid

    return {}


@machine_mapping_function(__all__)
def setup_interferometer_hardware_description_d3d(ods, pulse=133221):
    """
    Writes DIII-D CO2 interferometer chord locations into ODS.

    The chord endpoints ARE NOT RIGHT. Only the R for vertical lines or Z for horizontal lines is right.

    Data sources:
    DIII-D webpage: https://diii-d.gat.com/diii-d/Mci accessed 2018 June 07  D. Eldon

    :param ods: an OMAS ODS instance

    :param pulse: int

    :return: dict
        Information or instructions for follow up in central hardware description setup
    """

    # As of 2018 June 07, DIII-D has four interferometers
    # phi angles are compliant with odd COCOS
    ods['interferometer.channel.0.identifier'] = 'r0'
    r0 = ods['interferometer.channel.0.line_of_sight']
    r0['first_point.phi'] = r0['second_point.phi'] = 225 * (-np.pi / 180.0)
    r0['first_point.r'], r0['second_point.r'] = 3.0, 0.8  # These are not the real endpoints
    r0['first_point.z'] = r0['second_point.z'] = 0.0

    for i, r in enumerate([1.48, 1.94, 2.10]):
        ods['interferometer.channel'][i + 1]['identifier'] = 'v{}'.format(i + 1)
        los = ods['interferometer.channel'][i + 1]['line_of_sight']
        los['first_point.phi'] = los['second_point.phi'] = 240 * (-np.pi / 180.0)
        los['first_point.r'] = los['second_point.r'] = r
        los['first_point.z'], los['second_point.z'] = -1.8, 1.8  # These are not the real points

    for i in range(len(ods['interferometer.channel'])):
        ch = ods['interferometer.channel'][i]
        ch['line_of_sight.third_point'] = copy.copy(ch['line_of_sight.first_point'])

    if pulse < 125406:
        printe(
            'DIII-D CO2 pointnames were different before pulse 125406. The physical locations of the chords seems to '
            'have been the same, though, so there has not been a problem yet (I think).'
        )

    return {}


def find_thomson_lens_d3d(pulse, hw_call_sys, hw_revision='blessed'):
    """Read the Thomson scattering hardware map to figure out which lens each chord looks through"""
    cal_call = '.ts.{}.header.calib_nums'.format(hw_revision)
    cal_set = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=cal_call).data()[0]
    hwi_call = '.{}.hwmapints'.format(hw_call_sys)
    printd('  Reading hw map int values: treename = "tscal", cal_set = {}, hwi_call = {}'.format(cal_set, hwi_call), 'd3d')
    try:
        hw_ints = mdsvalue('d3d', treename='tscal', pulse=cal_set, TDI=hwi_call).data()
    except MDSplus.MdsException:
        printe('WARNING: Error reading Thomson scattering hardware map to determine which lenses were used!')
        return None
    else:
        if len(np.shape(hw_ints)) < 2:
            # Contingency needed for cases where all view-chords are taken off of divertor laser and reassigned to core
            hw_ints = hw_ints.reshape(1, -1)
        hw_lens = hw_ints[:, 2]
        return hw_lens


@machine_mapping_function(__all__)
def setup_thomson_scattering_hardware_description_d3d(ods, pulse=133221, revision='BLESSED'):
    """
    Gathers DIII-D Thomson measurement locations from MDSplus and loads them into OMAS

    :param revision: string
        Thomson scattering data revision, like 'BLESSED', 'REVISIONS.REVISION00', etc.

    :return: dict
        Information or instructions for follow up in central hardware description setup
    """
    printd('Setting up DIII-D Thomson locations...', 'd3d')

    tsdat = mdstree('d3d', treename='ELECTRONS', pulse=pulse)['TS'][revision]

    is_subsys = np.array([np.all([item in tsdat[k] for item in ['DENSITY', 'TEMP', 'R', 'Z']]) for k in list(tsdat.keys())])
    subsystems = np.array(list(tsdat.keys()))[is_subsys]

    i = 0
    for sub in subsystems:
        lenses = find_thomson_lens_d3d(pulse, sub, revision)
        try:
            nc = len(tsdat[sub]['R'].data())
        except (TypeError, KeyError):
            nc = 0
        for j in range(nc):
            ch = ods['thomson_scattering']['channel'][i]
            ch['name'] = 'TS_{sub:}_r{lens:+0d}_{ch:}'.format(sub=sub.lower(), ch=j, lens=lenses[j] if lenses is not None else -9)
            ch['identifier'] = '{}{:02d}'.format(sub[0], j)
            for pos in ['R', 'Z', 'PHI']:
                ch['position'][pos.lower()] = tsdat[sub][pos].data()[j] * (-np.pi / 180.0 if pos == 'PHI' else 1)
            i += 1
    return {}


@machine_mapping_function(__all__)
def setup_charge_exchange_hardware_description_d3d(ods, pulse=133221, analysis_type='CERQUICK'):
    """
    Gathers DIII-D CER measurement locations from MDSplus and loads them into OMAS

    :param analysis_type: string
        CER analysis quality level like CERQUICK, CERAUTO, or CERFIT.  CERQUICK is probably fine.

    :return: dict
        Information or instructions for follow up in central hardware description setup
    """
    printd('Setting up DIII-D CER locations...', 'd3d')

    cerdat = mdstree('d3d', treename='IONS', pulse=pulse)['CER'][analysis_type]

    subsystems = np.array([k for k in list(cerdat.keys()) if 'CHANNEL01' in list(cerdat[k].keys())])

    i = 0
    for sub in subsystems:
        try:
            channels = [k for k in list(cerdat[sub].keys()) if 'CHANNEL' in k]
        except (TypeError, KeyError):
            continue
        for j, channel in enumerate(channels):
            try:
                postime = cerdat[sub][channel]['TIME'].data()
            except Exception:
                continue
            inc = 0
            for pos in ['R', 'Z', 'VIEW_PHI']:
                posdat = cerdat[sub][channel][pos].data()
                if postime is not None:
                    inc = 1
                    ch = ods['charge_exchange']['channel'][i]
                    ch['name'] = 'imCERtang_{sub:}{ch:02d}'.format(sub=sub.lower()[0], ch=j + 1)
                    ch['identifier'] = '{}{:02d}'.format(sub[0], j + 1)
                    chpos = ch['position'][pos.lower().split('_')[-1]]
                    chpos['time'] = postime / 1000.0  # Convert ms to s
                    chpos['data'] = posdat * -np.pi / 180.0 if (pos == 'VIEW_PHI') and posdat is not None else posdat
            i += inc
    return {}


@machine_mapping_function(__all__)
def setup_langmuir_probes_hardware_description_d3d(ods, pulse=176235):
    """
    Load DIII-D Langmuir probe locations into an ODS

    :param ods: ODS instance

    :param pulse: int

    :return: dict
        Information or instructions for follow up in central hardware description setup
    """
    import MDSplus

    tdi = r'GETNCI("\\langmuir::top.probe_*.r", "LENGTH")'
    # "LENGTH" is the size of the data, I think (in bits?). Single scalars seem to be length 12.
    printd('Setting up Langmuir probes hardware description, pulse {}; checking availability, TDI={}'.format(pulse, tdi), 'd3d')
    m = mdsvalue('d3d', pulse=pulse, treename='LANGMUIR', TDI=tdi)
    try:
        data_present = m.data() > 0
    except MDSplus.MdsException:
        data_present = []
    nprobe = len(data_present)
    printd('Looks like up to {} Langmuir probes might have valid data for {}'.format(nprobe, pulse), 'd3d')
    j = 0
    for i in range(nprobe):
        if data_present[i]:
            try:
                r = mdsvalue('d3d', pulse=pulse, treename='langmuir', TDI=r'\langmuir::top.probe_{:03d}.r'.format(i)).data()
            except Exception:
                continue
            if r > 0:
                # Don't bother gathering more if r is junk
                z = mdsvalue('d3d', pulse=pulse, treename='langmuir', TDI=r'\langmuir::top.probe_{:03d}.z'.format(i)).data()
                pnum = mdsvalue('d3d', pulse=pulse, treename='langmuir', TDI=r'\langmuir::top.probe_{:03d}.pnum'.format(i)).data()
                label = mdsvalue('d3d', pulse=pulse, treename='langmuir', TDI=r'\langmuir::top.probe_{:03d}.label'.format(i)).data()
                printd('  Probe i={i:}, j={j:}, label={label:} passed the check; r={r:}, z={z:}'.format(**locals()), 'd3d')
                ods['langmuir_probes.embedded'][j]['position.r'] = r
                ods['langmuir_probes.embedded'][j]['position.z'] = z
                ods['langmuir_probes.embedded'][j]['position.phi'] = np.NaN  # Didn't find this in MDSplus
                ods['langmuir_probes.embedded'][j]['identifier'] = 'PROBE_{:03d}: PNUM={}'.format(i, pnum)
                ods['langmuir_probes.embedded'][j]['name'] = str(label).strip()
                j += 1
    return {}


if __name__ == '__main__':
    test_machine_mapping_functions(__all__, globals(), locals())
