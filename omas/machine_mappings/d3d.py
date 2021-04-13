import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd, unumpy
from omas.machine_mappings._common import *

__all__ = []


@machine_mapping_function(__all__)
def gas_injection_hardware(ods, pulse=133221):
    """
    Loads DIII-D gas injectors hardware geometry

    R and Z are from the tips of the arrows in puff_loc.pro; phi from angle listed in labels in puff_loc.pro .
    I recorded the directions of the arrows on the EFITviewer overlay, but I don't know how to include them in IMAS, so
    I commented them out.

    Warning: changes to gas injector configuration with time are not yet included. This is just the best picture I could
    make of the 2018 configuration.

    Data sources:
    EFITVIEWER: iris:/fusion/usc/src/idl/efitview/diagnoses/DIII-D/puff_loc.pro accessed 2018 June 05, revised 20090317
    DIII-D webpage: https://diii-d.gat.com/diii-d/Gas_Schematic accessed 2018 June 05
    DIII-D wegpage: https://diii-d.gat.com/diii-d/Gas_PuffLocations accessed 2018 June 05
    """
    if pulse < 100775:
        warnings.warn('DIII-D Gas valve locations not applicable for pulses earlier than 100775 (2000 JAN 17)')

    i = 0

    def pipe_copy(pipe_in):
        pipe_out = ods['gas_injection']['pipe'][i]
        for field in ['name', 'exit_position.r', 'exit_position.z', 'exit_position.phi']:
            pipe_out[field] = pipe_in[field]
        vvv = 0
        while f'valve.{vvv}.identifier' in pipe_in:
            valve_identifier = pipe_in[f'valve.{vvv}.identifier']
            vvv += 1
        return valve_identifier

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
    pipe_rfcomb['exit_position']['phi'] = np.nan  # Unknown, sorry
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
    pipe_cpmid['exit_position']['phi'] = np.nan  # Unknown, sorry
    pipe_cpmid['valve'][0]['identifier'] = '???'  # Seems to have been removed. Not on schematic.
    # pipe_cpmid['exit_position']['direction'] = 0.  # degrees, giving dir of pipe leading towards injector, up is 90
    i += 1


@machine_mapping_function(__all__)
def pf_active_hardware(ods):
    r"""
    Loads DIII-D tokamak poloidal field coil hardware geometry

    :param ods: ODS instance
    """
    # OMFITmhdin('/fusion/projects/codes/efit/Test/Green/example_2019/mhdin.dat', serverPicker='iris')
    # R        Z       dR      dZ    tilt1  tilt2
    # 0 in the last column really means 90 degrees
    # fmt: off
    fc_dat = np.array(
        [
            [0.8608, 0.16830, 0.0508, 0.32106, 0.0, 0.0],
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

    turns = [58, 58, 58, 58, 58, 55, 55, 58, 55, 58, 58, 58, 58, 58, 55, 55, 58, 55]

    ods = pf_coils_to_ods(ods, fc_dat)

    for k in range(len(fc_dat[:, 0])):
        fcid = 'F{}{}'.format((k % 9) + 1, 'AB'[int(fc_dat[k, 1] < 0)])
        ods['pf_active.coil'][k]['name'] = ods['pf_active.coil'][k]['identifier'] = fcid
        ods['pf_active.coil'][k]['element.0.identifier'] = fcid
        ods['pf_active.coil'][k]['element.0.turns_with_sign'] = turns[k]


@machine_mapping_function(__all__)
def pf_active_coil_current_data(ods, pulse=133221):
    # get pf_active hardware description --without-- placing the data in this ods
    # use `unwrap` to avoid calling `@machine_mapping_function` of `pf_active_hardware`
    ods1 = ODS()
    unwrap(pf_active_hardware)(ods1)

    # fetch the actual pf_active currents data
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
            mds_server='d3d',
            mds_tree='D3D',
            tdi_expression='ptdata2("{signal}",{pulse})',
            time_norm=0.001,
            data_norm=1.0,
        )


@machine_mapping_function(__all__)
def interferometer_hardware(ods, pulse=133221):
    """
    Loads DIII-D CO2 interferometer chord locations

    The chord endpoints ARE NOT RIGHT. Only the R for vertical lines or Z for horizontal lines is right.

    Data sources:
    DIII-D webpage: https://diii-d.gat.com/diii-d/Mci accessed 2018 June 07 by D. Eldon

    :param ods: an OMAS ODS instance

    :param pulse: int
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
        ch['line_of_sight.third_point'] = ch['line_of_sight.first_point']

    if pulse < 125406:
        printe(
            'DIII-D CO2 pointnames were different before pulse 125406. The physical locations of the chords seems to '
            'have been the same, though, so there has not been a problem yet (I think).'
        )


def thomson_scattering_hardware(ods, pulse=133221, revision='BLESSED'):
    """
    Gathers DIII-D Thomson measurement locations

    :param pulse: int

    :param revision: string
        Thomson scattering data revision, like 'BLESSED', 'REVISIONS.REVISION00', etc.
    """
    unwrap(thomson_scattering_data)(ods, pulse, revision, _measurements=False)


@machine_mapping_function(__all__)
def thomson_scattering_data(ods, pulse=133221, revision='BLESSED', _measurements=True):
    """
    Loads DIII-D Thomson measurement data

    :param pulse: int

    :param revision: string
        Thomson scattering data revision, like 'BLESSED', 'REVISIONS.REVISION00', etc.
    """
    systems = ['TANGENTIAL', 'DIVERTOR', 'CORE']

    # get the actual data
    query = {'calib_nums': f'.ts.{revision}.header.calib_nums'}
    for system in systems:
        for quantity in ['R', 'Z', 'PHI']:
            query[f'{system}_{quantity}'] = f'.TS.{revision}.{system}:{quantity}'
        if _measurements:
            for quantity in ['TEMP', 'TEMP_E', 'DENSITY', 'DENSITY_E', 'TIME']:
                query[f'{system}_{quantity}'] = f'.TS.{revision}.{system}:{quantity}'
    tsdat = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()

    # Read the Thomson scattering hardware map to figure out which lens each chord looks through
    cal_set = tsdat['calib_nums'][0]
    query = {}
    for system in systems:
        query[f'{system}_hwmapints'] = f'.{system}.hwmapints'
    hw_ints = mdsvalue('d3d', treename='TSCAL', pulse=cal_set, TDI=query).raw()

    # assign data in ODS
    i = 0
    for system in systems:
        if isinstance(tsdat[f'{system}_R'], Exception):
            continue
        nc = len(tsdat[f'{system}_R'])
        if not nc:
            continue

        # determine which lenses were used
        ints = hw_ints[f'{system}_hwmapints']
        if len(np.shape(ints)) < 2:
            # Contingency needed for cases where all view-chords are taken off of divertor laser and reassigned to core
            ints = ints.reshape(1, -1)
        lenses = ints[:, 2]

        # Assign data to ODS
        for j in range(nc):
            ch = ods['thomson_scattering']['channel'][i]
            ch['name'] = 'TS_{system}_r{lens:+0d}_{ch:}'.format(system=system.lower(), ch=j, lens=lenses[j] if lenses is not None else -9)
            ch['identifier'] = f'{system[0]}{j:02d}'
            ch['position']['r'] = tsdat[f'{system}_R'][j]
            ch['position']['z'] = tsdat[f'{system}_Z'][j]
            ch['position']['phi'] = -tsdat[f'{system}_PHI'][j] * np.pi / 180.0
            if _measurements:
                ch['n_e.time'] = tsdat[f'{system}_TIME'] / 1e3
                ch['n_e.data'] = unumpy.uarray(tsdat[f'{system}_DENSITY'][j], tsdat[f'{system}_DENSITY_E'][j])
                ch['t_e.time'] = tsdat[f'{system}_TIME'] / 1e3
                ch['t_e.data'] = unumpy.uarray(tsdat[f'{system}_TEMP'][j], tsdat[f'{system}_TEMP_E'][j])
            i += 1

def electrpn_cyclotron_emission_hardware(ods, pulse=133221, fast=False):
    """
    Gathers DIII-D Electron cyclotron emission locations

    :param pulse: int

    :param revision: string
        Thomson scattering data revision, like 'BLESSED', 'REVISIONS.REVISION00', etc.
    """
    unwrap(electron_cyclotron_emission_data)(ods, pulse, _measurements=False, fast=fast)


@machine_mapping_function(__all__)
def electron_cyclotron_emission_data(ods, pulse=133221, _measurements=True, fast=False):
    """
    Loads DIII-D Thomson measurement data

    :param pulse: int

    :param revision: string
        Thomson scattering data revision, like 'BLESSED', 'REVISIONS.REVISION00', etc.
    """
    fast = 'F' if fast else ''
    setup = '\\ECE::TOP.SETUP.'
    cal = '\\ECE::TOP.CAL%s.' % fast
    TECE = '\\ECE::TOP.TECE.TECE' + fast
    query = {}
    for node, quantities in zip([setup, cal], \
                             [['ECEPHI', 'ECETHETA', 'ECEZH', 'FREQ'],\
                             ['NUMCH']]):
        for quantity in quantities:
            query[quantity] = node + quantity
    query['TIME'] = f"dim_of({TECE + '01'})"
    ece_map = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()
    N_time = len(ece_map['TIME'])
    N_ch = ece_map['NUMCH'].item()
    if _measurements:
        query = {}
        for ich in range(1,N_ch+1):
            query[f'T{ich}'] = TECE + '{0:02d}'.format(ich)
        query['TIME'] = f"dim_of({TECE + '01'})"
    ece_data = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()
    # Read the Thomson scattering hardware map to figure out which lens each chord looks through
    # Not in mds+
    points = [{},{}]
    points[0]['r'] = 2.5
    points[1]['r'] = 0.8
    points[0]['phi'] = np.deg2rad(ece_map['ECEPHI']) 
    points[1]['phi'] = np.deg2rad(ece_map['ECEPHI'])
    dR = points[1]['r'] - points[0]['r']
    dz = np.sin(np.deg2rad(ece_map['ECETHETA']))
    points[0]['z'] = ece_map['ECEZH']
    points[1]['z'] = points[0]['z'] + dz
    for entry, point in zip([ods['ece']['line_of_sight']['first_point'], \
                             ods['ece']['line_of_sight']['second_point']], \
                            points):
        for key in point:
            entry[key] = point[key]
    f = np.zeros(N_time)
    # Assign data to ODS
    for ich in range(N_ch):
        ch = ods['ece']['channel'][ich]
        ch['name'] = 'ECE' + str(ich + 1)
        ch['identifier'] = TECE + '{0:02d}'.format(ich + 1)
        f[:] = ece_map['FREQ'][ich]
        ch['frequency']['data'] = f
        ch['time'] = ece_map['TIME']
        if _measurements:
            ch['t_e']['data']  = ece_data[f'T{ich + 1}'] * 1.e3

@machine_mapping_function(__all__)
def bolometer_hardware(ods, pulse=133221):
    """
    Load DIII-D bolometer chord locations

    Data sources:
    - iris:/fusion/usc/src/idl/efitview/diagnoses/DIII-D/bolometerpaths.pro
    - OMFIT-source/modules/_PCS_prad_control/SETTINGS/PHYSICS/reference/DIII-D/bolometer_geo , access 2018 June 11 by D. Eldon
    """
    printd('Setting up DIII-D bolometer locations...', topic='machine')

    # fmt: off
    if pulse < 91000:
        xangle = (
            np.array(
                [292.4, 288.35, 284.3, 280.25, 276.2, 272.15, 268.1, 264.87, 262.27, 259.67, 257.07, 254.47, 251.87, 249.27, 246.67, 243.81,
                 235.81, 227.81, 219.81, 211.81, 203.81, 195.81, 187.81, 179.8, 211.91, 206.41, 200.91, 195.41, 189.91, 184.41, 178.91,
                 173.41, 167.91, 162.41, 156.91, 156.3, 149.58, 142.86, 136.14, 129.77, 126.77, 123.77, 120.77, 117.77, 114.77, 111.77,
                 108.77, 102.25]
            )
            * np.pi
            / 180.0
        )  # Converted to rad

        xangle_width = None

        zxray = (
            np.array(
                [124.968, 124.968, 124.968, 124.968, 124.968, 124.968, 124.968, 124.968, 124.968, 124.968, 124.968, 124.968, 124.968,
                 124.968, 124.968, 129.87, 129.87, 129.87, 129.87, 129.87, 129.87, 129.87, 129.87, 129.87, -81.153, -81.153, -81.153,
                 -81.153, -81.153, -81.153, -81.153, -81.153, -81.153, -81.153, -81.153, -72.009, -72.009, -72.009, -72.009, -72.009,
                 -72.009, -72.009, -72.009, -72.009, -72.009, -72.009, -72.009, -72.009]
            )
            / 100.0
        )  # Converted to m

        rxray = (
            np.array(
                [196.771, 196.771, 196.771, 196.771, 196.771, 196.771, 196.771, 196.771, 196.771, 196.771, 196.771, 196.771, 196.771,
                 196.771, 196.771, 190.071, 190.071, 190.071, 190.071, 190.071, 190.071, 190.071, 190.071, 190.071, 230.72, 230.72,
                 230.72, 230.72, 230.72, 230.72, 230.72, 230.72, 230.72, 230.72, 230.72, 232.9, 232.9, 232.9, 232.9, 232.9, 232.9,
                 232.9, 232.9, 232.9, 232.9, 232.9, 232.9, 232.9]
            )
            / 100.0
        )  # Converted to m

    else:
        # There is a bigger step before the very last channel. Found in two different sources.
        xangle = (
            np.array(
                [269.4, 265.6, 261.9, 258.1, 254.4, 250.9, 247.9, 244.9, 241.9, 238.9, 235.9, 232.9, 228.0, 221.3, 214.5, 208.2, 201.1,
                 194.0, 187.7, 182.2, 176.7, 171.2, 165.7, 160.2, 213.7, 210.2, 206.7, 203.2, 199.7, 194.4, 187.4, 180.4, 173.4, 166.4,
                 159.4, 156.0, 149.2, 142.4, 135.8, 129.6, 126.6, 123.6, 120.6, 117.6, 114.6, 111.6, 108.6, 101.9]
            )
            * np.pi
            / 180.0
        )  # Converted to rad

        # Angular full width of the view-chord: calculations assume it's a symmetric cone.
        xangle_width = (
            np.array(
                [3.082, 3.206, 3.317, 3.414, 3.495, 2.866, 2.901, 2.928, 2.947, 2.957, 2.96, 2.955, 6.497, 6.342, 6.103, 6.331, 6.697,
                 6.979, 5.51, 5.553, 5.546, 5.488, 5.38, 5.223, 3.281, 3.348, 3.402, 3.444, 3.473, 6.95, 6.911, 6.768, 6.526, 6.188,
                 5.757, 5.596, 5.978, 6.276, 6.49, 2.979, 2.993, 2.998, 2.995, 2.984, 2.965, 2.938, 2.902, 6.183]
            )
            * np.pi
            / 180.0
        )

        zxray = (
            np.array(
                [72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817, 72.817,
                 82.332, 82.332, 82.332, 82.332, 82.332, 82.332, 82.332, 82.332, 82.332, -77.254, -77.254, -77.254, -77.254, -77.254,
                 -77.254, -77.254, -77.254, -77.254, -77.254, -77.254, -66.881, -66.881, -66.881, -66.881, -66.881, -66.881, -66.881,
                 -66.881, -66.881, -66.881, -66.881, -66.881, -66.881]
            )
            / 100.0
        )  # Converted to m

        rxray = (
            np.array(
                [234.881, 234.881, 234.881, 234.881, 234.881, 234.881, 234.881, 234.881, 234.881, 234.881, 234.881, 234.881, 234.881,
                 234.881, 234.881, 231.206, 231.206, 231.206, 231.206, 231.206, 231.206, 231.206, 231.206, 231.206, 231.894, 231.894,
                 231.894, 231.894, 231.894, 231.894, 231.894, 231.894, 231.894, 231.894, 231.894, 234.932, 234.932, 234.932, 234.932,
                 234.932, 234.932, 234.932, 234.932, 234.932, 234.932, 234.932, 234.932, 234.932]
            )
            / 100.0
        )  # Converted to m
    # fmt: on

    line_len = 3  # m  Make this long enough to go past the box for all chords.

    phi = np.array([60, 75])[(zxray > 0).astype(int)] * -np.pi / 180.0  # Convert to CCW radians
    fan = np.array(['Lower', 'Upper'])[(zxray > 0).astype(int)]
    fan_offset = np.array([0, int(len(rxray) // 2)])[(zxray < 0).astype(int)].astype(int)

    for i in range(len(zxray)):
        cnum = i + 1 - fan_offset[i]
        ods['bolometer']['channel'][i]['identifier'] = '{}{:02d}'.format(fan[i][0], cnum)
        ods['bolometer']['channel'][i]['name'] = '{} fan ch#{:02d}'.format(fan[i], cnum)
        cls = ods['bolometer']['channel'][i]['line_of_sight']  # Shortcut
        cls['first_point.r'] = rxray[i]
        cls['first_point.z'] = zxray[i]
        cls['first_point.phi'] = phi[i]
        cls['second_point.r'] = rxray[i] + line_len * np.cos(xangle[i])
        cls['second_point.z'] = zxray[i] + line_len * np.sin(xangle[i])
        cls['second_point.phi'] = cls['first_point.phi']

    return {'postcommands': ['trim_bolometer_second_points_to_box(ods)']}


@machine_mapping_function(__all__)
def langmuir_probes_hardware(ods, pulse=176235):
    """
    Load DIII-D Langmuir probe locations

    :param ods: ODS instance

    :param pulse: int
    """
    import MDSplus

    tdi = r'GETNCI("\\langmuir::top.probe_*.r", "LENGTH")'
    # "LENGTH" is the size of the data, I think (in bits?). Single scalars seem to be length 12.
    printd('Setting up Langmuir probes hardware description, pulse {}; checking availability, TDI={}'.format(pulse, tdi), topic='machine')
    m = mdsvalue('d3d', pulse=pulse, treename='LANGMUIR', TDI=tdi)
    try:
        data_present = m.data() > 0
    except MDSplus.MdsException:
        data_present = []
    nprobe = len(data_present)
    printd('Looks like up to {} Langmuir probes might have valid data for {}'.format(nprobe, pulse), topic='machine')
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
                printd('  Probe i={i:}, j={j:}, label={label:} passed the check; r={r:}, z={z:}'.format(**locals()), topic='machine')
                ods['langmuir_probes.embedded'][j]['position.r'] = r
                ods['langmuir_probes.embedded'][j]['position.z'] = z
                ods['langmuir_probes.embedded'][j]['position.phi'] = np.NaN  # Didn't find this in MDSplus
                ods['langmuir_probes.embedded'][j]['identifier'] = 'PROBE_{:03d}: PNUM={}'.format(i, pnum)
                ods['langmuir_probes.embedded'][j]['name'] = str(label).strip()
                j += 1


@machine_mapping_function(__all__)
def charge_exchange_hardware(ods, pulse=133221, analysis_type='CERQUICK'):
    """
    Gathers DIII-D CER measurement locations from MDSplus

    :param analysis_type: string
        CER analysis quality level like CERQUICK, CERAUTO, or CERFIT
    """
    import MDSplus

    printd('Setting up DIII-D CER locations...', topic='machine')

    cerdat = mdstree('d3d', 'IONS', pulse=pulse)['CER'][analysis_type]

    subsystems = np.array([k for k in list(cerdat.keys()) if 'CHANNEL01' in list(cerdat[k].keys())])

    # fetch
    TDIs = {}
    for sub in subsystems:
        try:
            channels = sorted([k for k in list(cerdat[sub].keys()) if 'CHANNEL' in k])
        except (TypeError, KeyError):
            continue
        for k, channel in enumerate(channels):
            for pos in ['TIME', 'R', 'Z', 'VIEW_PHI']:
                TDIs[f'{sub}_{channel}_{pos}'] = cerdat[sub][channel][pos].TDI
    data = mdsvalue('d3d', treename='IONS', pulse=pulse, TDI=TDIs).raw()

    # assign
    for sub in subsystems:
        try:
            channels = sorted([k for k in list(cerdat[sub].keys()) if 'CHANNEL' in k])
        except (TypeError, KeyError):
            continue
        for k, channel in enumerate(channels):
            postime = data[f'{sub}_{channel}_TIME']
            if isinstance(postime, Exception):
                continue
            ch = ods['charge_exchange.channel.+']
            for pos in ['R', 'Z', 'VIEW_PHI']:
                posdat = data[f'{sub}_{channel}_{pos}']
                ch['name'] = 'imCERtang_{sub:}{ch:02d}'.format(sub=sub.lower()[0], ch=k + 1)
                ch['identifier'] = '{}{:02d}'.format(sub[0], k + 1)
                chpos = ch['position'][pos.lower().split('_')[-1]]
                chpos['time'] = postime / 1000.0  # Convert ms to s
                chpos['data'] = posdat * -np.pi / 180.0 if pos == 'VIEW_PHI' and not isinstance(posdat, Exception) else posdat


@machine_mapping_function(__all__)
def magnetics_hardware(ods):
    r"""
    Load DIII-D tokamak flux loops and magnetic probes hardware geometry

    :param ods: ODS instance
    """
    # From  iris:/fusion/usc/src/idl/efitview/diagnoses/DIII-D/coils.dat
    # https://nomos.gat.com/DIII-D/diag/magnetics/magnetics.html
    # fmt: off
    R_flux_loop = [0.8929, 0.8936, 0.895, 0.8932, 1.0106, 2.511, 2.285, 1.2517,
                   1.6885, 0.8929, 0.8928, 0.8933, 0.8952, 1.0152, 2.509, 2.297,
                   1.2491, 1.6882, 0.9247, 0.9247, 0.9247, 0.9247, 0.9621, 1.1212,
                   1.6978, 2.209, 2.1933, 2.501, 2.4347, 0.9247, 0.9247, 0.9247,
                   0.9611, 1.1199, 1.8559, 2.215, 2.1933, 2.501, 2.4328, 1.3828,
                   1.5771, 1.3929, 1.5983, 1.7759]

    Z_flux_loop = [0.1683, 0.5092, 0.85, 1.1909, 1.4475, 0.332, 1.018,
                   1.5517, 1.5169, -0.1726, -0.5135, -0.8543, -1.1953, -1.4536,
                   -0.335, -1.01, -1.5527, -1.5075, 0., 0.3429, 0.6858,
                   1.0287, 1.3208, 1.4646, 1.4646, 1.213, 1.0629, 0.527,
                   0.4902, -0.3429, -0.6858, -1.0287, -1.3198, -1.4625, -1.4625,
                   -1.214, -1.0602, -0.524, -0.4909, -1.4625, -1.4625, -1.3057,
                   -1.3057, -1.3053]

    name_flux_loop = ['PSF1A', 'PSF2A', 'PSF3A', 'PSF4A',
                      'PSF5A', 'PSF6NA', 'PSF7NA', 'PSF8A',
                      'PSF9A', 'PSF1B', 'PSF2B', 'PSF3B',
                      'PSF4B', 'PSF5B', 'PSF6NB', 'PSF7NB',
                      'PSF8B', 'PSF9B', 'PSI11M', 'PSI12A',
                      'PSI23A', 'PSI34A', 'PSI45A', 'PSI58A',
                      'PSI9A', 'PSF7FA', 'PSI7A', 'PSF6FA',
                      'PSI6A', 'PSI12B', 'PSI23B', 'PSI34B',
                      'PSI45B', 'PSI58B', 'PSI9B', 'PSF7FB',
                      'PSI7B', 'PSF6FB', 'PSI6B', 'PSI89FB',
                      'PSI89NB', 'PSI1L', 'PSI2L', 'PSI3L']

    R_magnetic = [0.9729, 0.9787, 0.9726, 0.9767, 0.9793, 0.9764, 0.9785, 2.413,
                  1.7617, 2.2124, 2.2641, 2.2655, 2.3137, 2.4066, 2.4133, 0.9771,
                  0.9722, 0.9792, 0.9769, 0.9801, 0.9774, 2.4129, 0.9719, 2.0436,
                  2.2089, 2.2596, 2.2624, 2.3119, 2.414, 1.2194, 1.4017, 1.5841,
                  1.7825, 1.9243, 2.0673, 2.2193, 2.2702, 2.3189, 2.4159, 2.4182,
                  2.4162, 2.3154, 2.2631, 2.212, 2.0848, 1.894, 1.6989, 1.4769,
                  1.2541, 1.0479, 0.9724, 0.974, 0.9748, 0.9737, 0.9732, 0.9741,
                  0.9742, 0.9745, 0.9722, 1.0511, 1.445, 1.56, 1.723, 1.873,
                  1.506, 1.59, 1.73, 1.877, 1.218, 1.075, 0.976, 1.185,
                  1.086, 1.5104, 1.6103, 1.7103]

    Z_magnetic = [-3.7000e-03, 6.9900e-02, 5.1800e-01, 2.0940e-01, 3.4710e-01,
                  4.8510e-01, 7.5910e-01, -2.5000e-03, 1.3123e+00, 8.7320e-01,
                  7.5980e-01, 7.5780e-01, 6.2850e-01, 2.5440e-01, 2.9000e-03,
                  -6.9100e-02, -5.1650e-01, -2.0700e-01, -3.4280e-01, -4.8180e-01,
                  -7.5700e-01, -1.0000e-03, -1.8310e-01, -1.1098e+00, -8.6690e-01,
                  -7.5380e-01, -7.5100e-01, -6.2020e-01, -2.4410e-01, 1.4055e+00,
                  1.4070e+00, 1.4084e+00, 1.3230e+00, 1.2064e+00, 1.0904e+00,
                  8.6980e-01, 7.4620e-01, 6.2280e-01, 2.4920e-01, -9.0000e-04,
                  -2.4380e-01, -6.2400e-01, -7.4870e-01, -8.7250e-01, -1.1020e+00,
                  -1.3327e+00, -1.4063e+00, -1.4055e+00, -1.4047e+00, -1.3302e+00,
                  -1.1589e+00, -8.5420e-01, -5.1220e-01, -1.8700e-01, -2.3000e-03,
                  1.8170e-01, 5.1160e-01, 8.5030e-01, 1.1609e+00, 1.3304e+00,
                  1.2770e+00, 1.1870e+00, 1.1120e+00, 1.1160e+00, 1.2700e+00,
                  1.1980e+00, 1.1420e+00, 1.1330e+00, 1.2980e+00, 1.2020e+00,
                  9.7900e-01, 1.2860e+00, 1.2280e+00, -1.2915e+00, -1.2915e+00,
                  -1.2918e+00]

    A_magnetic = [90., 90.26, 89.8, 89.87, 90.07, 90.01,
                  89.74, -90.5, -39.5, -67.55, -67.8, -67.6,
                  -67.59, -89.39, -89.9, 90.01, 90.3, 89.59,
                  90.05, 89.86, 89.79, -89.78, 89.55, -129.1,
                  -112.57, -112.8, -112.6, -113., -89.58, 0.3,
                  0.667, 0.617, -39.283, -39.167, -39.417, -68.1,
                  -67.9, -68., -89.333, -89.9, 269.25, -113.4,
                  -112.8, -113.5, 230.817, 230.467, 179.867, 180.083,
                  180.033, 136.383, 89.567, 89.75, 90.267, 90.117,
                  89.933, 90.017, 89.7, 90.15, 90.483, 44.617,
                  -40.55, -40.77, -3.3, -2.75, 49.53, 49.53,
                  92.3, 90.5, 66.303, 0.729, 90.259, 156.55,
                  90.4, 180.01, 180.25, 181.03]

    S_magnetic = [0.1408, 0.1419, 0.1404, 0.1408, 0.1412, 0.1409, 0.1414,
                  0.1402, 0.1534, 0.1385, 0.1561, 0.155, 0.1399, 0.1406,
                  0.1409, 0.1418, 0.1392, 0.1414, 0.1411, 0.1407, 0.142,
                  0.1375, 0.1408, 0.1537, 0.1422, 0.1549, 0.1557, 0.1398,
                  0.1407, 0.1394, 0.1145, 0.1403, 0.1409, 0.1141, 0.1398,
                  0.1403, 0.1407, 0.1402, 0.1407, 0.1399, 0.1399, 0.1403,
                  0.14, 0.1407, 0.1403, 0.1406, 0.1415, 0.1403, 0.1403,
                  0.1403, 0.1405, 0.1402, 0.1404, 0.14, 0.1153, 0.14,
                  0.1403, 0.1404, 0.1404, 0.1407, 0.027, 0.027, 0.027,
                  0.054, -0.094, -0.088, -0.106, -0.1955, 0.027, 0.027,
                  0.027, -0.091, -0.107, 0.0271, 0.0277, 0.0273]

    name_magnetic = ['MPI11M067', 'MPI1A139', 'MPI2A067', 'MPI2A139',
                     'MPI3A139', 'MPI4A139', 'MPI5A139', 'MPI66M247',
                     'MPI79A147', 'MPI7NA142', 'MPI67A3', 'MPI67A067',
                     'MPI6FA142', 'MPI6NA157', 'MPI66M067', 'MPI1B139',
                     'MPI2B067', 'MPI2B139', 'MPI3B139', 'MPI4B139',
                     'MPI5B139', 'MPI66M157', 'MPI1B157', 'MPI79B142',
                     'MPI7NB142', 'MPI67B3', 'MPI67B067', 'MPI6FB142',
                     'MPI6NB157', 'MPI8A322', 'MPI89A322', 'MPI9A322',
                     'MPI79FA322', 'MPI79NA322', 'MPI7FA322', 'MPI7NA322',
                     'MPI67A322', 'MPI6FA322', 'MPI6NA322', 'MPI66M322',
                     'MPI6NB322', 'MPI6FB322', 'MPI67B322', 'MPI7NB322',
                     'MPI7FB322', 'MPI79B322', 'MPI9B322', 'MPI89B322',
                     'MPI8B322', 'MPI5B322', 'MPI4B322', 'MPI3B322',
                     'MPI2B322', 'MPI1B322', 'MPI11M322', 'MPI1A322',
                     'MPI2A322', 'MPI3A322', 'MPI4A322', 'MPI5A322', 'MPI1U157',
                     'MPI2U157', 'MPI3U157', 'MPI4U157', 'DSL1U180', 'DSL2U180',
                     'DSL3U180', 'DSL4U157', 'MPI5U157', 'MPI6U157', 'MPI7U157',
                     'DSL5U157', 'DSL6U157', 'MPI1L180', 'MPI2L180', 'MPI3L180']
    # fmt: on

    with omas_environment(ods, cocosio=1):
        for k, (r, z, name) in enumerate(zip(R_flux_loop, Z_flux_loop, name_flux_loop)):
            ods[f'magnetics.flux_loop.{k}.identifier'] = ods[f'magnetics.flux_loop.{k}.name'] = name
            ods[f'magnetics.flux_loop.{k}.position[0].r'] = r
            ods[f'magnetics.flux_loop.{k}.position[0].z'] = z
            ods[f'magnetics.flux_loop.{k}.type.index'] = 1

        for k, (r, z, a, s, name) in enumerate(zip(R_magnetic, Z_magnetic, A_magnetic, S_magnetic, name_magnetic)):
            ods[f'magnetics.b_field_pol_probe.{k}.identifier'] = ods[f'magnetics.b_field_pol_probe.{k}.name'] = name
            ods[f'magnetics.b_field_pol_probe.{k}.position.r'] = r
            ods[f'magnetics.b_field_pol_probe.{k}.position.z'] = z
            ods[f'magnetics.b_field_pol_probe.{k}.length'] = s
            ods[f'magnetics.b_field_pol_probe.{k}.poloidal_angle'] = -a / 180 * np.pi
            ods[f'magnetics.b_field_pol_probe.{k}.toroidal_angle'] = 0.0 / 180 * np.pi
            ods[f'magnetics.b_field_pol_probe.{k}.type.index'] = 1
            ods[f'magnetics.b_field_pol_probe.{k}.turns'] = 1


@machine_mapping_function(__all__)
def magnetics_floops_data(ods, pulse=133221):
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
            mds_server='d3d',
            mds_tree='D3D',
            tdi_expression='ptdata2("{signal}",{pulse})',
            time_norm=0.001,
            data_norm=1.0,
        )


@machine_mapping_function(__all__)
def magnetics_probes_data(ods, pulse=133221):
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
            mds_server='d3d',
            mds_tree='D3D',
            tdi_expression='ptdata2("{signal}",{pulse})',
            time_norm=0.001,
            data_norm=1.0,
        )


def unit_test_ece_ods():
    ods = ODS()
    electron_cyclotron_emission_data(ods, pulse=185069, fast=False)
    import matplotlib.pyplot as plt
    plt.plot([ods['ece']["line_of_sight"]["first_point"]["r"], \
              ods['ece']["line_of_sight"]["second_point"]["r"]], \
             [ods['ece']["line_of_sight"]["first_point"]["z"], \
              ods['ece']["line_of_sight"]["second_point"]["z"]])
    fig = plt.figure()
    plt.plot(ods['ece']['channel'][12]["time"], ods['ece']['channel'][12]["t_e"]['data'])
    plt.show()

if __name__ == '__main__':
    unit_test_ece_ods()
    # run_machine_mapping_functions(__all__, globals(), locals())
