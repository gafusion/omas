import os
import numpy as np
from inspect import unwrap

from numpy.lib.function_base import iterable
from omas import *
from omas.omas_utils import printd, unumpy
from omas.machine_mappings._common import *

__all__ = []
__regression_arguments__ = {'__all__': __all__}

# ================================
@machine_mapping_function(__regression_arguments__, pulse=133221)
def gas_injection_hardware(ods, pulse):
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


# ================================
@machine_mapping_function(__regression_arguments__)
def pf_active_hardware(ods, pulse):
    r"""
    Loads DIII-D tokamak poloidal field coil hardware geometry

    :param ods: ODS instance
    """
    from omfit_classes.omfit_efund import OMFITmhdin

    mhdin = get_support_file(OMFITmhdin, support_filenames('d3d', 'mhdin', pulse))
    mhdin.to_omas(ods, update='pf_active')

    coil_names = [
        'F1A',
        'F2A',
        'F3A',
        'F4A',
        'F5A',
        'F6A',
        'F7A',
        'F8A',
        'F9A',
        'F2B',
        'F2B',
        'F3B',
        'F4B',
        'F5B',
        'F6B',
        'F7B',
        'F8B',
        'F9B',
        'ECOILA',
        'ECOILB',
        'E567UP',
        'E567DN',
        'E89DN',
        'E89UP',
    ]
    for k, fcid in enumerate(coil_names):
        ods['pf_active.coil'][k]['name'] = fcid
        ods['pf_active.coil'][k]['identifier'] = fcid
        ods['pf_active.coil'][k]['element.0.name'] = fcid
        ods['pf_active.coil'][k]['element.0.identifier'] = fcid


@machine_mapping_function(__regression_arguments__, pulse=133221)
def pf_active_coil_current_data(ods, pulse):
    # get pf_active hardware description --without-- placing the data in this ods
    # use `unwrap` to avoid calling `@machine_mapping_function` of `pf_active_hardware`
    ods1 = ODS()
    unwrap(pf_active_hardware)(ods1, pulse)

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
            homogeneous_time=False,
        )

        # fetch uncertainties
        TDIs = {}
        for k in ods1['pf_active.coil']:
            identifier = ods1[f'pf_active.coil.{k}.identifier'].upper()
            TDIs[identifier] = f'pthead2("{identifier}",{pulse}), __rarray'

        data = mdsvalue('d3d', None, pulse, TDIs).raw()
        for k in ods1['pf_active.coil']:
            identifier = ods1[f'pf_active.coil.{k}.identifier'].upper()
            nt = len(ods[f'pf_active.coil.{k}.current.data'])
            ods[f'pf_active.coil.{k}.current.data_error_upper'] = abs(data[identifier][3] * data[identifier][4]) * np.ones(nt) * 10.0


# ================================
@machine_mapping_function(__regression_arguments__)
def coils_non_axisymmetric_hardware(ods, pulse):
    r"""
    Loads DIII-D tokamak non_axisymmetric field coil hardware geometry

    :param ods: ODS instance
    """

    from omfit_classes.omfit_omas_d3d import OMFITd3dcompfile

    coil_names = []
    for compfile in ['ccomp', 'icomp']:
        comp = get_support_file(OMFITd3dcompfile, support_filenames('d3d', compfile, pulse))
        compshot = -1
        for shot in comp:
            if pulse > compshot:
                compshot = shot

        coil_names += list(comp[compshot].keys())

    for k, fcid in enumerate(coil_names):
        ods['coils_non_axisymmetric.coil'][k]['name'] = fcid
        ods['coils_non_axisymmetric.coil'][k]['identifier'] = fcid


@machine_mapping_function(__regression_arguments__, pulse=133221)
def coils_non_axisymmetric_current_data(ods, pulse):
    # get pf_active hardware description --without-- placing the data in this ods
    # use `unwrap` to avoid calling `@machine_mapping_function` of `pf_active_hardware`
    ods1 = ODS()
    unwrap(coils_non_axisymmetric_hardware)(ods1, pulse)

    # fetch the actual pf_active currents data
    with omas_environment(ods):
        fetch_assign(
            ods,
            ods1,
            pulse,
            channels='coils_non_axisymmetric.coil',
            identifier='coils_non_axisymmetric.coil.{channel}.identifier',
            time='coils_non_axisymmetric.coil.{channel}.current.time',
            data='coils_non_axisymmetric.coil.{channel}.current.data',
            validity=None,
            mds_server='d3d',
            mds_tree='D3D',
            tdi_expression='ptdata2("{signal}",{pulse})',
            time_norm=0.001,
            data_norm=1.0,
        )


# ================================
@machine_mapping_function(__regression_arguments__, pulse=170325)
def ec_launcher_active_hardware(ods, pulse):
    setup = '.ECH.'
    # We need three queries in order to retrieve only the fields we need
    # First the amount of systems in use
    query = {'NUM_SYSTEMS': setup + 'NUM_SYSTEMS'}
    num_systems = mdsvalue('d3d', treename='RF', pulse=pulse, TDI=query).raw()['NUM_SYSTEMS']
    query = {}
    # Second query the used systems to resolve the gyrotron names
    for system_no in range(1, num_systems + 1):
        cur_system = f'SYSTEM_{system_no}.'
        query[f'GYROTRON_{system_no}'] = setup + cur_system + 'GYROTRON.NAME'
        query[f'FREQUENCY_{system_no}'] = setup + cur_system + 'GYROTRON.FREQUENCY'
        for field in ['LAUNCH_R', 'LAUNCH_Z', 'PORT']:
            query[field + f'_{system_no}'] = setup + cur_system + f'antenna.{field}'
    systems = mdsvalue('d3d', treename='RF', pulse=pulse, TDI=query).raw()
    query = {}
    gyrotron_names = []
    for system_no in range(1, num_systems + 1):
        if len(systems[f'GYROTRON_{system_no}']) == 0:
            """
            If nothing is connected to this system the gyrotron name is blank.
            """
            continue
        gyrotron = systems[f'GYROTRON_{system_no}']
        gyrotron_names.append(gyrotron)
        gyr = gyrotron.upper()
        gyr = gyr[:3]
        for field in ['STAT', 'XMFRAC', 'FPWRC', 'AZIANG', 'POLANG']:
            query[field + f'_{system_no}'] = setup + f'{gyrotron.upper()}.EC{gyr}{field}'
            if field in ['XMFRAC', 'FPWRC', 'AZIANG', 'POLANG']:
                query["TIME_" + field + f'_{system_no}'] = "dim_of(" + query[field + f'_{system_no}'] + "+01)"
    # Final, third query now that we have resolved all the TDIs related to gyrotron names
    gyrotrons = mdsvalue('d3d', treename='RF', pulse=pulse, TDI=query).raw()
    for system_no in range(1, num_systems + 1):
        system_index = system_no - 1
        if gyrotrons[f'STAT_{system_no}'] == 0:
            continue
        beams = ods['ec_launchers.beam']
        time = np.atleast_1d(gyrotrons[f'TIME_AZIANG_{system_no}']) / 1.0e3
        if len(time) == 1:
            beams[system_index]['time'] = np.atleast_1d(0)
        else:
            beams[system_index]['time'] = np.atleast_1d(gyrotrons[f'TIME_AZIANG_{system_no}']) / 1.0e3
        ntime = len(beams[system_index]['time'])
        beams[system_index]['steering_angle_tor'] = np.atleast_1d(np.deg2rad((gyrotrons[f'AZIANG_{system_no}'] - 180.0)))
        beams[system_index]['steering_angle_pol'] = np.atleast_1d(np.deg2rad((gyrotrons[f'POLANG_{system_no}'] - 90.0)))

        beams[system_index]['identifier'] = beams[system_index]['name'] = gyrotron_names[system_index]

        beams[system_index]['launching_position.r'] = np.atleast_1d(systems[f'LAUNCH_R_{system_no}'])[0] * np.ones(ntime)
        beams[system_index]['launching_position.z'] = np.atleast_1d(systems[f'LAUNCH_Z_{system_no}'])[0] * np.ones(ntime)

        phi = np.deg2rad(float(systems[f'PORT_{system_no}'].split(' ')[0]))
        phi = -phi - np.pi / 2.0
        beams[system_index]['launching_position.phi'] = phi * np.ones(ntime)

        beams[system_index]['frequency.data'] = np.atleast_1d(systems[f'FREQUENCY_{system_no}'])
        beams[system_index]['frequency.time'] = np.atleast_1d(0)

        beams[system_index]['power_launched.time'] = np.atleast_1d(gyrotrons[f'TIME_FPWRC_{system_no}']) / 1.0e3
        beams[system_index]['power_launched.data'] = np.atleast_1d(gyrotrons[f'FPWRC_{system_no}'])

        xfrac = gyrotrons[f'XMFRAC_{system_no}']

        if iterable(xfrac):
            beams[system_index]['mode'] = int(np.round(1.0 - 2.0 * xfrac)[0])
        else:
            beams[system_index]['mode'] = int(np.round(1.0 - 2.0 * xfrac))

        # The spot size and radius are computed using the evolution formula for Gaussian beams
        # see: https://en.wikipedia.org/wiki/Gaussian_beam#Beam_waist
        # The beam is divergent because the beam waist is focused on to the final launching mirror.
        # The values of the ECRH group are the beam waist w_0 = 1.72 cm and
        # the beam is focused onto the mirror meaning that it is paraxial at the launching point.
        # Hence, the inverse curvature radius is zero
        # Notably the ECRH beams are astigmatic in reality so this is just an approximation
        beams[system_index]['phase.angle'] = np.zeros(ntime)
        beams[system_index]['phase.curvature'] = np.zeros([2, ntime])
        beams[system_index]['spot.angle'] = np.zeros(ntime)
        beams[system_index]['spot.size'] = 0.0172 * np.ones([2, ntime])


# ================================
@machine_mapping_function(__regression_arguments__, pulse=133221)
def interferometer_hardware(ods, pulse):
    """
    Loads DIII-D CO2 interferometer chord locations

    The chord endpoints are approximative. They do not take into account 
    the vessel wall contour of the shot. The values have been taken from OMFITprofiles.

    Data sources:
    DIII-D webpage: https://diii-d.gat.com/diii-d/Mci accessed 2018 June 07 by D. Eldon

    :param ods: an OMAS ODS instance

    :param pulse: int
    """

    # As of 2018 June 07, DIII-D has four interferometers
    # phi angles are compliant with odd COCOS
    ods['interferometer.channel.0.identifier'] = ods['interferometer.channel.0.name'] = 'r0'
    r0 = ods['interferometer.channel.0.line_of_sight']
    r0['first_point.phi'] = r0['second_point.phi'] = 225 * (-np.pi / 180.0)
    r0['first_point.r'], r0['second_point.r'] = 2.36, 1.01  # End points from OMFITprofiles
    r0['first_point.z'] = r0['second_point.z'] = 0.0
    Z_top = 1.24
    Z_bottom = -1.375
    for i, r in enumerate([1.48, 1.94, 2.10]):
        ods['interferometer.channel'][i + 1]['identifier'] = ods['interferometer.channel'][i + 1]['name'] = 'v{}'.format(i + 1)
        los = ods['interferometer.channel'][i + 1]['line_of_sight']
        los['first_point.phi'] = los['second_point.phi'] = 240 * (-np.pi / 180.0)
        los['first_point.r'] = los['second_point.r'] = r
        los['first_point.z'], los['second_point.z'] = Z_top, Z_bottom  # End points from OMFITprofiles

    for i in range(len(ods['interferometer.channel'])):
        ch = ods['interferometer.channel'][i]
        ch['line_of_sight.third_point'] = ch['line_of_sight.first_point']


@machine_mapping_function(__regression_arguments__, pulse=133221)
def interferometer_data(ods, pulse):
    """
    Loads DIII-D interferometer measurement data

    :param pulse: int
    """
    from scipy.interpolate import interp1d

    ods1 = ODS()
    unwrap(interferometer_hardware)(ods1, pulse=pulse)

    # fetch
    TDIs = {}
    for k, channel in enumerate(ods1['interferometer.channel']):
        identifier = ods1[f'interferometer.channel.{k}.identifier'].upper()
        TDIs[identifier] = f"\\BCI::TOP.DEN{identifier}"
        TDIs[f'{identifier}_validity'] = f"\\BCI::TOP.STAT{identifier}"
    TDIs['time'] = f"dim_of({TDIs['R0']})"
    TDIs['time_valid'] = f"dim_of({TDIs['R0_validity']})"
    data = mdsvalue('d3d', 'BCI', pulse, TDIs).raw()
    # assign
    for k, channel in enumerate(ods1['interferometer.channel']):
        identifier = ods1[f'interferometer.channel.{k}.identifier'].upper()
        ods[f'interferometer.channel.{k}.n_e_line.time'] = data['time'] / 1.0e3
        ods[f'interferometer.channel.{k}.n_e_line.data'] = data[identifier] * 1e6
        ods[f'interferometer.channel.{k}.n_e_line.validity_timed'] = interp1d(
            data['time_valid'] / 1.0e3,
            -data[f'{identifier}_validity'],
            kind='nearest',
            bounds_error=False,
            fill_value=(-data[f'{identifier}_validity'][0], -data[f'{identifier}_validity'][-1]),
            assume_sorted=True,
        )(ods[f'interferometer.channel.{k}.n_e_line.time'])


# ================================
@machine_mapping_function(__regression_arguments__, pulse=133221)
def thomson_scattering_hardware(ods, pulse, revision='BLESSED'):
    """
    Gathers DIII-D Thomson measurement locations

    :param pulse: int

    :param revision: string
        Thomson scattering data revision, like 'BLESSED', 'REVISIONS.REVISION00', etc.
    """
    unwrap(thomson_scattering_data)(ods, pulse, revision, _get_measurements=False)


@machine_mapping_function(__regression_arguments__, pulse=133221)
def thomson_scattering_data(ods, pulse, revision='BLESSED', _get_measurements=True):
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
        if _get_measurements:
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
            if not _get_measurements:
                ch['name'] = 'TS_{system}_r{lens:+0d}_{ch:}'.format(
                    system=system.lower(), ch=j, lens=lenses[j] if lenses is not None else -9
                )
                ch['identifier'] = f'{system[0]}{j:02d}'
                ch['position']['r'] = tsdat[f'{system}_R'][j]
                ch['position']['z'] = tsdat[f'{system}_Z'][j]
                ch['position']['phi'] = -tsdat[f'{system}_PHI'][j] * np.pi / 180.0
            else:
                ch['n_e.time'] = tsdat[f'{system}_TIME'] / 1e3
                ch['n_e.data'] = unumpy.uarray(tsdat[f'{system}_DENSITY'][j], tsdat[f'{system}_DENSITY_E'][j])
                ch['t_e.time'] = tsdat[f'{system}_TIME'] / 1e3
                ch['t_e.data'] = unumpy.uarray(tsdat[f'{system}_TEMP'][j], tsdat[f'{system}_TEMP_E'][j])
            i += 1


# ================================
@machine_mapping_function(__regression_arguments__, pulse=133221)
def electron_cyclotron_emission_hardware(ods, pulse, fast_ece=False):
    """
    Gathers DIII-D Electron cyclotron emission locations

    :param pulse: int

    :param fast_ece: bool
        Use data sampled at high frequency
    """
    unwrap(electron_cyclotron_emission_data)(ods, pulse, fast_ece=fast_ece, _measurements=False)


@machine_mapping_function(__regression_arguments__, pulse=133221)
def electron_cyclotron_emission_data(ods, pulse=133221, fast_ece=False, _measurements=True):
    """
    Loads DIII-D Electron cyclotron emission data

    :param pulse: int

    :param fast_ece: bool
            Use data sampled at high frequency
    """
    fast_ece = 'F' if fast_ece else ''
    setup = '\\ECE::TOP.SETUP.'
    cal = '\\ECE::TOP.CAL%s.' % fast_ece
    TECE = '\\ECE::TOP.TECE.TECE' + fast_ece

    query = {}
    for node, quantities in zip([setup, cal], [['ECEPHI', 'ECETHETA', 'ECEZH', 'FREQ'], ['NUMCH']]):
        for quantity in quantities:
            query[quantity] = node + quantity
    query['TIME'] = f"dim_of({TECE + '01'})"
    ece_map = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()
    N_time = len(ece_map['TIME'])
    N_ch = ece_map['NUMCH'].item()

    if _measurements:
        query = {}
        for ich in range(1, N_ch + 1):
            query[f'T{ich}'] = TECE + '{0:02d}'.format(ich)
        ece_data = mdsvalue('d3d', treename='ELECTRONS', pulse=pulse, TDI=query).raw()

    # Not in mds+
    if not _measurements:
        points = [{}, {}]
        points[0]['r'] = 2.5
        points[1]['r'] = 0.8
        points[0]['phi'] = np.deg2rad(ece_map['ECEPHI'])
        points[1]['phi'] = np.deg2rad(ece_map['ECEPHI'])
        dz = np.sin(np.deg2rad(ece_map['ECETHETA']))
        points[0]['z'] = ece_map['ECEZH']
        points[1]['z'] = points[0]['z'] + dz
        for entry, point in zip([ods['ece.line_of_sight.first_point'], ods['ece.line_of_sight.second_point']], points):
            for key in point:
                entry[key] = point[key]

    # Assign data to ODS
    f = np.zeros(N_time)
    for ich in range(N_ch):
        ch = ods['ece']['channel'][ich]
        if _measurements:
            ch['t_e']['data'] = ece_data[f'T{ich + 1}'] * 1.0e3
        else:
            ch['name'] = 'ECE' + str(ich + 1)
            ch['identifier'] = TECE + '{0:02d}'.format(ich + 1)
            ch['time'] = ece_map['TIME'] * 1.0e-3
            f[:] = ece_map['FREQ'][ich]
            ch['frequency']['data'] = f * 1.0e9


@machine_mapping_function(__regression_arguments__, pulse=133221)
def bolometer_hardware(ods, pulse):
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

        etendue = [ 3.0206e4, 2.9034e4, 2.8066e4, 2.7273e4, 2.6635e4, 4.0340e4, 3.9855e4, 3.9488e4, 3.9235e4, 3.9091e4, 3.9055e4, 3.9126e4, 0.7972e4,
                     0.8170e4, 0.8498e4, 0.7549e4, 0.7129e4, 0.6854e4, 1.1162e4, 1.1070e4, 1.1081e4, 1.1196e4, 1.1419e4, 1.1761e4, 2.9321e4, 2.8825e4,
                     2.8449e4, 2.8187e4, 2.8033e4, 0.7058e4, 0.7140e4, 0.7334e4, 0.7657e4, 0.8136e4, 0.8819e4, 0.7112e4, 0.6654e4, 0.6330e4, 0.6123e4,
                     2.9621e4, 2.9485e4, 2.9431e4, 2.9458e4, 2.9565e4, 2.9756e4, 3.0032e4, 3.0397e4, 0.6406e4, ]

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
        ods['bolometer']['channel'][i]['etendue'] = etendue[i]

        '''The etendue is used as follows:
        The fundamental profile is radiated power in W/cm**3
        Along a sightline integral this would be int(Prad,dl) W/cm**2
        However the bolometer collects a wider angle and intgrates
        over a volume.  The GAprofiles tools use a monte-carlo response
        grid on 2D R,Z EFIT domain.  This can be approximated by
        the detector etendue.

        The etendue for each channel defined as 4*pi*L^2/Ad/Aa/cos
        where L is the distance from detector to aperture,
        Ad is detector area, Aa is aperture area and cos is the
        angle between the detector and aperture normal vectors.
        and has units of cm**-2.  Thus a line integrated radiated
        power int(Prad,dl) in cm**-2 needs to be divided by the
        etendue to compare to reported power in Watts.'''

    return {'postcommands': ['trim_bolometer_second_points_to_box(ods)']}


@machine_mapping_function(__regression_arguments__, pulse=149472)
def bolometer_data(ods, pulse):
    """
    Load DIII-D bolometer data

    """
    printd('Setting up DIII-D bolometer data...', topic='machine')

    ods1 = ODS()
    unwrap(bolometer_hardware)(ods1, pulse)

    # first get the list of signals that we want to fetch
    TDIs = {}
    for ch in ods1['bolometer.channel']:
        ch_name = ods1[f'bolometer.channel[{ch}].identifier']
        TDI = f'\\BOLOM::TOP.PRAD_01.POWER.BOL_{ch_name}_P'
        TDIs[f'{ch}_data'] = f"data({TDI})"
        TDIs[f'{ch}_time'] = f"dim_of({TDI},0)"

    # then fetch all the data for all signals
    all_data = mdsvalue('d3d', 'BOLOM', pulse, TDIs).raw()

    # assign the data to the ods
    for ch in ods1['bolometer.channel']:
        data = all_data[f'{ch}_data']
        error = data * 0.2
        error[error < 1e-5] = 1e-5
        time = all_data[f'{ch}_time']
        ods[f'bolometer.channel[{ch}].power.data'] = data
        ods[f'bolometer.channel[{ch}].power.data_error_upper'] = error
        ods[f'bolometer.channel[{ch}].power.time'] = time / 1e3


# ================================
@machine_mapping_function(__regression_arguments__, pulse=176235)
def langmuir_probes_hardware(ods, pulse):
    """
    Load DIII-D Langmuir probe locations without the probe's measurements

    :param ods: ODS instance

    :param pulse: int
        For a workable example, see 176235

    Data are written into ods instead of being returned.
    """

    unwrap(langmuir_probes_data)(ods, pulse, _get_measurements=False)


@machine_mapping_function(__regression_arguments__, pulse=176235)
def langmuir_probes_data(ods, pulse, _get_measurements=True):
    """
    Gathers DIII-D Langmuir probe measurements and loads them into an ODS

    :param ods: ODS instance

    :param pulse: int
        For example, see 176235

    :param _get_measurements: bool
        Gather measurements from the probes, like saturation current, in addition to the hardware

    Data are written into ods instead of being returned.
    """
    import MDSplus

    tdi = r'GETNCI("\\langmuir::top.probe_*.r", "LENGTH")'
    # "LENGTH" is the size of the data, I think (in bits?). Single scalars seem to be length 12.
    printd(
        f'Setting up Langmuir probes {"data" if _get_measurements else "hardware description"}, '
        f'pulse {pulse}; checking availability, TDI={tdi}',
        topic='machine',
    )
    m = mdsvalue('d3d', pulse=pulse, treename='LANGMUIR', TDI=tdi)
    try:
        data_present = m.data() > 0
    except MDSplus.MdsException:
        data_present = []
    nprobe = len(data_present)
    printd('Looks like up to {nprobe} Langmuir probes might have valid data for DIII-D#{pulse}', topic='machine')
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
                if _get_measurements:
                    t = mdsvalue('d3d', pulse=pulse, treename='langmuir', TDI=rf'\langmuir::top.probe_{i:03d}.time').data()
                    ods['langmuir_probes.embedded'][j]['time'] = t

                    nodes = dict(
                        isat='ion_saturation_current',
                        dens='n_e',
                        area='surface_area_effective',
                        temp='t_e',
                        angle='b_field_angle',
                        pot='v_floating',
                        heatflux='heat_flux_parallel',
                    )
                    # Unit conversions: DIII-D MDS --> imas
                    unit_conversions = dict(
                        dens=1e6,  # cm^-3 --> m^-3   (MDSplus reports the units as 1e13 cm^-3, but this can't be)
                        isat=1,  # A --> A
                        temp=1,  # eV --> eV
                        area=1e-4,  # cm^2 --> m^2
                        pot=1,  # V --> V
                        angle=np.pi / 180,  # degrees --> radians
                        heatflux=1e3 * 1e4,  # kW cm^-2 --> W m^-2
                    )
                    for tdi_part, imas_part in nodes.items():
                        mds_dat = mdsvalue('d3d', pulse=pulse, treename='langmuir', TDI=rf'\langmuir::top.probe_{i:03d}.{tdi_part}')
                        if np.array_equal(t, mds_dat.dim_of(0)):
                            ods['langmuir_probes.embedded'][j][f'{imas_part}.data'] = mds_dat.data() * unit_conversions.get(tdi_part, 1)
                        else:
                            raise ValueError('Time base for Langmuir probe {i:03d} does not match {tdi_part} data')
                j += 1


# ================================
@machine_mapping_function(__regression_arguments__, pulse=133221)
def charge_exchange_hardware(ods, pulse, analysis_type='CERQUICK'):
    """
    Gathers DIII-D CER measurement locations from MDSplus

    :param analysis_type: string
        CER analysis quality level like CERQUICK, CERAUTO, or CERFIT
    """
    unwrap(charge_exchange_data)(ods, pulse, analysis_type, _measurements=False)


@machine_mapping_function(__regression_arguments__, pulse=133221)
def charge_exchange_data(ods, pulse, analysis_type='CERQUICK', _measurements=True):
    """
    Gathers DIII-D CER measurement data from MDSplus

    :param analysis_type: string
        CER analysis quality level like CERQUICK, CERAUTO, or CERFIT
    """
    import MDSplus

    printd('Setting up DIII-D CER data...', topic='machine')

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
            fetch_keys = ['TIME', 'R', 'Z', 'VIEW_PHI']
            if _measurements:
                fetch_keys.extend([])
            for pos in [0]:
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
            ch['name'] = 'impCER_{sub:}{ch:02d}'.format(sub=sub.lower()[0], ch=k + 1)
            ch['identifier'] = '{}{:02d}'.format(sub[0], k + 1)
            for pos in ['R', 'Z', 'VIEW_PHI']:
                posdat = data[f'{sub}_{channel}_{pos}']
                chpos = ch['position'][pos.lower().split('_')[-1]]
                chpos['time'] = postime / 1000.0  # Convert ms to s
                chpos['data'] = posdat * -np.pi / 180.0 if pos == 'VIEW_PHI' and not isinstance(posdat, Exception) else posdat


# ================================
@machine_mapping_function(__regression_arguments__)
def magnetics_weights(ods, pulse, time_index):
    r"""
    Load DIII-D tokamak magnetics equilibrium weights

    :param ods: ODS instance
    """
    from omfit_classes.omfit_omas_d3d import OMFITd3dfitweight

    fitweight = get_support_file(OMFITd3dfitweight, support_filenames('d3d', 'fitweight', pulse))
    weight_ishot = -1
    for ishot in fitweight:
        if pulse > ishot and ishot > weight_ishot:
            weight_ishot = ishot
    ods['equilibrium.time_slice.{time_index}.constraints.bpol_probe.:.weight'] = fitweight[weight_ishot]['fwtmp2']
    ods['equilibrium.time_slice.{time_index}.constraints.flux_loop.:.weight'] = fitweight[weight_ishot]['fwtsi']


@machine_mapping_function(__regression_arguments__)
def magnetics_hardware(ods, pulse):
    r"""
    Load DIII-D tokamak flux loops and magnetic probes hardware geometry

    :param ods: ODS instance
    """
    from omfit_classes.omfit_efund import OMFITmhdin

    mhdin = get_support_file(OMFITmhdin, support_filenames('d3d', 'mhdin', pulse))
    mhdin.to_omas(ods, update='magnetics')


@machine_mapping_function(__regression_arguments__, pulse=133221)
def magnetics_floops_data(ods, pulse, nref=0):
    from scipy.interpolate import interp1d
    from omfit_classes.omfit_omas_d3d import OMFITd3dcompfile

    ods1 = ODS()
    unwrap(magnetics_hardware)(ods1, pulse)

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

    # Set reference flux loop to zero before
    ods[f'magnetics.flux_loop.{nref}.flux.data'] *= 0.0

    for compfile in ['btcomp', 'ccomp', 'icomp']:
        comp = get_support_file(OMFITd3dcompfile, support_filenames('d3d', compfile, pulse))
        compshot = -1
        for shot in comp:
            if pulse > compshot:
                compshot = shot
        for compsig in comp[compshot]:
            if compsig == 'N1COIL' and pulse > 112962:
                continue
            m = mdsvalue('d3d', pulse=pulse, TDI=f'ptdata("{compsig}",{pulse})', treename=None)
            compsig_data = m.data()
            compsig_time = m.dim_of(0) / 1000.0
            for channel in ods['magnetics.flux_loop']:
                if f'magnetics.flux_loop.{channel}.identifier' in ods1:
                    sig = ods1[f'magnetics.flux_loop.{channel}.identifier']
                    sigraw_data = ods[f'magnetics.flux_loop.{channel}.flux.data']
                    sigraw_time = ods[f'magnetics.flux_loop.{channel}.flux.time']
                    compsig_data_interp = interp1d(compsig_time, compsig_data, bounds_error=False, fill_value=(0, 0))(sigraw_time)
                    ods[f'magnetics.flux_loop.{channel}.flux.data'] -= comp[compshot][compsig][sig] * compsig_data_interp

    # fetch uncertainties
    TDIs = {}
    for k in ods1['magnetics.flux_loop']:
        identifier = ods1[f'magnetics.flux_loop.{k}.identifier'].upper()
        TDIs[identifier] = f'pthead2("{identifier}",{pulse}), __rarray'

    data = mdsvalue('d3d', None, pulse, TDIs).raw()
    for k in ods1['magnetics.flux_loop']:
        identifier = ods1[f'magnetics.flux_loop.{k}.identifier'].upper()
        nt = len(ods[f'magnetics.flux_loop.{k}.flux.data'])
        ods[f'magnetics.flux_loop.{k}.flux.data_error_upper'] = 10 * abs(data[identifier][3] * data[identifier][4]) * np.ones(nt)


@machine_mapping_function(__regression_arguments__, pulse=133221)
def magnetics_probes_data(ods, pulse):
    from omfit_classes.omfit_omas_d3d import OMFITd3dcompfile

    ods1 = ODS()
    unwrap(magnetics_hardware)(ods1, pulse)

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

    for compfile in ['btcomp', 'ccomp', 'icomp']:
        comp = get_support_file(OMFITd3dcompfile, support_filenames('d3d', compfile, pulse))
        compshot = -1
        for shot in comp:
            if pulse > compshot:
                compshot = shot
        for compsig in comp[compshot]:
            if compsig == 'N1COIL' and pulse > 112962:
                continue
            m = mdsvalue('d3d', pulse=pulse, TDI=f"[ptdata2(\"{compsig}\",{pulse})]", treename=None)
            compsig_data = m.data()

            compsig_time = m.dim_of(0) / 1000
            for channel in ods1['magnetics.b_field_pol_probe']:
                if 'magnetics.b_field_pol_probe.{channel}.identifier' in ods1:
                    sig = 'magnetics.b_field_pol_probe.{channel}.identifier'
                    sigraw_data = ods[f'magnetics.b_field_pol_probe.{channel}.field.data']
                    sigraw_time = ods[f'magnetics.b_field_pol_probe.{channel}.field.time']
                    compsig_data_interp = np.interp(sigraw_time, compsig_time, compsig_data)
                    ods[f'magnetics.b_field_pol_probe.{channel}.field.data'] -= comp[compshot][compsig][sig] * compsig_data_interp

    # fetch uncertainties
    TDIs = {}
    for k in ods1['magnetics.b_field_pol_probe']:
        identifier = ods1[f'magnetics.b_field_pol_probe.{k}.identifier'].upper()
        TDIs[identifier] = f'pthead2("{identifier}",{pulse}), __rarray'

    data = mdsvalue('d3d', None, pulse, TDIs).raw()

    for k in ods1['magnetics.b_field_pol_probe']:
        identifier = ods1[f'magnetics.b_field_pol_probe.{k}.identifier'].upper()
        nt = len(ods[f'magnetics.b_field_pol_probe.{k}.field.data'])
        ods[f'magnetics.b_field_pol_probe.{k}.field.data_error_upper'] = abs(data[identifier][3] * data[identifier][4]) * np.ones(nt) * 10.0


@machine_mapping_function(__regression_arguments__, pulse=133221)
def ip_bt_dflux_data(ods, pulse):
    r"""
    Load DIII-D tokamak Ip, Bt, and diamagnetic flux data

    :param ods: ODS instance

    :param pulse: shot number
    """

    mappings = {'magnetics.ip.0': 'IP', 'tf.b_field_tor_vacuum_r': 'BT', 'magnetics.diamagnetic_flux.0': 'DIAMAG3'}

    with omas_environment(ods, cocosio=1):
        TDIs = {}
        for key, val in mappings.items():
            TDIs[key + '.data'] = f'ptdata("{val}",{pulse})'
            TDIs[key + '.time'] = f'dim_of(ptdata2("{val}",{pulse}),0)/1000.'
            TDIs[key + '.data_error_upper'] = f'pthead2("{val}",{pulse}), __rarray'

        data = mdsvalue('d3d', None, pulse, TDIs).raw()
        for key in TDIs.keys():
            if 'data_error_upper' in key:
                nt = len(ods[key[:-12]])
                ods[key] = abs(data[key][3] * data[key][4]) * np.ones(nt) * 10.0
            else:
                ods[key] = data[key]

            if 'magnetics.diamagnetic_flux.0.data' in key:
                ods[key] *= 1e-3

        ods['tf.b_field_tor_vacuum_r.data'] *= 1.6955


# ================================
@machine_mapping_function(__regression_arguments__, pulse=133221)
def core_profiles_global_quantities_data(ods, pulse):
    from scipy.interpolate import interp1d

    ods1 = ODS()
    unwrap(magnetics_hardware)(ods1)

    with omas_environment(ods, cocosio=1):
        cp = ods['core_profiles']
        gq = ods['core_profiles.global_quantities']
        if 'time' not in cp:
            m = mdsvalue('d3d', pulse=pulse, TDI="\\TOP.PROFILES.EDENSFIT", treename="ZIPFIT01")
            cp['time'] = m.dim_of(1) * 1e-3
        t = cp['time']

        m = mdsvalue('d3d', pulse=pulse, TDI=f"ptdata2(\"VLOOP\",{pulse})", treename=None)

        gq['v_loop'] = interp1d(m.dim_of(0) * 1e-3, m.data(), bounds_error=False, fill_value=np.NaN)(t)


# ================================
if __name__ == '__main__':
    test_machine_mapping_functions(['ec_launcher_active_hardware'], globals(), locals())
    test_machine_mapping_functions(['thomson_scattering_hardware'], globals(), locals())
