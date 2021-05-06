import os
import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd, printe, unumpy
from omas.machine_mappings._common import *

__all__ = []
__regression_arguments__ = {'__all__': __all__}

"""
Connects to tokamak-profiledb.ccfe.ac.uk and generates an ODS with core profiles and zero-D parameters

options:
tok     : string  : 'jet' or 'd3d'
db      : string  : '98' or '08'  (1998 or 2008 database)
ngrid   : int     : nrho grid output into ods

Data sources:
profiles-db with more information:tokamak-profiledb.ccfe.ac.uk

example usage:
    ods = ODS()
    ods.open('profiles_db_ccfe', pulse=81499, options={'tok':'d3d', 'db':'08'})
"""

@machine_mapping_function(__regression_arguments__, pulse=77557, tok="d3d", db='98', ngrid=201)
def profile_db_to_ODS(ods, pulse, tok, db, ngrid):
    ods['dataset_description.ids_properties.comment'] = f'Comment for {tok}'

    if ods is None:
        ods = ODS()
    if tok == 'DIII-D':
        tok = 'd3d'
    elif tok =='JET':
        tok = 'jet'

    # fmt: off
    available_in_database = {
        'd3d': [69627, 69648, 71378, 71384, 77557, 77559, 78106, 78109, 78281, 78283, 78316, 78328, 81321, 81329, 81499, 81507, 82183,
                82188, 82205, 82788, 84682, 87031, 89943, 90105, 90108, 90117, 90118, 92664, 95989, 98549, 98775, 98777, 99251, 99411,
                99696, 103818, 104276, 106919, 106956, 111203, 111221, 111239, 118334, 118341, 118348, 118446],
        'jet': [19649, 19691, 26087, 26095, 32745, 33131, 33140, 33465, 34340, 35156, 35171, 35174, 37379, 37718, 37728, 37944, 38285,
                38287, 38407, 38415, 40542, 40847, 42762, 42794, 42976, 42982, 42997, 43002, 43134, 43452, 46123, 46664, 49687, 50844,
                51599, 51976, 52009, 52014, 52015, 52022, 52024, 52025, 52096, 52979, 53028, 53030, 53212, 53299, 53501, 53521, 53532,
                53537, 53550, 55935, 57987, 58159, 58323, 60927, 60931, 60933]

        }
    # fmt: on

    if tok not in available_in_database:
        print(f"tokamak not in database see: {available_in_database.keys()}")
    if pulse not in available_in_database[tok]:
        print(f"Shot {pulse} not in {tok} database, available pulses = {available_in_database[tok]}")

    zero_d_ods_locations = {
        'R': 'summary.boundary.geometric_axis_r.value',
        'a': 'summary.boundary.minor_radius.value',
        'dwdt': 'summary.global_quantities.denergy_thermal_dt.value',
        'hl_mode': 'summary.global_quantities.h_mode.value',
        'Ip': 'summary.global_quantities.ip.value',
        'zeff': 'summary.volume_average.zeff.value',
        'tau_exp': 'summary.global_quantities.tau_energy.value',
        'Bt': 'summary.global_quantities.b0.value',
        'kappa': 'summary.boundary.elongation.value',
        'delta_u': 'summary.boundary.triangularity_upper.value',
        'delta_l': 'summary.boundary.triangularity_lower.value',
        'nel': 'summary.line_average.n_e.value',
        'power_nbi': 'summary.heating_current_drive.power_launched_nbi.value',
        'power_ec': 'summary.heating_current_drive.power_launched_ec.value',
        'power_ohm': 'summary.global_quantities.power_ohm.value',
        'power_ic': 'summary.heating_current_drive.power_launched_ic.value',
        'power_loss': 'summary.global_quantities.power_loss.value',
        'stored_energy': 'summary.global_quantities.energy_thermal.value',
    }
    zero_d_profiles_locations = {
        'a': "['ZEROD']['AMIN']",
        'R': "['ZEROD']['RGEO']",
        'dwdt': "['ZEROD']['DWDIA']",
        'Ip': "['ZEROD']['IP']",
        'zeff': "['ZEROD']['ZEFF']",
        'tau_exp': "['ZEROD']['TAUTH']",
        'Bt': "['ZEROD']['BT']",
        'kappa': "['ZEROD']['KAPPA']",
        'delta_l': "['ZEROD']['DELTA']",
        'delta_u': "['ZEROD']['DELTA']",
        'nel': "['ZEROD']['NEL']",
        'power_nbi': "['ZEROD']['PINJ']",
        'power_ec': "['ZEROD']['PECH']",
        'power_ohm': "['ZEROD']['POHM']",
        'power_ic': "['ZEROD']['PECH']",
        'power_loss': "['ZEROD']['PLTH']",
        'stored_energy': "['ZEROD']['WTH']",
    }

    one_d_ods_locations = {
        'zeff': 'core_profiles.profiles_1d[0].zeff',
        'volume': 'core_profiles.profiles_1d[0].grid.volume',
        'omega': 'core_profiles.profiles_1d[0].rotation_frequency_tor_sonic',
        'q': 'core_profiles.profiles_1d[0].q',
        'Ti': 'core_profiles.profiles_1d[0].t_i_average',
        'Te': 'core_profiles.profiles_1d[0].electrons.temperature',
        'ne': 'core_profiles.profiles_1d[0].electrons.density_thermal',
        'jtot': 'core_profiles.profiles_1d[0].j_tor',
    }
    one_d_profiles_locations = {
        'volume': "['TWOD']['VOLUME']",
        'omega': "['TWOD']['VROT']",
        'q': "['TWOD']['Q']",
        'Ti': "['TWOD']['TI']",
        'Te': "['TWOD']['TE']",
        'ne': "['TWOD']['NE']",
        'q_nbi_i': "['TWOD']['QNBII']",
        'q_nbi_e': "['TWOD']['QNBIE']",
        's_nbi_se': "['TWOD']['SNBII']",
        'q_ohm_e': "['TWOD']['QOHM']",
        'jtot': "['TWOD']['CURTOT']",
    }

    mds_tree = mdstree(server='tokamak-profiledb.ccfe.ac.uk:8000', pulse=int(pulse), treename=f'pr{db}_{tok}')

    ods['summary.global_quantities.h_mode.value'] = [True]
    ods['dataset_description.data_entry.machine'] = tok
    ods['dataset_description.data_entry.pulse'] = int(mds_tree['ZEROD']['SHOT'].data())
    ods['summary.time'] = mds_tree['ZEROD']['TIME'].data()
    for key, location in zero_d_profiles_locations.items():
        try:
            if key in zero_d_ods_locations:
                #print(location, eval(f"mds_tree{location}").data()[-1])
                ods[zero_d_ods_locations[key]] = np.array(eval(f"mds_tree{location}").data())

        except Exception as e:
            printe(repr(e))
            if key in zero_d_ods_locations:
                ods[zero_d_ods_locations[key]] = [0.0]

    heating_idx_dict = {'nbi': 2, 'ec': 3, 'lh': 4, 'ic': 5, 'fusion': 6, 'ohm': 7}
    source_index = 0
    ion_elec = {'i': 'total_ion_energy', 'e': 'electrons.energy', 'se': 'electrons.particles'}

    rho_init = np.linspace(0, 1, len(mds_tree['TWOD']['VOLUME'].data()[0]))
    if ngrid > 0:
        rho_tor_norm = np.linspace(0, 1, ngrid)
    else:
        rho_tor_norm = rho_init
    ods['core_profiles.profiles_1d[0].grid.rho_tor_norm'] = rho_tor_norm

    for key, location in one_d_profiles_locations.items():
        try:
            if 'q_' in key or 's_' in key:
                name = key.split(sep='_')[1]
                ods[f'core_sources.source[{source_index}].identifier.index'] = heating_idx_dict[name]
                ods[f'core_sources.source[{source_index}].identifier.name'] = name
                ods[f'core_sources.source[{source_index}].profiles_1d[0].grid.rho_tor_norm'] = rho_tor_norm
                ods[f'core_sources.source[{source_index}].profiles_1d[0].grid.volume'] = np.interp(
                    x=rho_tor_norm, xp=rho_init, fp=mds_tree['TWOD']['VOLUME'].data()[0]
                )
                ods[f"core_sources.source[{source_index}].profiles_1d[0].{ion_elec[key.split(sep='_')[-1]]}"] = np.interp(
                    x=rho_tor_norm, xp=rho_init, fp=eval(f"mds_tree{location}").data()[0]
                )

                source_index += 1

            elif key in one_d_ods_locations:
                ods[one_d_ods_locations[key]] = np.interp(x=rho_tor_norm, xp=rho_init, fp=eval(f"mds_tree{location}").data()[0])

                if key == 'omega':
                    ods[one_d_ods_locations[key]] /= 2 * np.pi

        except Exception as e:
            printe(repr(e))
            if key in zero_d_ods_locations:
                ods[one_d_ods_locations[key]] = np.zeros(ngrid)


# =====================
if __name__ == '__main__':
    test_machine_mapping_functions(__all__, globals(), locals())
