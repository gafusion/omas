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
    elif tok == 'JET':
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

    zero_d_locations = {
        "['ZEROD']['RGEO']": 'summary.boundary.geometric_axis_r.value',
        "['ZEROD']['AMIN']": 'summary.boundary.minor_radius.value',
        "['ZEROD']['DWTOT']": 'summary.global_quantities.denergy_thermal_dt.value',
        # 'hl_mode': 'summary.global_quantities.h_mode.value',
        "['ZEROD']['IP']": 'summary.global_quantities.ip.value',
        "['ZEROD']['ZEFF']": 'summary.volume_average.zeff.value',
        "['ZEROD']['TAUTH']": 'summary.global_quantities.tau_energy.value',
        "['ZEROD']['BT']": 'summary.global_quantities.b0.value',
        "['ZEROD']['KAPPA']": 'summary.boundary.elongation.value',
        "['ZEROD']['DELTA']": 'summary.boundary.triangularity_upper.value',
        "['ZEROD']['DELTA']": 'summary.boundary.triangularity_lower.value',
        "['ZEROD']['NEL']": 'summary.line_average.n_e.value',
        "['ZEROD']['PINJ']": 'summary.heating_current_drive.power_launched_nbi.value',
        "['ZEROD']['PECH']": 'summary.heating_current_drive.power_launched_ec.value',
        "['ZEROD']['POHM']": 'summary.global_quantities.power_ohm.value',
        "['ZEROD']['PICRH']": 'summary.heating_current_drive.power_launched_ic.value',
        "['ZEROD']['PLTH']": 'summary.global_quantities.power_loss.value',
        "['ZEROD']['WTH']": 'summary.global_quantities.energy_thermal.value',
    }

    one_d_locations = {
        # 'zeff': 'core_profiles.profiles_1d[0].zeff',
        "['TWOD']['VOLUME']": 'core_profiles.profiles_1d[0].grid.volume',
        "['TWOD']['VROT']": 'core_profiles.profiles_1d[0].rotation_frequency_tor_sonic',  # this is wrong
        "['TWOD']['Q']": 'core_profiles.profiles_1d[0].q',
        "['TWOD']['TI']": 'core_profiles.profiles_1d[0].t_i_average',
        "['TWOD']['TE']": 'core_profiles.profiles_1d[0].electrons.temperature',
        "['TWOD']['NE']": 'core_profiles.profiles_1d[0].electrons.density_thermal',
        "['TWOD']['CURTOT']": 'core_profiles.profiles_1d[0].j_tor',
        "['TWOD']['QNBII']": 'q_nbi_i',
        "['TWOD']['QNBIE']": 'q_nbi_e',
        "['TWOD']['SNBII']": 's_nbi_se',
        "['TWOD']['QOHM']": 'q_ohm_e',
    }

    mds_tree = mdstree(server='tokamak-profiledb.ccfe.ac.uk:8000', pulse=int(pulse), treename=f'pr{db}_{tok}')

    ods['summary.global_quantities.h_mode.value'] = [True]
    ods['dataset_description.data_entry.machine'] = tok
    ods['dataset_description.data_entry.pulse'] = int(mds_tree['ZEROD']['SHOT'].data())
    ods['summary.time'] = mds_tree['ZEROD']['TIME'].data()
    for mds_location, ods_location in zero_d_locations.items():
        try:
            ods[ods_location] = np.array(eval(f"mds_tree{mds_location}").data())
        except Exception as _excp:
            printe(repr(_excp))
            ods[ods_location] = [0.0]

    heating_idx_dict = {'nbi': 2, 'ec': 3, 'lh': 4, 'ic': 5, 'fusion': 6, 'ohm': 7}
    ion_elec = {'i': 'total_ion_energy', 'e': 'electrons.energy', 'se': 'electrons.particles'}

    rho_init = np.linspace(0, 1, len(mds_tree['TWOD']['VOLUME'].data()[0]))
    if ngrid > 0:
        rho_tor_norm = np.linspace(0, 1, ngrid)
    else:
        rho_tor_norm = rho_init
    ods['core_profiles.profiles_1d[0].grid.rho_tor_norm'] = rho_tor_norm

    for mds_location, ods_location in one_d_locations.items():
        try:
            if '.' not in ods_location:
                name = ods_location.split(sep='_')[1]
                ods[f'core_sources.source.+.identifier.index'] = heating_idx_dict[name]
                ods[f'core_sources.source.-1.identifier.name'] = name
                ods[f'core_sources.source.-1.profiles_1d[0].grid.rho_tor_norm'] = rho_tor_norm
                ods[f'core_sources.source.-1.profiles_1d[0].grid.volume'] = np.interp(
                    x=rho_tor_norm, xp=rho_init, fp=mds_tree['TWOD']['VOLUME'].data()[0]
                )
                ods[f"core_sources.source.-1.profiles_1d[0].{ion_elec[ods_location.split(sep='_')[-1]]}"] = np.interp(
                    x=rho_tor_norm, xp=rho_init, fp=eval(f"mds_tree{mds_location}").data()[0]
                )

            else:
                ods[ods_location] = np.interp(x=rho_tor_norm, xp=rho_init, fp=eval(f"mds_tree{mds_location}").data()[0])
                if 'rotation_frequency_tor_sonic' in ods_location:
                    ods[ods_location] /= 2 * np.pi

        except Exception as e:
            printe(repr(e))
            ods[ods_location] = np.zeros(ngrid)


# =====================
if __name__ == '__main__':
    test_machine_mapping_functions(__all__, globals(), locals())
