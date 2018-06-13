from __future__ import print_function, division, unicode_literals

from .omas_utils import *

__all__ = []

def add_to__ODS__(f):
    __all__.append(f.__name__)
    return f

# constants class that mimics scipy.constants
class constants(object):
    e = 1.6021766208e-19


@add_to__ODS__
def core_profiles_pressures(ods, update=True):
    '''
    calculates individual ions pressures
        `core_profiles.profiles_1d.:.ion.:.pressure_thermal` #Pressure (thermal) associated with random motion ~average((v-average(v))^2)
        `core_profiles.profiles_1d.:.ion.:.pressure`         #Pressure (thermal+non-thermal)

    as well as total pressures

        `core_profiles.profiles_1d.:.pressure_thermal`       #Thermal pressure (electrons+ions)
        `core_profiles.profiles_1d.:.pressure_ion_total`     #Total (sum over ion species) thermal ion pressure
        `core_profiles.profiles_1d.:.pressure_perpendicular` #Total perpendicular pressure (electrons+ions, thermal+non-thermal)
        `core_profiles.profiles_1d.:.pressure_parallel`      #Total parallel pressure (electrons+ions, thermal+non-thermal)

    NOTE: the fast particles ion pressures are read, not set by this function:
        `core_profiles.profiles_1d.:.ion.:.pressure_fast_parallel`      #Pressure (thermal) associated with random motion ~average((v-average(v))^2)
        `core_profiles.profiles_1d.:.ion.:.pressure_fast_perpendicular` #Pressure (thermal+non-thermal)

    :param ods: input ods

    :param update: operate in place

    :return: updated ods
    '''
    ods_p = ods
    if not update:
        from omas import ODS
        ods_p = ODS().copy_attrs_from(ods)

    for time_index in ods['core_profiles']['profiles_1d']:
        prof1d = ods['core_profiles']['profiles_1d'][time_index]
        prof1d_p = ods_p['core_profiles']['profiles_1d'][time_index]
        __p__ = prof1d['electrons']['density'] * prof1d['electrons']['temperature'] * constants.e
        prof1d_p['pressure_thermal'] = __p__
        prof1d_p['pressure_ion_total'] = __p__ * 0.0
        prof1d_p['pressure_perpendicular'] = __p__ / 3.
        prof1d_p['pressure_parallel'] = __p__ / 3.
        for k in range(len(prof1d['ion'])):
            prof1d_p['ion'][k]['pressure'] = __p__ * 0.0
            for therm_fast, density in [('therm', 'density'), ('fast', 'density_fast')]:
                if (len(prof1d['ion']) and density in prof1d['ion'][k] and numpy.sum(
                        numpy.abs(prof1d['ion'][k][density])) > 0):
                    if therm_fast == 'therm':
                        __p__ = prof1d['ion'][k]['density'] * prof1d['ion'][k]['temperature'] * constants.e
                        prof1d_p['ion'][k]['pressure_thermal'] = __p__
                        prof1d_p['pressure_ion_total'] += __p__
                        prof1d_p['pressure_thermal'] += __p__
                        prof1d_p['pressure_perpendicular'] += __p__ / 3.
                        prof1d_p['pressure_parallel'] += __p__ / 3.
                    else:
                        if not update:
                            prof1d_p['ion'][k]['pressure_fast_perpendicular'] = prof1d['ion'][k][
                                'pressure_fast_perpendicular']
                            prof1d_p['ion'][k]['pressure_fast_parallel'] = prof1d['ion'][k]['pressure_fast_parallel']
                        prof1d_p['pressure_perpendicular'] += prof1d['ion'][k]['pressure_fast_perpendicular']
                        prof1d_p['pressure_parallel'] += prof1d['ion'][k]['pressure_fast_parallel']
                        __p__ = prof1d_p['ion'][k]['pressure_fast_perpendicular'] * 2 + prof1d_p['ion'][k]['pressure_fast_parallel']
                    prof1d_p['ion'][k]['pressure'] += __p__

        #extra pressure information that is not within IMAS structure is set only if consistency_check is not True
        if ods_p.consistency_check is not True:
            prof1d_p['pressure'] = prof1d_p['pressure_perpendicular'] * 2 + prof1d_p['pressure_parallel']
            prof1d_p['pressure_electron_total'] = prof1d_p['pressure_thermal'] - prof1d_p['pressure_ion_total']
            prof1d_p['pressure_fast'] = prof1d_p['pressure'] - prof1d_p['pressure_thermal']

    return ods_p


def define_cocos(cocos_ind):
    """
    Returns dictionary with COCOS coefficients given a COCOS index

    https://docs.google.com/document/d/1-efimTbI55SjxL_yE_GKSmV4GEvdzai7mAj5UYLLUXw/edit

    :param cocos_ind: COCOS index

    :return: dictionary with COCOS coefficients
    """

    cocos=dict.fromkeys(['sigma_Bp', 'sigma_RpZ', 'sigma_rhotp', 'sign_q_pos', 'sign_pprime_pos', 'exp_Bp'])

    # all multipliers shouldn't change input values if cocos_ind is None
    if cocos_ind is None:
        cocos['exp_Bp']          = 0
        cocos['sigma_Bp']        = +1
        cocos['sigma_RpZ']       = +1
        cocos['sigma_rhotp']     = +1
        cocos['sign_q_pos']      = 0
        cocos['sign_pprime_pos'] = 0
        return cocos

    # if COCOS>=10, this should be 1
    cocos['exp_Bp'] = 0
    if cocos_ind>=10:
        cocos['exp_Bp'] = +1

    if cocos_ind in [1,11]:
        # These cocos are for
        # (1)  psitbx(various options), Toray-GA
        # (11) ITER, Boozer
        cocos['sigma_Bp']        = +1
        cocos['sigma_RpZ']       = +1
        cocos['sigma_rhotp']     = +1
        cocos['sign_q_pos']      = +1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [2,12,-12]:
        # These cocos are for
        # (2)  CHEASE, ONETWO, HintonHazeltine, LION, XTOR, MEUDAS, MARS, MARS-F
        # (12) GENE
        # (-12) ASTRA
        cocos['sigma_Bp']        = +1
        cocos['sigma_RpZ']       = -1
        cocos['sigma_rhotp']     = +1
        cocos['sign_q_pos']      = +1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [3,13]:
        # These cocos are for
        # (3) Freidberg*, CAXE and KINX*, GRAY, CQL3D^, CarMa, EFIT* with : ORB5, GBSwith : GT5D
        # (13) 	CLISTE, EQUAL, GEC, HELENA, EU ITM-TF up to end of 2011
        cocos['sigma_Bp']        = -1
        cocos['sigma_RpZ']       = +1
        cocos['sigma_rhotp']     = -1
        cocos['sign_q_pos']      = -1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [4,14]:
        #These cocos are for
        cocos['sigma_Bp']        = -1
        cocos['sigma_RpZ']       = -1
        cocos['sigma_rhotp']     = -1
        cocos['sign_q_pos']      = -1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [5,15]:
        # These cocos are for
        # (5) TORBEAM, GENRAY^
        cocos['sigma_Bp']        = +1
        cocos['sigma_RpZ']       = +1
        cocos['sigma_rhotp']     = -1
        cocos['sign_q_pos']      = -1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [6,16]:
        #These cocos are for
        cocos['sigma_Bp']        = +1
        cocos['sigma_RpZ']       = -1
        cocos['sigma_rhotp']     = -1
        cocos['sign_q_pos']      = -1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [7,17]:
        #These cocos are for
        # (17) LIUQE*, psitbx(TCV standard output)
        cocos['sigma_Bp']        = -1
        cocos['sigma_RpZ']       = +1
        cocos['sigma_rhotp']     = +1
        cocos['sign_q_pos']      = +1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [8,18]:
        #These cocos are for
        cocos['sigma_Bp']        = -1
        cocos['sigma_RpZ']       = -1
        cocos['sigma_rhotp']     = +1
        cocos['sign_q_pos']      = +1
        cocos['sign_pprime_pos'] = +1

    return cocos


def cocos_transform(cocosin_index, cocosout_index):
    """
    Returns a dictionary with coefficients for how various quantities should get multiplied in order to go from cocosin_index to cocosout_index

    https://docs.google.com/document/d/1-efimTbI55SjxL_yE_GKSmV4GEvdzai7mAj5UYLLUXw/edit

    :param cocosin_index: COCOS index in

    :param cocosout_index: COCOS index out

    :return: dictionary with transformation multipliers
    """

    # Don't transform if either cocos is undefined
    if (cocosin_index is None) or (cocosout_index is None):
        printd("No COCOS tranformation for "+str(cocosin_index)+" to "+str(cocosout_index),topic='cocos')
        sigma_Ip_eff    = 1
        sigma_B0_eff    = 1
        sigma_Bp_eff    = 1
        exp_Bp_eff      = 0
        sigma_rhotp_eff = 1
    else:
        printd("COCOS tranformation from "+str(cocosin_index)+" to "+str(cocosout_index),topic='cocos')
        cocosin = define_cocos(cocosin_index)
        cocosout = define_cocos(cocosout_index)

        sigma_Ip_eff    = cocosin['sigma_RpZ'] * cocosout['sigma_RpZ']
        sigma_B0_eff    = cocosin['sigma_RpZ'] * cocosout['sigma_RpZ']
        sigma_Bp_eff    = cocosin['sigma_Bp'] * cocosout['sigma_Bp']
        exp_Bp_eff      = cocosout['exp_Bp'] - cocosin['exp_Bp']
        sigma_rhotp_eff = cocosin['sigma_rhotp'] * cocosout['sigma_rhotp']

    # Transform
    transforms = {}
    transforms['1/PSI'] = sigma_Ip_eff * sigma_Bp_eff / (2 * numpy.pi) ** exp_Bp_eff
    transforms['invPSI'] = transforms['1/PSI']
    transforms['dPSI'] = transforms['1/PSI']
    transforms['F_FPRIME'] = transforms['dPSI']
    transforms['PPRIME'] = transforms['dPSI']
    transforms['PSI'] = sigma_Ip_eff * sigma_Bp_eff * (2 * numpy.pi) ** exp_Bp_eff
    transforms['Q'] = sigma_Ip_eff * sigma_B0_eff * sigma_rhotp_eff
    transforms['TOR'] = sigma_B0_eff
    transforms['BT'] = transforms['TOR']
    transforms['IP'] = transforms['TOR']
    transforms['F'] = transforms['TOR']
    transforms['POL'] = sigma_B0_eff * sigma_rhotp_eff
    transforms['BP'] = transforms['POL']

    printd(transforms,topic='cocos')

    return transforms


@contextmanager
def cocos_environment(ods, cocosin=None, cocosout=None):
    '''
    Provides OMAS environment within wich a certain COCOS convention is defined

    :param ods: ODS on which to operate

    :param cocosin: input COCOS convention

    :param cocosout: output COCOS convention

    :return: ODS with COCOS convention set
    '''
    old_cocosin = ods.cocosin
    old_cocosout = ods.cocosout
    if cocosin is not None:
        ods.cocosin = cocosin
    if cocosout is not None:
        ods.cocosout = cocosout
    try:
        yield ods
    finally:
        ods.cocosin = old_cocosin
        ods.cocosout = old_cocosout


# This dictionary defines the IMAS locations and the corresponding `cocos_transform` function
cocos_signals = {}

def print_utility_cocos_signals(structures):
    '''
    This is a utility function for printing the cocos_signals entries as the ones defined in omas/omas_physics.py
    The printed text must be copied to omas/omas_physics.py and the COCOS transformations assigned by hand.
    '''
    if isinstance(structures, basestring):
        structures = [structures]
    from .omas_utils import _structures
    from .omas_utils import i2o
    from .omas_core import ODS
    ods = ODS()
    for structure in structures:
        ods[structure]
        for item in sorted(list(list(_structures.values())[0].keys())):
            if item.startswith(structure) and '_error_' not in item:
                print("cocos_signals['%s']=" % i2o(item))

# EQUILIBRIUM
cocos_signals['equilibrium.time_slice.:.constraints.b_field_tor_vacuum_r.exact'] = 'BT'
cocos_signals['equilibrium.time_slice.:.constraints.b_field_tor_vacuum_r.measured'] = 'BT'
cocos_signals['equilibrium.time_slice.:.constraints.b_field_tor_vacuum_r.reconstructed'] = 'BT'
cocos_signals['equilibrium.time_slice.:.constraints.b_field_tor_vacuum_r.standard_deviation'] = 'BT'
cocos_signals['equilibrium.time_slice.:.constraints.ip.exact'] = 'IP'
cocos_signals['equilibrium.time_slice.:.constraints.ip.measured'] = 'IP'
cocos_signals['equilibrium.time_slice.:.constraints.ip.reconstructed'] = 'IP'
cocos_signals['equilibrium.time_slice.:.constraints.ip.standard_deviation'] = 'IP'
cocos_signals['equilibrium.time_slice.:.constraints.q.:.exact'] = 'Q'
cocos_signals['equilibrium.time_slice.:.constraints.q.:.measured'] = 'Q'
cocos_signals['equilibrium.time_slice.:.constraints.q.:.reconstructed'] = 'Q'
cocos_signals['equilibrium.time_slice.:.constraints.q.:.standard_deviation'] = 'Q'
cocos_signals['equilibrium.time_slice.:.ggd.:.b_field_tor.:.values'] = 'BT'
cocos_signals['equilibrium.time_slice.:.ggd.:.j_parallel.:.values'] = 'IP'
cocos_signals['equilibrium.time_slice.:.ggd.:.j_tor.:.values'] = 'IP'
cocos_signals['equilibrium.time_slice.:.ggd.:.phi.:.values'] = 'BT'
cocos_signals['equilibrium.time_slice.:.ggd.:.psi.:.values'] = 'PSI'
cocos_signals['equilibrium.time_slice.:.global_quantities.ip'] = 'IP'
cocos_signals['equilibrium.time_slice.:.global_quantities.magnetic_axis.b_field_tor'] = 'BT'
cocos_signals['equilibrium.time_slice.:.global_quantities.magnetic_axis.b_tor'] = 'BT'
cocos_signals['equilibrium.time_slice.:.global_quantities.psi_axis'] = 'PSI'
cocos_signals['equilibrium.time_slice.:.global_quantities.psi_boundary'] = 'PSI'
cocos_signals['equilibrium.time_slice.:.global_quantities.q_95'] = 'Q'
cocos_signals['equilibrium.time_slice.:.global_quantities.q_axis'] = 'Q'
cocos_signals['equilibrium.time_slice.:.global_quantities.q_min.value'] = 'Q'
cocos_signals['equilibrium.time_slice.:.profiles_1d.b_average'] = 'BT'
cocos_signals['equilibrium.time_slice.:.profiles_1d.b_max'] = 'BT'
cocos_signals['equilibrium.time_slice.:.profiles_1d.b_min'] = 'BT'
cocos_signals['equilibrium.time_slice.:.profiles_1d.darea_dpsi'] = 'dPSI'
cocos_signals['equilibrium.time_slice.:.profiles_1d.dpressure_dpsi'] = 'dPSI'
cocos_signals['equilibrium.time_slice.:.profiles_1d.dpsi_drho_tor'] = 'PSI'
cocos_signals['equilibrium.time_slice.:.profiles_1d.dvolume_dpsi'] = 'dPSI'
cocos_signals['equilibrium.time_slice.:.profiles_1d.f'] = 'F'
cocos_signals['equilibrium.time_slice.:.profiles_1d.f_df_dpsi'] = 'F_FPRIME'
cocos_signals['equilibrium.time_slice.:.profiles_1d.j_parallel'] = 'IP'
cocos_signals['equilibrium.time_slice.:.profiles_1d.j_tor'] = 'IP'
cocos_signals['equilibrium.time_slice.:.profiles_1d.phi'] = 'BT'
cocos_signals['equilibrium.time_slice.:.profiles_1d.psi'] = 'PSI'
cocos_signals['equilibrium.time_slice.:.profiles_1d.q'] = 'Q'
cocos_signals['equilibrium.time_slice.:.profiles_2d.:.b_field_tor'] = 'BT'
cocos_signals['equilibrium.time_slice.:.profiles_2d.:.b_tor'] = 'BT'
cocos_signals['equilibrium.time_slice.:.profiles_2d.:.j_parallel'] = 'IP'
cocos_signals['equilibrium.time_slice.:.profiles_2d.:.j_tor'] = 'IP'
cocos_signals['equilibrium.time_slice.:.profiles_2d.:.phi'] = 'BT'
cocos_signals['equilibrium.time_slice.:.profiles_2d.:.psi'] = 'PSI'
cocos_signals['equilibrium.vacuum_toroidal_field.b0'] = 'BT'

# CORE_PROFILES
cocos_signals['core_profiles.profiles_1d.:.j_total'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.e_field_parallel'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.j_non_inductive'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.e_field.poloidal'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.e_field.parallel'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.e_field.toroidal'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.e_field.diamagnetic'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.j_tor'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.momentum_tor'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.ion.:.velocity_tor'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.ion.:.state.:.velocity.poloidal'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.ion.:.state.:.velocity.parallel'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.ion.:.state.:.velocity.toroidal'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.ion.:.state.:.velocity.diamagnetic'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.ion.:.velocity_pol'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.ion.:.velocity.poloidal'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.ion.:.velocity.parallel'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.ion.:.velocity.toroidal'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.ion.:.velocity.diamagnetic'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.state.:.velocity.poloidal'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.state.:.velocity.parallel'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.state.:.velocity.toroidal'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.state.:.velocity.diamagnetic'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.velocity.poloidal'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.velocity.parallel'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.velocity.toroidal'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.neutral.:.velocity.diamagnetic'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.grid.psi'] = 'PSI'
cocos_signals['core_profiles.profiles_1d.:.j_bootstrap'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.q'] = 'Q'
cocos_signals['core_profiles.profiles_1d.:.j_ohmic'] = 'IP'
cocos_signals['core_profiles.profiles_1d.:.electrons.velocity_tor'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.electrons.velocity_pol'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.electrons.velocity.poloidal'] = 'BP'
cocos_signals['core_profiles.profiles_1d.:.electrons.velocity.parallel'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.electrons.velocity.toroidal'] = 'BT'
cocos_signals['core_profiles.profiles_1d.:.electrons.velocity.diamagnetic'] = 'BT'
cocos_signals['core_profiles.vacuum_toroidal_field.b0'] = 'BT'
cocos_signals['core_profiles.global_quantities.ip'] = 'IP'
cocos_signals['core_profiles.global_quantities.v_loop'] =  'IP'
cocos_signals['core_profiles.global_quantities.current_bootstrap'] = 'IP'
cocos_signals['core_profiles.global_quantities.current_non_inductive'] = 'IP'

# CORE_SOURCES
cocos_signals['core_sources.source.:.global_quantities.:.current_parallel'] = 'IP'
cocos_signals['core_sources.source.:.global_quantities.:.torque_tor'] = 'BT'
cocos_signals['core_sources.source.:.profiles_1d.:.momentum_tor'] = 'BT'
cocos_signals['core_sources.source.:.profiles_1d.:.grid.psi'] = 'PSI'
cocos_signals['core_sources.source.:.profiles_1d.:.current_parallel_inside'] = 'IP'
cocos_signals['core_sources.source.:.profiles_1d.:.torque_tor_inside'] = 'BT'
cocos_signals['core_sources.source.:.profiles_1d.:.j_parallel'] = 'IP'
cocos_signals['core_sources.vacuum_toroidal_field.b0'] = 'BT'

# cocos_structures contains the list of all the IDSs that have been checked for COCOS convention transformations
cocos_structures = []
for item in cocos_signals:
    structure_name = item.split(separator)[0]
    if structure_name not in cocos_structures:
        cocos_structures.append(structure_name)
