'''physics-based ODS methods and utilities

-------
'''

from __future__ import print_function, division, unicode_literals
from .omas_utils import *
from .omas_core import ODS

__all__ = []
__ods__ = []

def add_to__ODS__(f):
    '''
    anything wrapped here will be available as a ODS method with name 'physics_'+f.__name__
    '''
    __ods__.append(f.__name__)
    __all__.append(f.__name__)
    return f

def add_to__ALL__(f):
    __all__.append(f.__name__)
    return f

# constants class that mimics scipy.constants
class constants(object):
    e = 1.6021766208e-19

@add_to__ODS__
def core_profiles_consistent(ods, update=True, use_electrons_density=False):
    '''
    Calls all core_profiles consistency functions including
      - core_profiles_pressures
      - core_profiles_densities
      - core_profiles_zeff

    :param ods: input ods

    :param update: operate in place

    :param use_electrons_density:
            denominator is core_profiles.profiles_1d.:.electrons.density
            instead of sum Z*n_i in Z_eff calculation

    :return: updated ods
    '''
    ods = core_profiles_pressures(ods, update=update)
    core_profiles_densities(ods, update=True)
    core_profiles_zeff(ods, update=True, use_electrons_density=use_electrons_density)
    return ods

@add_to__ODS__
def core_profiles_pressures(ods, update=True):
    '''
    Calculates individual ions pressures

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

        if not update:
            prof1d_p['grid']['rho_tor_norm'] = prof1d['grid']['rho_tor_norm']

        __zeros__ = 0.*prof1d['grid']['rho_tor_norm']

        prof1d_p['pressure_thermal']       = copy.deepcopy(__zeros__)
        prof1d_p['pressure_ion_total']     = copy.deepcopy(__zeros__)
        prof1d_p['pressure_perpendicular'] = copy.deepcopy(__zeros__)
        prof1d_p['pressure_parallel']      = copy.deepcopy(__zeros__)

        # electrons
        prof1d_p['electrons']['pressure'] = copy.deepcopy(__zeros__)

        __p__ = None
        if 'density_thermal' in prof1d['electrons'] and 'temperature' in prof1d['electrons']:
            __p__ = prof1d['electrons']['density_thermal'] * prof1d['electrons']['temperature'] * constants.e
        elif 'pressure_thermal' in prof1d['electrons']:
            __p__ = prof1d['electrons']['pressure_thermal']

        if __p__ is not None:
            prof1d_p['electrons']['pressure_thermal'] = __p__
            prof1d_p['electrons']['pressure']  += __p__
            prof1d_p['pressure_thermal']       += __p__
            prof1d_p['pressure_perpendicular'] += __p__ / 3.
            prof1d_p['pressure_parallel']      += __p__ / 3.

        if 'pressure_fast_perpendicular' in prof1d['electrons']:
            __p__ = prof1d['electrons']['pressure_fast_perpendicular']
            if not update:
                prof1d_p['electrons']['pressure_fast_perpendicular'] = __p__
            prof1d_p['electrons']['pressure']  += 2. * __p__
            prof1d_p['pressure_perpendicular'] += __p__

        if 'pressure_fast_parallel' in prof1d['electrons']:
            __p__ = prof1d['electrons']['pressure_fast_parallel']
            if not update:
                prof1d_p['electrons']['pressure_fast_parallel'] = __p__
            prof1d_p['electrons']['pressure'] += __p__
            prof1d_p['pressure_parallel']     += __p__

        #ions
        for k in range(len(prof1d['ion'])):

            prof1d_p['ion'][k]['pressure'] = copy.deepcopy(__zeros__)

            __p__ = None
            if 'density_thermal' in prof1d['ion'][k] and 'temperature' in prof1d['ion'][k]:
                __p__ = prof1d['ion'][k]['density_thermal'] * prof1d['ion'][k]['temperature'] * constants.e
            elif 'pressure_thermal' in prof1d['ion'][k]:
                __p__ = prof1d['ion'][k]['pressure_thermal']

            if __p__ is not None:
                prof1d_p['ion'][k]['pressure_thermal'] = __p__
                prof1d_p['ion'][k]['pressure']     += __p__
                prof1d_p['pressure_thermal']       += __p__
                prof1d_p['pressure_perpendicular'] += __p__ / 3.
                prof1d_p['pressure_parallel']      += __p__ / 3.
                prof1d_p['pressure_ion_total']     += __p__

            if 'pressure_fast_perpendicular' in prof1d['ion'][k]:
                __p__ = prof1d['ion'][k]['pressure_fast_perpendicular']
                if not update:
                    prof1d_p['ion'][k]['pressure_fast_perpendicular'] = __p__
                prof1d_p['ion'][k]['pressure']     += 2. * __p__
                prof1d_p['pressure_perpendicular'] += __p__

            if 'pressure_fast_parallel' in prof1d['ion'][k]:
                __p__ = prof1d['ion'][k]['pressure_fast_parallel']
                if not update:
                    prof1d_p['ion'][k]['pressure_fast_parallel'] = __p__
                prof1d_p['ion'][k]['pressure'] +=  __p__
                prof1d_p['pressure_parallel']  += __p__

        #extra pressure information that is not within IMAS structure is set only if consistency_check is not True
        if ods_p.consistency_check is not True:
            prof1d_p['pressure'] = prof1d_p['pressure_perpendicular'] * 2 + prof1d_p['pressure_parallel']
            prof1d_p['pressure_electron_total'] = prof1d_p['pressure_thermal'] - prof1d_p['pressure_ion_total']
            prof1d_p['pressure_fast'] = prof1d_p['pressure'] - prof1d_p['pressure_thermal']

    return ods_p

@add_to__ODS__
def core_profiles_densities(ods, update=True):
    '''
    calculates density from density_thermal and density_fast

    :param ods: input ods

    :param update: operate in place

    :return: updated ods
    '''

    ods_n = ods
    if not update:
        from omas import ODS
        ods_n = ODS().copy_attrs_from(ods)

    for time_index in ods['core_profiles']['profiles_1d']:
        prof1d = ods['core_profiles']['profiles_1d'][time_index]
        prof1d_n = ods_n['core_profiles']['profiles_1d'][time_index]

        if not update:
            prof1d_n['grid']['rho_tor_norm'] = prof1d['grid']['rho_tor_norm']

        __zeros__ = 0.*prof1d['grid']['rho_tor_norm']

        # electrons
        prof1d_n['electrons']['density'] = copy.deepcopy(__zeros__)
        for density in ['density_thermal','density_fast']:
            if density in prof1d['electrons']:
                 prof1d_n['electrons']['density'] += prof1d['electrons'][density]

        # ions
        for k in range(len(prof1d['ion'])):
            prof1d_n['ion'][k]['density'] = copy.deepcopy(__zeros__)
            for density in ['density_thermal','density_fast']:
                if density in prof1d['ion'][k]:
                    prof1d_n['ion'][k]['density'] += prof1d['ion'][k][density]
    return ods_n

@add_to__ODS__
def core_profiles_zeff(ods, update=True, use_electrons_density=False):
    '''
    calculates effective charge

    :param ods: input ods

    :param update: operate in place

    :param use_electrons_density:
            denominator core_profiles.profiles_1d.:.electrons.density
            instead of sum Z*n_i

    :return: updated ods
    '''

    ods_z = core_profiles_densities(ods,update=update)

    for time_index in ods['core_profiles']['profiles_1d']:
        prof1d = ods['core_profiles']['profiles_1d'][time_index]
        prof1d_z = ods_z['core_profiles']['profiles_1d'][time_index]

        Z2n = 0.*prof1d_z['grid']['rho_tor_norm']
        Zn  = 0.*prof1d_z['grid']['rho_tor_norm']

        for k in range(len(prof1d['ion'])):
            Z = prof1d['ion'][k]['element'][0]['z_n'] # from old ODS
            n = prof1d_z['ion'][k]['density']         # from new ODS
            Z2n += n*Z**2
            Zn  += n*Z
            if use_electrons_density:
                prof1d_z['zeff'] = Z2n/prof1d_z['electrons']['density']
            else:
                prof1d_z['zeff'] = Z2n/Zn
    return ods_z

@add_to__ODS__
def current_from_eq(ods, time_index):
    """
    This function sets the currents in ods['core_profiles']['profiles_1d'][time_index]
    using ods['equilibrium']['time_slice'][time_index]['profiles_1d']['j_tor']

    :param ods: ODS to update in-place

    :param time_index: ODS time index to updated

    """

    rho = ods['equilibrium.time_slice'][time_index]['profiles_1d.rho_tor_norm']

    with omas_environment(ods, coordsio={'core_profiles.profiles_1d.%d.grid.rho_tor_norm' % time_index: rho}):
        # call all current ohmic to start
        fsa_invR = ods['equilibrium']['time_slice'][time_index]['profiles_1d']['gm9']
        JtoR_tot = ods['equilibrium']['time_slice'][time_index]['profiles_1d']['j_tor'] * fsa_invR
        if 'core_profiles.vacuum_toroidal_field.b0' in ods:
            B0 = ods['core_profiles']['vacuum_toroidal_field']['b0'][time_index]
        elif 'equilibrium.vacuum_toroidal_field.b0' in ods:
            R0 = ods['equilibrium']['vacuum_toroidal_field']['r0']
            B0 = ods['equilibrium']['vacuum_toroidal_field']['b0'][time_index]
            ods['core_profiles']['vacuum_toroidal_field']['r0'] = R0
            ods.set_time_array('core_profiles.vacuum_toroidal_field.b0', time_index, B0)

        JparB_tot = transform_current(rho, JtoR=JtoR_tot,
                                      equilibrium=ods['equilibrium']['time_slice'][time_index],
                                      includes_bootstrap=True)

    try:
        core_profiles_currents(ods, time_index, rho, j_total=JparB_tot / B0)
    except AssertionError:
        # redo but wipe out old current components since we can't make it consistent
        core_profiles_currents(ods, time_index, rho,
                               j_actuator=None, j_bootstrap=None,
                               j_ohmic=None, j_non_inductive=None,
                               j_total=JparB_tot / B0)

    return

@add_to__ODS__
def core_profiles_currents(ods, time_index, rho_tor_norm,
                           j_actuator='default', j_bootstrap='default',
                           j_ohmic='default', j_non_inductive='default',
                           j_total='default', warn=True):
    """
    This function sets currents in ods['core_profiles']['profiles_1d'][time_index]

    If provided currents are inconsistent with each other or ods, ods is not updated and an error is thrown.

    Updates integrated currents in ods['core_profiles']['global_quantities']
    (N.B.: `equilibrium` IDS is required for evaluating j_tor and integrated currents)

    :param ods: ODS to update in-place

    :param time_index: ODS time index to updated

    :param rho_tor_norm:  normalized rho grid upon which each j is given

    For each j:
      - ndarray: set in ods if consistent
      - 'default': use value in ods if present, else set to None
      - None: try to calculate from currents; delete from ods if you can't

    :param j_actuator: Non-inductive, non-bootstrap current <J.B>/B0
        N.B.: used for calculating other currents and consistency, but not set in ods

    :param j_bootstrap: Bootstrap component of <J.B>/B0

    :param j_ohmic: Ohmic component of <J.B>/B0

    :param j_non_inductive: Non-inductive component of <J.B>/B0
        Consistency requires j_non_inductive = j_actuator + j_bootstrap, either
        as explicitly provided or as computed from other components.

    :param j_total: Total <J.B>/B0
        Consistency requires j_total = j_ohmic + j_non_inductive either as
        explicitly provided or as computed from other components.
    """

    from scipy.integrate import cumtrapz

    prof1d = ods['core_profiles']['profiles_1d'][time_index]

    # SETUP DEFAULTS
    data = {}
    with omas_environment(ods, coordsio={'core_profiles.profiles_1d.%d.grid.rho_tor_norm' % time_index: rho_tor_norm}):
        for j in ['j_actuator', 'j_bootstrap', 'j_non_inductive', 'j_ohmic', 'j_total']:
            if isinstance(eval(j), basestring) and eval(j) == 'default':
                if j in prof1d:
                    data[j] = copy.deepcopy(prof1d[j])
                elif (j == 'j_actuator') and (('j_bootstrap' in prof1d) and ('j_non_inductive' in prof1d)):
                    data['j_actuator'] = prof1d['j_non_inductive'] - prof1d['j_bootstrap']
                else:
                    data[j] = None
            else:
                data[j] = eval(j)
    j_actuator = data['j_actuator']
    j_bootstrap = data['j_bootstrap']
    j_ohmic = data['j_ohmic']
    j_non_inductive = data['j_non_inductive']
    j_total = data['j_total']

    # =================
    # UPDATE FORWARD
    # =================

    # j_non_inductive
    if (j_actuator is not None) and (j_bootstrap is not None):
        if j_non_inductive is None:
            j_non_inductive = j_actuator + j_bootstrap

    # j_total
    if (j_ohmic is not None) and (j_non_inductive is not None):
        if j_total is None:
            j_total = j_ohmic + j_non_inductive

    # get some quantities we'll use below
    if 'equilibrium.time_slice.%d' % time_index in ods:
        eq = ods['equilibrium']['time_slice'][time_index]
        if 'core_profiles.vacuum_toroidal_field.b0' in ods:
            B0 = ods['core_profiles']['vacuum_toroidal_field']['b0'][time_index]
        elif 'equilibrium.vacuum_toroidal_field.b0' in ods:
            R0 = ods['equilibrium']['vacuum_toroidal_field']['r0']
            B0 = ods['equilibrium']['vacuum_toroidal_field']['b0'][time_index]
            ods['core_profiles']['vacuum_toroidal_field']['r0'] = R0
            ods.set_time_array('core_profiles.vacuum_toroidal_field.b0', time_index, B0)
        fsa_invR = numpy.interp(rho_tor_norm, eq['profiles_1d']['rho_tor_norm'], eq['profiles_1d']['gm9'])
    else:
        # can't do any computations with the equilibrium
        if warn:
            printe("Warning: ods['equilibrium'] does not exist: Can't convert between j_total and j_tor or calculate integrated currents")
        eq = None

    # j_tor
    if (j_total is not None) and (eq is not None):
        JparB_tot = j_total * B0
        JtoR_tot = transform_current(rho_tor_norm, JparB=JparB_tot, equilibrium=eq, includes_bootstrap=True)
        j_tor = JtoR_tot / fsa_invR
    else:
        j_tor = None

    # =================
    # UPDATE BACKWARD
    # =================

    if j_total is not None:

        # j_non_inductive
        if (j_non_inductive is None) and (j_ohmic is not None):
            j_non_inductive = j_total - j_ohmic

        # j_ohmic
        elif (j_ohmic is None) and (j_non_inductive is not None):
            j_ohmic = j_total - j_non_inductive

    if j_non_inductive is not None:

        # j_actuator
        if (j_actuator is None) and (j_bootstrap is not None):
            j_actuator = j_non_inductive - j_bootstrap

        # j_bootstrap
        if (j_bootstrap is None) and (j_actuator is not None):
            j_bootstrap = j_non_inductive - j_actuator

    # ===============
    # CONSISTENCY?
    # ===============

    if (j_actuator is not None) and (j_bootstrap is None):
        err = "Cannot set j_actuator without j_bootstrap provided or calculable"
        raise RuntimeError(err)

    # j_non_inductive
    err = 'j_non_inductive inconsistent with j_actuator and j_bootstrap'
    if ((j_non_inductive is not None) and ((j_actuator is not None) or (j_bootstrap is not None))):
        assert numpy.allclose(j_non_inductive, j_actuator + j_bootstrap), err

    # j_total
    err = 'j_total inconsistent with j_ohmic and j_non_inductive'
    if ((j_total is not None) and ((j_ohmic is not None) or (j_non_inductive is not None))):
        assert numpy.allclose(j_total, j_ohmic + j_non_inductive), err

    # j_tor
    err = 'j_tor inconsistent with j_total'
    if (j_total is not None) and (j_tor is not None):
        if eq is not None:
            JparB_tot = j_total * B0
            JtoR_tot = transform_current(rho_tor_norm, JparB=JparB_tot, equilibrium=eq, includes_bootstrap=True)
            assert numpy.allclose(j_tor, JtoR_tot / fsa_invR), err
        else:
            if warn:
                printe("Warning: ods['equilibrium'] does not exist")
                printe("         can't determine if " + err)

    # =============
    # UPDATE ODS
    # =============

    with omas_environment(ods, coordsio={'core_profiles.profiles_1d.%d.grid.rho_tor_norm' % time_index: rho_tor_norm}):
        for j in ['j_bootstrap', 'j_non_inductive', 'j_ohmic', 'j_total', 'j_tor']:
            if eval(j) is not None:
                prof1d[j] = eval(j)
            elif j in prof1d:
                del prof1d[j]

    # ======================
    # INTEGRATED CURRENTS
    # ======================

    if eq is None:
        # can't integrate currents without the equilibrium
        return

    # Calculate integrated currents
    rho_eq = eq['profiles_1d']['rho_tor_norm']
    vp = eq['profiles_1d']['dvolume_dpsi']
    psi = eq['profiles_1d']['psi']
    fsa_invR = eq['profiles_1d']['gm9']
    with omas_environment(ods, coordsio={'core_profiles.profiles_1d.%d.grid.rho_tor_norm' % time_index: rho_eq}):

        currents = [('j_bootstrap', 'current_bootstrap', True),
                    ('j_non_inductive', 'current_non_inductive', True),
                    ('j_tor', 'ip', False)]

        for Jname, Iname, transform in currents:
            if Jname in prof1d:
                J = prof1d[Jname]
                if transform:
                    # transform <J.B>/B0 to <Jt/R>
                    J = transform_current(rho_eq, JparB=J * B0, equilibrium=eq, includes_bootstrap=True)
                else:
                    # already <Jt/R>/<1/R>
                    J *= fsa_invR
                ods.set_time_array('core_profiles.global_quantities.%s' % Iname, time_index,
                                   cumtrapz(vp * J, psi)[-1] / (2. * numpy.pi))
            elif 'core_profiles.global_quantities.%s' % Iname in ods:
                # set current to zero if this time_index exists already
                if time_index < len(ods['core_profiles.global_quantities.%s' % Iname]):
                    ods['core_profiles.global_quantities.%s' % Iname][time_index] = 0.

    return

@add_to__ALL__
def transform_current(rho, JtoR=None, JparB=None, equilibrium=None, includes_bootstrap=False):
    """
    Given <Jt/R> returns <J.B>, or vice versa
    Transformation obeys <J.B> = (1/f)*(<B^2>/<1/R^2>)*(<Jt/R> + dp/dpsi*(1 - f^2*<1/R^2>/<B^2>))
    N.B. input current must be in the same COCOS as equilibrium.cocosio

    :param rho: normalized rho grid for input JtoR or JparB

    :param JtoR: input <Jt/R> profile (cannot be set along with JparB)

    :param JparB: input <J.B> profile (cannot be set along with JtoR)

    :param equilibrium: equilibrium.time_slice[:] ODS containing quanities needed for transformation

    :param includes_bootstrap: set to True if input current includes bootstrap

    :return: <Jt/R> if JparB set or <J.B> if JtoR set

    Example: given total <Jt/R> on rho grid with an existing ods, return <J.B>
             JparB = transform_current(rho, JtoR=JtoR,
                                       equilibrium=ods['equilibrium']['time_slice'][0],
                                       includes_bootstrap=True)
    """

    if (JtoR is not None) and (JparB is not None):
        raise RuntimeError("JtoR and JparB cannot both be set")
    if equilibrium is None:
        raise ValueError("equilibrium ODS must be provided, specifically equilibrium.time_slice[:]")

    cocos = define_cocos(equilibrium.cocosio)
    rho_eq = equilibrium['profiles_1d.rho_tor_norm']
    fsa_B2 = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.gm5'])
    fsa_invR2 = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.gm1'])
    f = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.f'])
    dpdpsi = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.dpressure_dpsi'])

    # diamagnetic term to get included with bootstrap currrent
    JtoR_dia = dpdpsi * (1. - fsa_invR2 * f ** 2 / fsa_B2)
    JtoR_dia *= cocos['sigma_Bp'] * (2. * numpy.pi) ** cocos['exp_Bp']

    if JtoR is not None:
        Jout = fsa_B2 * (JtoR + includes_bootstrap * JtoR_dia) / (f * fsa_invR2)
    elif JparB is not None:
        Jout = f * fsa_invR2 * JparB / fsa_B2 - includes_bootstrap * JtoR_dia

    return Jout

@add_to__ALL__
def search_ion(ion_ods, label=None, Z=None, A=None, no_matches_raise_error=True, multiple_matches_raise_error=True):
    '''
    utility function used to identify the ion number and element numbers given the ion label and or their Z and/or A

    :param ion_ods: ODS location that ends with .ion

    :param label: ion label

    :param Z: ion element charge

    :param A: ion element mass

    :parame no_matches_raise_error: whether to raise a IndexError when no ion matches are found

    :parame multiple_matches_raise_error: whether to raise a IndexError when multiple ion matches are found

    :return: dictionary with matching ions labels, each with list of matching ion elements
    '''
    if not ion_ods.location.endswith('.ion'):
        raise (ValueError('ods location must end with `.ion`'))
    match = {}
    for ki in ion_ods:
        if label is None or (label is not None and 'label' in ion_ods[ki] and ion_ods[ki]['label'] == label):
            if A is not None or Z is not None and 'element' in ion_ods[ki]:
                for ke in ion_ods[ki]['element']:
                    if A is not None and A == ion_ods[ki]['element'][ke]['a'] and Z is not None and Z == ion_ods[ki]['element'][ke]['z_n']:
                        match.setdefault(ki, []).append(ke)
                    elif A is not None and A == ion_ods[ki]['element'][ke]['a']:
                        match.setdefault(ki, []).append(ke)
                    elif Z is not None and Z == ion_ods[ki]['element'][ke]['z_n']:
                        match.setdefault(ki, []).append(ke)
            elif 'element' in ion_ods[ki] and len(ion_ods[ki]['element']):
                match.setdefault(ki, []).extend(range(len(ion_ods[ki]['element'])))
            else:
                match[ki] = []
    if multiple_matches_raise_error and (len(match) > 1 or len(match) == 1 and len(list(match.values())[0]) > 1):
        raise (IndexError('Multiple ion match query: label=%s  Z=%s  A=%s' % (label, Z, A)))
    if no_matches_raise_error and len(match) == 0:
        raise (IndexError('No ion match query: label=%s  Z=%s  A=%s' % (label, Z, A)))
    return match

@add_to__ALL__
def search_in_array_structure(ods, conditions, no_matches_return=0, no_matches_raise_error=False, multiple_matches_raise_error=True):
    '''
    search for the index in an array structure that matches some conditions

    :param ods: ODS location that is an array of structures

    :param conditions: dictionary (or ODS) whith entries that must match and their values
                       * condition['name']=value  : check value
                       * condition['name']=True   : check existance
                       * condition['name']=False  : check not existance
                       NOTE: True/False as flags for (not)existance is not an issue since IMAS does not support booleans

    :param no_matches_return: what index to return if no matches are found

    :param no_matches_raise_error: wheter to raise an error in no matches are found

    :param multiple_matches_raise_error: whater to raise an error if multiple matches are found

    :return: list with indeces matching conditions
    '''

    if ods.omas_data is not None and not isinstance(ods.omas_data, list):
        raise (Exception('ods location must be an array of structures'))

    if isinstance(conditions, ODS):
        conditions = conditions.flat()

    match = []
    for k in ods:
        k_match = True
        for key in conditions:
            if conditions[key] is False:
                if key in ods[k]:
                    k_match = False
                    break
            elif conditions[key] is True:
                if key not in ods[k]:
                    k_match = False
                    break
            elif key not in ods[k] or ods[k][key] != conditions[key]:
                k_match = False
                break
        if k_match:
            match.append(k)

    if not len(match):
        if no_matches_raise_error:
            raise (IndexError('no matches for conditions: %s' % conditions))
        match = [no_matches_return]

    if multiple_matches_raise_error and len(match) > 1:
        raise (IndexError('multiple matches for conditions: %s' % conditions))

    return match

@add_to__ALL__
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

@add_to__ALL__
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
    transforms[None] = 1

    printd(transforms, topic='cocos')

    return transforms

@add_to__ALL__
@contextmanager
def omas_environment(ods, cocosio=None, coordsio=None, unitsio=None, **kw):
    '''
    Provides environment for data input/output to/from OMAS

    :param ods: ODS on which to operate

    :param cocosio: COCOS convention

    :param coordsio: dictionary/ODS with coordinates for data interpolation

    :param unitsio: True/False whether data read from OMAS should have units

    :param kw: extra keywords set attributes of the ods (eg. 'consistency_check','dynamic_path_creation','imas_version')

    :return: ODS with environmen set
    '''

    if isinstance(coordsio, dict):
        from omas import ODS
        tmp = ODS(cocos=ods.cocos)
        tmp.dynamic_path_creation='dynamic_array_structures'
        with omas_environment(tmp, cocosio=cocosio):
            tmp.update(coordsio)
        coordsio = tmp

    if cocosio is not None and not isinstance(cocosio,int):
        raise(ValueError('cocosio can only be an integer'))

    bkp_coordsio = ods.coordsio
    bkp_cocosio = ods.cocosio
    bkp_unitsio = ods.unitsio
    if cocosio is not None:
        ods.cocosio = cocosio
    if coordsio is not None:
        ods.coordsio = (ods, coordsio)
    if unitsio is not None:
        ods.unitsio = unitsio
    bkp_args = {}
    for item in kw:
        bkp_args[item] = getattr(ods, item)
        setattr(ods, item, kw[item])
    try:
        if coordsio is not None:
            with omas_environment(coordsio, cocosio=cocosio):
                yield ods
        else:
                yield ods
    finally:
        ods.cocosio = bkp_cocosio
        ods.coordsio = bkp_coordsio
        ods.unitsio = bkp_unitsio
        for item in kw:
            setattr(ods, item, bkp_args[item])

def generate_cocos_signals(structures=[], threshold=0):
    """
    This is a utility function for generating the omas_cocos.py Python file

    :param structures: list of structures for which to generate COCOS signals

    :param threshold: score threshold below which singals entries will not be written in omas_cocos.py
    * 0 is a reasonable threshold for catching signals that should have an associated COCOS transform
    * 10000 (or any high number) is a way to hide signals in omas_cocos.py that are unassigned

    :return: dictionary structure with tally of score and reason for scoring for every entry
    """
    # units of entries currently in cocos_singals
    cocos_units = []
    for item in cocos_signals:
        info = omas_info_node(item)
        if len(info): # info may have no length if nodes are deleted between IMAS versions
            units = info['units']
            if units not in cocos_units:
                cocos_units.append(units)
    cocos_units = set(cocos_units).difference(set(['?']))

    # make sure to keep around structures that are already in omas_cocos.py
    cocos_structures = []
    for item in _cocos_signals:
        structure_name = item.split('.')[0]
        if structure_name not in cocos_structures:
            cocos_structures.append(structure_name)

    if isinstance(structures, basestring):
        structures = [structures]
    structures += cocos_structures
    structures = numpy.unique(structures)

    from .omas_utils import _structures
    from .omas_utils import i2o
    from .omas_core import ODS
    ods = ODS()
    out = {}
    text = []
    csig=["'''List of automatic COCOS transformations\n\n-------\n'''",
          '# COCOS signals candidates are generated by running utilities/generate_cocos_signals.py',
          '# Running this script is useful to keep track of new signals that IMAS adds in new data structure releases',
          '#',
          '# In this file you are only allowed to edit/add entries to the `cocos_signals` dictionary',
          '# The comments indicate `#[ADD_OR_DELETE_SUGGESTION]# MATCHING_SCORE # RATIONALE_FOR_ADD_OR_DELETE`',
          '#',
          '# Proceed as follows:',
          '# 1. Edit transformations in this file (if a signal is missing, it can be added here)',
          '# 2. Run `utilities/generate_cocos_signals.py` (which will update this same file)',
          '# 3. Commit changes',
          '',
          'cocos_signals = {}']

    # loop over structures
    for structure in structures:
        text.extend(['', '# ' + structure.upper()])
        csig.extend(['', '# ' + structure.upper()])

        out[structure] = {}
        ods[structure]
        d = dict_structures(imas_version=omas_rcparams['default_imas_version'])
        m = 0
        # generate score and add reason for scoring
        for item in sorted(list(_structures[(structure,omas_rcparams['default_imas_version'])].keys())):
            item = i2o(item)
            item_ = item
            if any([item.endswith(k) for k in [':.values',':.value',':.data']]):
                item_ = l2o(p2l(item)[:-2])
            elif any([item.endswith(k) for k in ['.values','.value','.data']]):
                item_ = l2o(p2l(item)[:-1])
            m = max(m, len(item))
            score = 0
            rationale = []
            if item.startswith(structure) and '_error_' not in item:
                entry = "cocos_signals['%s']=" % i2o(item)
                info = omas_info_node(item)
                units = info.get('units', None)
                data_type = info.get('data_type', None)
                documentation = info.get('documentation', '')
                if data_type in ['structure', 'STR_0D', 'struct_array']:
                    continue
                elif units in [None, 's']:
                    out[structure].setdefault(-1, []).append((item, '[%s]' % units))
                    continue
                elif any([(item_.endswith('.'+k) or item_.endswith('_'+k) or '.'+k+'.' in item) for k in
                          ['chi_squared', 'standard_deviation', 'weight', 'coefficients', 'r', 'z', 'beta_tor',
                           'beta_pol', 'radial', 'rho_tor_norm', 'darea_drho_tor',
                           'dvolume_drho_tor','ratio','fraction','rate','d','flux','v','b_field_max','width_tor']]):
                    out[structure].setdefault(-1, []).append((item, p2l(item_)[-1]))
                    continue
                elif any([k in documentation for k in ['always positive']]):
                    out[structure].setdefault(-1, []).append((item, documentation))
                    continue
                n = item.count('.')
                for pnt, key in enumerate(p2l(item)):
                    pnt = pnt / float(n)
                    for k in ['q', 'ip', 'b0', 'phi', 'psi', 'f', 'f_df']:
                        if key == k:
                            rationale += [k]
                            score += pnt
                            break
                    for k in ['q', 'j', 'phi', 'psi', 'ip', 'b', 'f', 'v', 'f_df']:
                        if key.startswith('%s_' % k) and not any([key.startswith(k) for k in ['psi_norm']]):
                            rationale += [k]
                            score += pnt
                            break
                    for k in ['velocity', 'current', 'b_field', 'e_field', 'torque', 'momentum']:
                        if k in key and key not in ['heating_current_drive']:
                            rationale += [k]
                            score += pnt
                            break
                    for k in ['_dpsi']:
                        if k in key and k + '_norm' not in key:
                            rationale += [k]
                            score += pnt
                            break
                    for k in ['poloidal', 'toroidal', 'parallel', '_tor', '_pol', '_par', 'tor_', 'pol_', 'par_']:
                        if ((key.endswith(k) or key.startswith(k)) and not
                            any([key.startswith(k) for k in ['conductivity_', 'pressure_', 'rho_', 'length_']])):
                            rationale += [k]
                            score += pnt
                            break
                if units in cocos_units:
                    if len(rationale):
                        rationale += ['[%s]' % units]
                        score += 1
                out[structure].setdefault(score, []).append((item, '  '.join(rationale)))

        # generate output
        for score in reversed(sorted(out[structure])):
            for item, rationale in out[structure][score]:

                message = '       '
                if cocos_signals.get(item, '?') == '?':
                    if score > 0:
                        message = '#[ADD?]'
                    else:
                        message = '#[DEL?]'
                elif score < 0:
                    message = '#[DEL?]'

                txt = ("cocos_signals['%s']='%s'" % (item, cocos_signals.get(item, '?'))).ljust(m + 20) + message + '# %f # %s' % (score, rationale)
                text.append(txt)
                if score > threshold or (item in cocos_signals and cocos_signals[item] != '?'):
                    csig.append(txt)

        # write omas_cocos.py
        filename = os.path.abspath(str(os.path.dirname(__file__)) + '/omas_cocos.py')
        with open(filename, 'w') as f:
            f.write('\n'.join(csig))

        # print to screen (note that this prints ALL the entries, whereas omas_cocos.py only contains entries that score above a give threshold)
        print('\n'.join(text) + '\n\n' + '-' * 20 + '\n\nUpdated ' + filename)

    return out

# cocos_signals contains the IMAS locations and the corresponding `cocos_transform` function
from .omas_cocos import cocos_signals as _cocos_signals

# The CocosSignals class is just a dictionary that raises warnings when users access
# entries that are likely to need a COCOS transformation, but do not have one.
class CocosSignals(dict):
    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if value == '?':
            warnings.warn('`%s` may require defining its COCOS transform in omas/omas_cocos.py')
        return value

# cocos_signals is the actual dictionary
cocos_signals = CocosSignals()
cocos_signals.update(_cocos_signals)
