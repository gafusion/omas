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
def update_current(ods, time_index, j_ohmic=None, j_bootstrap=None,
                   j_non_inductive=None, j_total = None, j_tor = None):
    """
    This function:
        - Sets the given currents in ods['core_profiles']['profiles_1d'][time_index]
        - Updates j_non_inductive, j_total, and/or j_tor if they are not
            explicitly provided and either sufficient information is
            in the ODS or set equal to 'update'
        - Updates integrated currents in ods['core_profiles']['global_quantities']

    :param ods: ODS to update in-place

    :param time_index: ODS time index to updated

    :param j_ohmic: Ohmic component of <J.B>/B0
                    Set to ods['core_profiles']['profiles_1d'][time_index]['j_ohmic']

    :param j_bootstrap: Bootstrap component of <J.B>/B0
                        Set to ods['core_profiles']['profiles_1d'][time_index]['j_bootstrap']

    :param j_non_inductive: Non-inductive component of <J.B>/B0
                            Set to ods['core_profiles']['profiles_1d'][time_index]['j_non_inductive']
                            'update' forces j_non_inductive to be updated with new bootstrap current

    :param j_total: Total <J.B>/B0
                    Set to ods['core_profiles']['profiles_1d'][time_index]['j_total']
                    'update' forces 'j_total' to be updated with new ohmic or non_inductive currents

    :param j_tor: Total <Jt/R>/<1/R>
                  Set to ods['core_profiles']['profiles_1d'][time_index]['j_tor']

    """

    from scipy.integrate import cumtrapz

    prof1d = ods['core_profiles']['profiles_1d'][time_index]

    # save existing current values
    j_old = {}
    for j in ['j_ohmic', 'j_bootstrap', 'j_non_inductive', 'j_total', 'j_tor']:
        if j in prof1d:
            j_old[j] = copy.deepcopy(prof1d[j])

    # Ohmic current
    if isinstance(j_ohmic, numpy.ndarray):
        prof1d['j_ohmic'] = j_ohmic

    # Bootstrap current
    if isinstance(j_bootstrap, numpy.ndarray):
        prof1d['j_bootstrap'] = j_bootstrap

    # Total non-inductive current
    if isinstance(j_non_inductive, numpy.ndarray):
        # use the provided current
        prof1d['j_non_inductive'] = j_non_inductive
    elif 'j_bootstrap' in prof1d:
        # update j_non_inductive with latest bootstrap current
        if 'j_non_inductive' in prof1d:
            if ('j_bootstrap' in j_old) or (j_non_inductive=='update'):
                prof1d['j_non_inductive'] += prof1d['j_bootstrap']
            if 'j_bootstrap' in j_old:
                prof1d['j_non_inductive'] -= j_old['j_bootstrap']
        else:
            prof1d['j_non_inductive'] = prof1d['j_bootstrap']

    # Total parallel current
    if isinstance(j_total, numpy.ndarray):
        # use the provided current
        prof1d['j_total'] = j_total
    else:
        # update total current with latest currents
        for j in ['j_ohmic', 'j_non_inductive']:
            if j in prof1d:
                if 'j_total' in prof1d:
                    if (j in j_old) or (j_total == 'update'):
                        prof1d['j_total'] += prof1d[j]
                    if j in j_old:
                        prof1d['j_total'] -= j_old[j]
                else:
                    prof1d['j_total'] = prof1d[j]

    # get some quantities we'll use below
    eq = ods['equilibrium']['time_slice'][time_index]
    if 'core_profiles.vacuum_toroidal_field.b0' in ods:
        B0 = ods['core_profiles']['vacuum_toroidal_field']['b0'][time_index]
    elif 'equilibrium.vacuum_toroidal_field.b0' in ods:
        R0 = ods['equilibrium']['vacuum_toroidal_field']['r0']
        B0 = ods['equilibrium']['vacuum_toroidal_field']['b0'][time_index]
        ods['core_profiles']['vacuum_toroidal_field']['r0'] = R0
        ods.set_time_array('core_profiles.vacuum_toroidal_field.b0', time_index, B0)
    rho = prof1d['grid']['rho_tor_norm']
    fsa_invR = numpy.interp(rho, eq['profiles_1d']['rho_tor_norm'],
                            eq['profiles_1d']['gm9'])

    # Total toroidal current
    if isinstance(j_tor, numpy.ndarray):
        # use the provided current
        prof1d['j_tor'] = j_tor

        if j_total is None:
            JtoR_tot = j_tor*fsa_invR
            JparB_tot = transform_current(rho, JtoR=JtoR_tot,
                                          equilibrium=eq,
                                          includes_bootstrap=True)
            prof1d['j_total'] = JparB_tot/B0

    elif 'j_total' in prof1d:
        # update toroidal current using transformation
        JparB_tot = prof1d['j_total']*B0
        JtoR_tot = transform_current(rho, JparB=JparB_tot,
                                     equilibrium=eq,
                                     includes_bootstrap=True)
        prof1d['j_tor'] = JtoR_tot/fsa_invR


    # Calculate integrated currents
    rho_eq   = eq['profiles_1d']['rho_tor_norm']
    vp       = eq['profiles_1d']['dvolume_dpsi']
    psi      = eq['profiles_1d']['psi']
    fsa_invR = eq['profiles_1d']['gm9']
    with omas_environment(ods,
                          coordsio={'core_profiles.profiles_1d.%d.grid.rho_tor_norm'%time_index:rho_eq}):

        currents = [('j_bootstrap', 'current_bootstrap', True),
                    ('j_non_inductive', 'current_non_inductive', True),
                    ('j_tor', 'ip', False)]

        for Jname, Iname, transform in currents:
            if Jname in prof1d:
                J = prof1d[Jname]
            else:
                J = 0.*rho_eq
            if transform:
                # transform <J.B>/B0 to <Jt/R>
                J = transform_current(rho_eq, JparB=J*B0,
                                      equilibrium=eq, includes_bootstrap=True)
            else:
                # already <Jt/R>/<1/R>
                J *= fsa_invR
            ods.set_time_array('core_profiles.global_quantities.%s'%Iname,time_index,
                               cumtrapz(vp*J,psi)[-1]/(2.*numpy.pi))

    return

def transform_current(rho, JtoR=None, JparB=None,
                      equilibrium=None, includes_bootstrap=False):
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
    fsa_B2    = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.gm5'])
    fsa_invR2 = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.gm1'])
    f         = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.f'])
    dpdpsi    = numpy.interp(rho, rho_eq, equilibrium['profiles_1d.dpressure_dpsi'])

    # diamagnetic term to get included with bootstrap currrent
    JtoR_dia = dpdpsi*(1. - fsa_invR2*f**2/fsa_B2)
    JtoR_dia *= cocos['sigma_Bp']*(2.*numpy.pi)**cocos['exp_Bp']

    if JtoR is not None:
        Jout = fsa_B2*(JtoR + includes_bootstrap*JtoR_dia)/(f*fsa_invR2)
    elif JparB is not None:
        Jout = f*fsa_invR2*JparB/fsa_B2 - includes_bootstrap*JtoR_dia

    return Jout


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
    transforms[None] = 1

    printd(transforms, topic='cocos')

    return transforms

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
        units = omas_info_node(item)['units']
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
    csig=['# This file contains the list of automatic COCOS transformations',
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
        d = dict_structures(imas_version=default_imas_version)
        m = 0
        # generate score and add reason for scoring
        for item in sorted(list(list(_structures[d[structure]].keys()))):
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

#=======================
# BACKWARD COMPATIBILITY
#=======================

@contextmanager
def coords_environment(ods, coordsio=None):
    '''
    DEPRECATED: use omas_environment(ods, coordsio=...) instead

    Provides OMAS environment within wich coordinates are interpolated

    :param ods: ODS on which to operate

    :param coordsio: dictionary of coordinates

    :return: ODS with coordinate interpolations set
    '''
    warnings.warn('coords_environment is deprecated. Use omas_environment(ods, coordsio=...) instead.')
    with omas_environment(ods, coordsio=coordsio):
        yield ods

@contextmanager
def cocos_environment(ods, cocosio=None):
    '''
    DEPRECATED: use omas_environment(ods, cocosio=...) instead

    Provides OMAS environment within wich a certain COCOS convention is defined

    :param ods: ODS on which to operate

    :param cocosio: input/output COCOS convention

    :return: ODS with COCOS convention set
    '''
    warnings.warn('cocos_environment is deprecated. Use omas_environment(ods, cocosio=...) instead.')
    with omas_environment(ods, cocosio=cocosio):
        yield ods
