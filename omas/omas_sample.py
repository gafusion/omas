from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS
from .omas_physics import constants
from .omas_plot import geo_type_lookup

__all__ = []


def add_to_ODS(f):
    __all__.append(f.__name__)
    return f


def ods_sample():
    '''
    returns an ODS populated with all of the samples

    :return: sample ods
    '''
    ods = ODS()
    for item in __all__:
        printd('Adding %s sample data to ods' % item, topic='sample')
        ods = eval(item)(ods)
    return ods


@add_to_ODS
def misc(ods):
    """
    create sample ODS data
    """

    #check effect of disabling dynamic path creation
    try:
        ods.dynamic_path_creation = False
        ods['info.user']
    except LookupError:
        ods['info'] = ODS()
        ods['info.user'] = unicode(os.environ['USER'])
    else:
        raise(Exception('OMAS error handling dynamic_path_creation=False'))
    finally:
        ods.dynamic_path_creation = True

    #check that accessing leaf that has not been set raises a ValueError, even with dynamic path creation turned on
    try:
        ods['info.machine']
    except ValueError:
        pass
    else:
        raise(Exception('OMAS error querying leaf that has not been set'))

    # info ODS is used for keeping track of IMAS metadata
    ods['info.machine'] = 'ITER'
    ods['info.imas_version'] = default_imas_version
    ods['info.shot'] = 1
    ods['info.run'] = 0

    # check .get() method
    assert (ods.get('info.shot') == ods['info.shot'])
    assert (ods.get('info.bad', None) is None)

    # check that keys is an iterable (so that Python 2/3 work the same way)
    keys = ods.keys()
    keys[0]

    # check that dynamic path creation during __getitem__ does not leave empty fields behind
    try:
        print(ods['wall.description_2d.0.limiter.unit.0.outline.r'])
    except ValueError:
        assert 'wall.description_2d.0.limiter.unit.0.outline' not in ods

    ods['equilibrium']['time_slice'][0]['time'] = 1000.
    ods['equilibrium']['time_slice'][0]['global_quantities']['ip'] = 1.5

    ods2 = copy.deepcopy(ods)
    ods2['equilibrium']['time_slice'][1] = ods['equilibrium']['time_slice'][0]
    ods2['equilibrium.time_slice.1.time'] = 2000.

    ods2['equilibrium']['time_slice'][2] = copy.deepcopy(ods['equilibrium']['time_slice'][0])
    ods2['equilibrium.time_slice[2].time'] = 3000.

    assert(ods2['equilibrium']['time_slice'][0]['global_quantities'].ulocation==ods2['equilibrium']['time_slice'][2]['global_quantities'].ulocation)

    ods2['equilibrium.time_slice.1.global_quantities.ip'] = 2.

    # check different ways of addressing data
    for item in [ods2['equilibrium.time_slice']['1.global_quantities'],
                 ods2[['equilibrium', 'time_slice', 1, 'global_quantities']],
                 ods2[('equilibrium', 'time_slice', 1, 'global_quantities')],
                 ods2['equilibrium.time_slice.1.global_quantities'],
                 ods2['equilibrium.time_slice[1].global_quantities']]:
        assert item.ulocation=='equilibrium.time_slice.:.global_quantities'

    ods2['equilibrium.time_slice.0.profiles_1d.psi'] = numpy.linspace(0, 1, 10)

    # pprint(ods.paths())
    # pprint(ods2.paths())

    # check data slicing
    assert numpy.all(ods2['equilibrium.time_slice[:].global_quantities.ip']==numpy.array([1.5,2.0,1.5]))

    # uncertain scalar
    ods2['equilibrium.time_slice[2].global_quantities.ip'] = ufloat(3,0.1)

    # uncertain array
    ods2['equilibrium.time_slice[2].profiles_1d.q'] = uarray([0.,1.,2.,3.],[0,.1,.2,.3])

    ckbkp = ods.consistency_check
    tmp = pickle.dumps(ods2)
    ods2 = pickle.loads(tmp)
    if ods2.consistency_check != ckbkp:
        raise (Exception('consistency_check attribute changed'))

    # check flattening
    tmp = ods2.flat()
    # pprint(tmp)

    # check deepcopy
    ods3=ods2.copy()

    return ods3


@add_to_ODS
def equilibrium(ods, time_index=0, include_profiles=False, include_phi=False, include_wall=True):
    """
    Expands an ODS by adding a (heavily down-sampled) psi map to it with low precision. This function can overwrite
    existing data if you're not careful. The original is modified, so deepcopy first if you want different ODSs.

    :param ods: ODS instance

    :param time_index: int
        Under which time index should fake equilibrium data be loaded?

    :param include_profiles: bool
        Include 1D profiles of pressure, q, p', FF'

    :param include_phi: bool
        Include 1D and 2D profiles of phi (toroidal flux, for calculating rho)

    :param include_wall: bool
        Include the first wall

    :return: ODS instance with equilibrium data added
        Since the original is modified, it is not necessary to catch the return, but it may be convenient to do so in
        some contexts. If you do not want the original to be modified, deepcopy it first.
    """

    # These arrays were decimated to make them smaller; we don't need something nice looking for these tests and we
    # don't want to take up space storing huge arrays when little ones will do. Data originally from shot 173225.
    psi_small = numpy.array([
        [0.37, 0.22, 0.07, 0.07, 0.17, 0.15, 0.32, 0.63, 0.81, 1.01, 1.13],
        [0.29, 0.22, 0.24, 0.22, 0.25, 0.33, 0.49, 0.77, 1.08, 1.59, 1.83],
        [0.53, 0.43, 0.35, 0.26, 0.25, 0.3, 0.44, 0.74, 1.08, 1.72, 2.02],
        [0.75, 0.56, 0.35, 0.16, 0.03, -0.02, 0.09, 0.36, 0.74, 1.22, 1.69],
        [0.7, 0.51, 0.24, -0.06, -0.34, -0.53, -0.53, -0.26, 0.21, 0.84, 1.55],
        [0.72, 0.48, 0.14, -0.26, -0.67, -0.99, -1.08, -0.82, -0.31, 0.42, 1.02],
        [0.71, 0.47, 0.13, -0.27, -0.68, -1., -1.1, -0.85, -0.35, 0.35, 0.97],
        [0.62, 0.45, 0.21, -0.07, -0.33, -0.51, -0.52, -0.29, 0.14, 0.7, 1.31],
        [0.59, 0.48, 0.34, 0.2, 0.09, 0.05, 0.13, 0.38, 0.71, 1.17, 1.5],
        [0.48, 0.44, 0.43, 0.4, 0.42, 0.46, 0.58, 0.82, 1.11, 1.67, 1.9],
        [0.46, 0.44, 0.5, 0.55, 0.57, 0.61, 0.72, 0.9, 1.11, 1.46, 1.6],
    ])
    grid1_small = numpy.array([0.83, 0.99, 1.15, 1.3, 1.46, 1.62, 1.77, 1.94, 2.09, 2.25, 2.4])
    grid2_small = numpy.array([-1.58, -1.29, -0.99, -0.69, -0.39, -0.1, 0.2, 0.5, 0.79, 1.1, 1.38])
    bdry_r_small = numpy.array([1.08, 1.12, 1.27, 1.54, 1.82, 2.03, 2.2, 2.22, 2.07, 1.9, 1.68, 1.45, 1.29, 1.16, 1.1])
    bdry_z_small = numpy.array([1.00e-03, 5.24e-01, 8.36e-01, 9.42e-01, 8.37e-01, 6.49e-01, 3.46e-01, -9.38e-02,
                                -4.57e-01, -6.89e-01, -8.93e-01, -1.07e+00, -9.24e-01, -5.16e-01, -1.00e-01])
    wall_r_small = numpy.array([1.01, 1., 1.01, 1.09, 1.17, 1.2, 1.23, 1.31, 1.37, 1.36, 1.42, 1.5, 1.46, 1.54, 2.05,
                                2.41, 2.2, 1.64, 1.1, 1.03])
    wall_z_small = numpy.array([-0., 1.21, 1.12, 1.17, 1.19, 1.17, 1.29, 1.31, 1.32, 1.16, 1.18, 1.23, 1.1, 1.14, 0.81,
                                0.09, -0.59, -1.27, -1.3, -0.38])

    ods['equilibrium.time_slice'][time_index]['profiles_1d.psi'] = numpy.linspace(0, 1, 11)
    ods['equilibrium.time_slice'][time_index]['profiles_1d.pressure'] = (1-numpy.linspace(0, 1, 11))**2
    ods['equilibrium.time_slice'][time_index]['profiles_2d.0.psi'] = psi_small
    ods['equilibrium.time_slice'][time_index]['profiles_2d.0.grid.dim1'] = grid1_small
    ods['equilibrium.time_slice'][time_index]['profiles_2d.0.grid.dim2'] = grid2_small
    ods['equilibrium.time_slice'][time_index]['boundary.outline.r'] = bdry_r_small
    ods['equilibrium.time_slice'][time_index]['boundary.outline.z'] = bdry_z_small
    ods['equilibrium.time_slice'][time_index]['time'] = 2.0

    ods['equilibrium.time_slice'][time_index]['global_quantities.magnetic_axis.r'] = 1.77
    ods['equilibrium.time_slice'][time_index]['global_quantities.magnetic_axis.z'] = 0.05

    if include_profiles:
        ods['equilibrium.time_slice'][time_index]['profiles_1d.pressure'] = numpy.array([
            102419.25, 86877.77, 72251.59, 59232.62, 47208.7, 36590.41, 27260.38, 18938.6, 12319.77, 6356.33, 2239.37])
        ods['equilibrium.time_slice'][time_index]['profiles_1d.q'] = numpy.array([
            -0.92, -1.08, -1.08, -1.26, -1.32, -1.42, -1.75, -1.68, -2.43, -2.28, -3.75])
        ods['equilibrium.time_slice'][time_index]['profiles_1d.dpressure_dpsi'] = numpy.array([
            -89728.68, -82951.89, -75891.34, -69131.76, -62145.72, -55283.47, -48456.6, -41369.76, -34742.06, -27488.37,
            -20863.08])
        ods['equilibrium.time_slice'][time_index]['profiles_1d.f_df_dpsi'] = numpy.array([
            -0.31, -0.27, -0.23, -0.19, -0.15, -0.12, -0.09, -0.07, -0.05, -0.03, -0.02])

    if include_phi:
        ods['equilibrium.time_slice'][time_index]['profiles_1d.phi'] = numpy.array([
            1.11e-05, -1.77e-01, -3.77e-01, -5.85e-01, -8.19e-01, -1.07e+00, -1.36e+00, -1.68e+00, -2.05e+00, -2.48e+00,
            -3.03e+00])
        ods['equilibrium.time_slice'][time_index]['profiles_2d.0.phi'] = numpy.array([
            [-4.09, -2.65, -1.87, -1.86, -2.54, -2.63, -4.17, -7.24, -9.35, -11.34, -12.54],
            [-4.48, -3.37, -3.14, -2.92, -3.3, -4.16, -5.91, -8.89, -12.19, -17.83, -20.1],
            [-5.97, -5., -4.21, -3.34, -3.42, -4.04, -5.57, -8.7, -12.09, -18.82, -21.72],
            [-7.06, -5.69, -4., -2.73, -2.19, -2.14, -2.78, -4.74, -7.78, -11.38, -14.28],
            [-7.25, -5.49, -3.21, -1.79, -1.12, -0.7, -0.76, -1.32, -3.1, -6.86, -11.06],
            [-7.24, -5.16, -2.74, -1.31, -0.6, -0.06, -0.03, -0.51, -1.16, -4.18, -8.06],
            [-7.06, -5.12, -2.79, -1.37, -0.65, -0.09, -0.03, -0.45, -1.21, -4.16, -8.42],
            [-7.17, -5.71, -3.67, -2.17, -1.44, -1.03, -1.07, -1.65, -3.41, -7.17, -11.55],
            [-7.23, -6.16, -4.88, -3.68, -3., -2.83, -3.31, -4.91, -7.35, -10.69, -13.04],
            [-6.16, -5.8, -5.57, -5.14, -5.11, -5.38, -6.24, -8.08, -10.29, -14.48, -16.14],
            [-5.85, -5.73, -5.81, -5.8, -5.95, -6.2, -7., -8.46, -10.1, -12.77, -13.89]])

    if include_wall:
        ods['wall.description_2d.0.limiter.unit.0.outline.r'] = wall_r_small
        ods['wall.description_2d.0.limiter.unit.0.outline.z'] = wall_z_small

    return ods


@add_to_ODS
def profiles(ods, time_index=0, nx=11, add_junk_ion=False, include_pressure=True):
    """
    Add made up sample profiles to an ODS. Although made up, the profiles satisfy quasi-neutrality and should very
    roughly resemble something similar to a real plasma a little bit.

    :param ods: ODS instance

    :param time_index: int

    :param nx: int
        Number of points in test profiles

    :param add_junk_ion: bool
        Flag for adding a junk ion for testing how well functions tolerate problems. This will be missing labels, etc.

    :param include_pressure: bool
        Include pressure profiles when temperature and density are added

    :return: ODS instance with profiles added.
        Since the original is modified, it is not necessary to catch the return, but it may be convenient to do so in
        some contexts. If you do not want the original to be modified, deepcopy it first.
    """

    ee = constants.e

    prof1d = ods['core_profiles.profiles_1d'][time_index]
    x = prof1d['grid.rho_tor_norm'] = numpy.linspace(0, 1, nx)
    prof1d['electrons.density'] = 1e19 * (6.5 - 1.9*x - 4.5*x**7)  # m^-3
    prof1d['electrons.density_thermal'] = prof1d['electrons.density']
    prof1d['electrons.temperature'] = 1000 * (4 - 3*x - 0.9*x**9)  # eV
    if include_pressure:
        prof1d['electrons.pressure'] = prof1d['electrons.density'] * prof1d['electrons.temperature'] * ee
        prof1d['electrons.pressure_thermal'] = prof1d['electrons.density_thermal'] * prof1d['electrons.temperature'] * ee

    prof1d['ion.0.label'] = 'D+'
    prof1d['ion.0.element.0.z_n'] = 1.0
    prof1d['ion.0.element.0.atoms_n'] = 1.0
    prof1d['ion.0.element.0.a'] = 2.
    prof1d['ion.0.density'] = prof1d['electrons.density'] * 0.6 / prof1d['ion.0.element.0.z_n']
    prof1d['ion.0.density_fast'] = prof1d['ion.0.density'].max() * 0.32 * numpy.exp(-(x**2)/0.3**2/2.)
    prof1d['ion.0.density_thermal'] = prof1d['ion.0.density'] - prof1d['ion.0.density_fast']
    prof1d['ion.0.temperature'] = prof1d['electrons.temperature'] * 1.1
    if include_pressure:
        prof1d['ion.0.pressure'] = prof1d['ion.0.temperature'] * prof1d['ion.0.density'] * ee
        prof1d['ion.0.pressure_thermal'] = prof1d['ion.0.temperature'] * prof1d['ion.0.density_thermal'] * ee

    prof1d['ion.1.label'] = 'C+6'
    prof1d['ion.1.element.0.z_n'] = 6.0
    prof1d['ion.1.element.0.atoms_n'] = 1.0
    prof1d['ion.1.element.0.a'] = 12.
    prof1d['ion.1.density'] = (prof1d['electrons.density'] - prof1d['ion.0.density']) / prof1d['ion.1.element.0.z_n']
    prof1d['ion.1.density_thermal'] = prof1d['ion.1.density']
    prof1d['ion.1.temperature'] = prof1d['ion.0.temperature']*0.98
    if include_pressure:
        prof1d['ion.1.pressure'] = prof1d['ion.1.temperature'] * prof1d['ion.1.density'] * ee
        prof1d['ion.1.pressure_thermal'] = prof1d['ion.1.temperature'] * prof1d['ion.1.density_thermal'] * ee

    if add_junk_ion:
        junki = prof1d['ion.2']
        junki['density'] = x*0
        junki['temperature'] = x*0

        junkn = prof1d['neutral.0']
        if include_pressure:
            junki['pressure'] = x*0
            junkn['pressure'] = x*0

    return ods


@add_to_ODS
def pf_active(ods, nc_weird=0, nc_undefined=0):
    """
    Adds some FAKE active PF coil locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real.

    :param ods: ODS instance

    :param nc_weird: int
        Number of coils with badly defined geometry to include for testing plot overlay robustness

    :param nc_undefined: int
        Number of coils with undefined geometry_type (But valid r, z outlines) to include for testing plot overlay
        robustness.

    :return: ODS instance with FAKE PF ACTIVE HARDWARE INFORMATION added.
    """

    nc_reg = 4
    nc = nc_reg+nc_weird+nc_undefined
    fc_dat = [
        #  R        Z       dR      dZ    tilt1  tilt2
        [.8608,  .16830,  .0508,  .32106,  0.0,  0.0],
        [1.0041,  1.5169,  .13920,  .11940,  45.0,  0.0],
        [2.6124,  0.4376,  0.17320,  0.1946,  0.0,  92.40],
        [2.3834, -1.1171, 0.1880, 0.16920, 0.0, -108.06],
    ]

    rect_code = geo_type_lookup('rectangle', 'pf_active', ods.imas_version, reverse=True)
    outline_code = geo_type_lookup('outline', 'pf_active', ods.imas_version, reverse=True)

    for i in range(nc_reg):
        if (fc_dat[i][4] == 0) and (fc_dat[i][5] == 0):
            rect = ods['pf_active.coil'][i]['element.0.geometry.rectangle']
            rect['r'] = fc_dat[i][0]
            rect['z'] = fc_dat[i][1]
            rect['width'] = fc_dat[i][2]  # Or width in R
            rect['height'] = fc_dat[i][3]  # Or height in Z
            ods['pf_active.coil'][i]['element.0.geometry.geometry_type'] = rect_code
        else:
            outline = ods['pf_active.coil'][i]['element.0.geometry.outline']
            fdat = fc_dat[i]
            fdat[4] = -fc_dat[i][4] * numpy.pi / 180
            fdat[5] = -(fc_dat[i][5] * numpy.pi / 180 if fc_dat[i][5] != 0 else numpy.pi / 2)
            outline['r'] = [
                fdat[0] - fdat[2] / 2. - fdat[3] / 2. * numpy.tan((numpy.pi/2. + fdat[5])),
                fdat[0] - fdat[2] / 2. + fdat[3] / 2. * numpy.tan((numpy.pi/2. + fdat[5])),
                fdat[0] + fdat[2] / 2. + fdat[3] / 2. * numpy.tan((numpy.pi/2. + fdat[5])),
                fdat[0] + fdat[2] / 2. - fdat[3] / 2. * numpy.tan((numpy.pi/2. + fdat[5])),
             ]
            outline['z'] = [
                fdat[1] - fdat[3] / 2. - fdat[2] / 2. * numpy.tan(-fdat[4]),
                fdat[1] + fdat[3] / 2. - fdat[2] / 2. * numpy.tan(-fdat[4]),
                fdat[1] + fdat[3] / 2. + fdat[2] / 2. * numpy.tan(-fdat[4]),
                fdat[1] - fdat[3] / 2. + fdat[2] / 2. * numpy.tan(-fdat[4]),
            ]
            ods['pf_active.coil'][i]['element.0.geometry.geometry_type'] = outline_code

    for i in range(nc_reg, nc_reg+nc_weird):
        # This isn't a real geometry_type, so it should trigger the contingency
        ods['pf_active.coil'][i]['element.0.geometry.geometry_type'] = 99
    for i in range(nc_reg+nc_weird, nc):
        # This one doesn't have geometry_type defined, so the plot overlay will have trouble looking up which type it is
        outline = ods['pf_active.coil'][i]['element.0.geometry.outline']
        outline['r'] = [1.5, 1.6, 1.7, 1.5]
        outline['z'] = [0.1, 0.3, -0.1, 0]

    return ods


@add_to_ODS
def magnetics(ods):
    """
    Adds some FAKE magnetic probe locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real.

    :param ods: ODS instance

    :return: ODS instance with FAKE MAGNETICS HARDWARE INFORMATION added.
    """

    nbp = 12
    nfl = 7

    r0 = 1.5
    z0 = 0.0
    abp = 0.8
    afl = 1.0

    angle_bp = numpy.linspace(0, 2*numpy.pi, nbp+1)[:-1]
    rp = r0 + abp * numpy.cos(angle_bp)
    zp = z0 + abp * numpy.sin(angle_bp)

    angle_fl = numpy.linspace(0, 2*numpy.pi, nfl + 1)[:-1]
    rf = r0 + afl * numpy.cos(angle_fl)
    zf = z0 + afl * numpy.sin(angle_fl)

    for i in range(nbp):
        ods['magnetics.bpol_probe'][i]['identifier'] = 'FAKE bpol probe {}'.format(i)
        ods['magnetics.bpol_probe'][i]['position.r'] = rp[i]
        ods['magnetics.bpol_probe'][i]['position.z'] = zp[i]
        ods['magnetics.bpol_probe'][i]['position.phi'] = 6.5

    for i in range(nfl):
        ods['magnetics.flux_loop'][i]['identifier'] = 'FAKE flux loop {}'.format(i)
        ods['magnetics.flux_loop'][i]['position.0.r'] = rf[i]
        ods['magnetics.flux_loop'][i]['position.0.z'] = zf[i]

    return ods


@add_to_ODS
def thomson_scattering(ods, nc=10):
    """
    Adds some FAKE Thomson scattering channel locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real.

    :param ods: ODS instance

    :param nc: Number of channels to add.

    :return: ODS instance with FAKE THOMSON HARDWARE INFORMATION added.
    """

    r = numpy.linspace(1.935, 1.945, nc)
    z = numpy.linspace(-0.7, 0.2, nc)
    for i in range(nc):
        ch = ods['thomson_scattering.channel'][i]
        ch['identifier'] = 'F_TS_{:02d}'.format(i)  # F for fake
        ch['name'] = 'Fake Thomson channel for testing {}'.format(i)
        ch['position.phi'] = 6.5  # This angle in rad should look bad to someone who doesn't notice the Fake labels
        ch['position.r'] = r[i]
        ch['position.z'] = z[i]

    return ods


@add_to_ODS
def charge_exchange(ods, nc=10):
    """
    Adds some FAKE CER channel locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real. This function can overwrite existing data if you're not careful.
    The original is modified, so deepcopy first if you want different ODSs.

    :param ods: ODS instance

    :param nc: Number of channels to add.

    :return: ODS instance with FAKE CER HARDWARE INFORMATION added.
    """

    r = numpy.linspace(1.4, 2.2, nc)
    z = numpy.linspace(0.05, -0.07, nc)
    for i in range(nc):
        ch = ods['charge_exchange.channel'][i]
        ch['identifier'] = 'FAKE_CER_{:02d}'.format(i)  # F for fake
        ch['name'] = 'Fake CER channel for testing {}'.format(i)
        for x in ['r', 'z', 'phi']:
            ch['position'][x]['time'] = numpy.array([0])
        ch['position.phi.data'] = numpy.array([6.5])
        ch['position.r.data'] = numpy.array([r[i]])
        ch['position.z.data'] = numpy.array([z[i]])

    return ods


@add_to_ODS
def interferometer(ods):
    """
    Adds some FAKE interferometer locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real.

    :param ods: ODS instance

    :return: ODS instance with FAKE INTERFEROMETER HARDWARE INFORMATION added.
    """
    ods['interferometer.channel.0.identifier'] = 'FAKE horz. interf.'
    r0 = ods['interferometer.channel.0.line_of_sight']
    r0['first_point.phi'] = r0['second_point.phi'] = 225 * (-numpy.pi / 180)
    r0['first_point.r'], r0['second_point.r'] = 3.0, 0.8
    r0['first_point.z'] = r0['second_point.z'] = 0.0

    i = 0
    ods['interferometer.channel'][i + 1]['identifier'] = 'FAKE vert. interf.'
    los = ods['interferometer.channel'][i + 1]['line_of_sight']
    los['first_point.phi'] = los['second_point.phi'] = 240 * (-numpy.pi / 180)
    los['first_point.r'] = los['second_point.r'] = 1.48
    los['first_point.z'], los['second_point.z'] = -1.8, 1.8

    for j in range(len(ods['interferometer.channel'])):
        ch = ods['interferometer.channel'][j]
        ch['line_of_sight.third_point'] = copy.deepcopy(ch['line_of_sight.first_point'])

    return ods


@add_to_ODS
def bolometer(ods, nc=10):
    """
    Adds some FAKE bolometer chord locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real.

    :param ods: ODS instance

    :param nc: 10  # Number of fake channels to make up for testing

    :return: ODS instance with FAKE BOLOMETER HARDWARE INFORMATION added.
    """

    angles = numpy.pi + numpy.linspace(-numpy.pi/4.0, numpy.pi/4.0, nc)

    # FAKE origin for the FAKE bolometer fan
    r0 = 2.5
    z0 = 0.05

    for i in range(nc):
        ch = ods['bolometer.channel'][i]['line_of_sight']
        ch['first_point.r'] = r0
        ch['first_point.z'] = z0 + 0.001 * i
        ch['second_point.r'] = ch['first_point.r'] + numpy.cos(angles[i])
        ch['second_point.z'] = ch['first_point.z'] + numpy.sin(angles[i])
        ods['bolometer.channel'][i]['identifier'] = 'fake bolo {}'.format(i)

    ods['bolometer.channel'][nc-1]['identifier'] = 'bolo fan 2 fake'  # This tests separate colors per fan in overlay

    return ods


@add_to_ODS
def gas_injection(ods):
    """
    Adds some FAKE gas injection locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real. This function can overwrite existing data if you're not careful.
    The original is modified, so deepcopy first if you want different ODSs.

    :param ods: ODS instance

    :return: ODS instance with FAKE GAS INJECTION HARDWARE INFORMATION added.
    """

    ods['gas_injection.pipe.0.name'] = 'FAKE_GAS_A'
    ods['gas_injection.pipe.0.exit_position.r'] = 2.25
    ods['gas_injection.pipe.0.exit_position.z'] = 0.0
    ods['gas_injection.pipe.0.exit_position.phi'] = 6.5
    ods['gas_injection.pipe.0.valve.0.identifier'] = 'FAKE_GAS_VALVE_A'

    ods['gas_injection.pipe.1.name'] = 'FAKE_GAS_B'
    ods['gas_injection.pipe.1.exit_position.r'] = 1.65
    ods['gas_injection.pipe.1.exit_position.z'] = 1.1
    ods['gas_injection.pipe.1.exit_position.phi'] = 6.5
    ods['gas_injection.pipe.1.valve.0.identifier'] = 'FAKE_GAS_VALVE_B'

    ods['gas_injection.pipe.2.name'] = 'FAKE_GAS_C'
    ods['gas_injection.pipe.2.exit_position.r'] = 2.1
    ods['gas_injection.pipe.2.exit_position.z'] = -0.6
    ods['gas_injection.pipe.2.valve.0.identifier'] = 'FAKE_GAS_VALVE_C'
    # This one deliberately doesn't have a phi angle defined, for testing purposes.

    return ods

