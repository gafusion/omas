from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS


__all__ = []


def add_to_ODS(f):
    __all__.append(f.__name__)
    return f


def ods_sample():
    '''
    returns an ODS populated with all of the samples

    :return: sample ods
    '''
    ods=ODS()
    for item in __all__:
        printd('Adding %s sample data to ods'%item,topic='sample')
        ods=eval(item)(ods)
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

    assert(ods2['equilibrium']['time_slice'][0]['global_quantities'].location==ods2['equilibrium']['time_slice'][2]['global_quantities'].location)

    ods2['equilibrium.time_slice.1.global_quantities.ip'] = 2.

    # check different ways of addressing data
    for item in [ods2['equilibrium.time_slice']['1.global_quantities'],
                 ods2[['equilibrium', 'time_slice', 1, 'global_quantities']],
                 ods2[('equilibrium', 'time_slice', '1', 'global_quantities')],
                 ods2['equilibrium.time_slice.1.global_quantities'],
                 ods2['equilibrium.time_slice[1].global_quantities']]:
        assert item.location=='equilibrium.time_slice.:.global_quantities'

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
def equilibrium(ods, time_index=0):
    """
    Expands an ODS by adding a (heavily down-sampled) psi map to it with low precision. This function can overwrite
    existing data if you're not careful. The original is modified, so deepcopy first if you want different ODSs.

    :param ods: ODS instance

    :return: ODS instance with equilibrium data added
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

    ods['equilibrium.time_slice'][time_index]['profiles_1d.psi'] = numpy.linspace(0,1,11)
    ods['equilibrium.time_slice'][time_index]['profiles_2d.0.psi'] = psi_small
    ods['equilibrium.time_slice'][time_index]['profiles_2d.0.grid.dim1'] = grid1_small
    ods['equilibrium.time_slice'][time_index]['profiles_2d.0.grid.dim2'] = grid2_small
    ods['equilibrium.time_slice'][time_index]['boundary.outline.r'] = bdry_r_small
    ods['equilibrium.time_slice'][time_index]['boundary.outline.z'] = bdry_z_small

    ods['equilibrium.time_slice'][time_index]['global_quantities.magnetic_axis.r'] = 1.77
    ods['equilibrium.time_slice'][time_index]['global_quantities.magnetic_axis.z'] = 0.05

    ods['wall.description_2d.0.limiter.unit.0.outline.r'] = wall_r_small
    ods['wall.description_2d.0.limiter.unit.0.outline.z'] = wall_z_small
    return ods


@add_to_ODS
def pf_active(ods):
    """
    Adds some FAKE active PF coil locations so that the overlay plot will work in tests. It's fine to test
    with dummy data as long as you know it's not real.

    :param ods: ODS instance

    :return: ODS instance with FAKE PF ACTIVE HARDWARE INFORMATION added.
    """

    nc = 2
    fc_dat = [
        #  R        Z       dR      dZ    tilt1  tilt2
        [.8608,  .16830,  .0508,  .32106,  0.0,  90.0],
        [1.0041,  1.5169,  .13920,  .11940,  45.0,  90.0],
        [2.6124,  0.4376,  0.17320,  0.1946,  0.0,  92.40],
        [2.3834, -1.1171, 0.1880, 0.16920, 0.0, -108.06],
    ]
    for i in range(nc):
        oblique = ods['pf_active.coil'][i]['element.0.geometry.oblique']
        oblique['r'] = fc_dat[i][0]
        oblique['z'] = fc_dat[i][1]
        oblique['length'] = fc_dat[i][2]  # Or width in R
        oblique['thickness'] = fc_dat[i][3]  # Or height in Z
        oblique['alpha'] = fc_dat[i][4] * numpy.pi/180
        oblique['beta'] = fc_dat[i][5] * numpy.pi/180
        ods['pf_active.coil'][i]['identifier'] = 'FAKE PF COIL {}'.format(i)

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
    ods['gas_injection.pipe.2.exit_position.phi'] = 6.5
    ods['gas_injection.pipe.2.valve.0.identifier'] = 'FAKE_GAS_VALVE_C'

    return ods

