'''sample ODS methods and utilities

-------
'''

from .omas_utils import *
from .omas_core import ODS
from .omas_physics import constants
from .omas_plot import geo_type_lookup

__all__ = []
__ods__ = []


def add_to_ODS(f):
    """
    anything wrapped here will be available as a ODS method with name 'sample_'+f.__name__
    """
    __ods__.append(f.__name__)
    return f


def ods_sample(ntimes=1):
    """
    Returns an ODS populated with all of the samples

    :param ntimes: number of time slices to generate

    :return: sample ods
    """
    return ODS().sample(ntimes=ntimes)


@add_to_ODS
def dataset_description(ods):
    ods['dataset_description.data_entry.machine'] = 'test'
    ods['dataset_description.data_entry.pulse'] = 100
    return ods


@add_to_ODS
def equilibrium(
    ods, time_index=0, include_profiles=True, include_phi=True, include_psi=True, include_wall=True, include_q=True, include_xpoint=False
):
    """
    Add sample equilibrium data
    This method operates in in-place.

    :param ods: ODS instance

    :param time_index: int
        Under which time index should fake equilibrium data be loaded?

    :param include_profiles: bool
        Include 1D profiles of pressure, q, p', FF'
        They are in the sample set, so not including them means deleting them.

    :param include_phi: bool
        Include 1D and 2D profiles of phi (toroidal flux, for calculating rho)
        This is in the sample set, so not including it means deleting it.

    :param include_psi: bool
        Include 1D and 2D profiles of psi (poloidal flux)
        This is in the sample set, so not including it means deleting it.

    :param include_wall: bool
        Include the first wall
        This is in the sample set, so not including it means deleting it.

    :param include_q: bool
        Include safety factor
        This is in the sample set, so not including it means deleting it.

    :param include_xpoint: bool
        Include X-point R-Z coordinates
        This is not in the sample set, so including it means making it up

    :return: ODS instance with equilibrium data added
    """
    from omas import load_omas_json

    eq = load_omas_json(imas_json_dir + '/../samples/sample_equilibrium_ods.json', consistency_check=False)

    phi = eq['equilibrium.time_slice.0.profiles_1d.phi']
    psi = eq['equilibrium.time_slice.0.profiles_1d.psi']
    q = eq['equilibrium.time_slice.0.profiles_1d.q']

    if not include_profiles:
        del eq['equilibrium.time_slice.0.profiles_1d']

    if not include_phi:
        if 'profiles_1d' in eq['equilibrium.time_slice.0']:
            del eq['equilibrium.time_slice.0.profiles_1d.phi']
        del eq['equilibrium.time_slice.0.profiles_2d.0.phi']
    else:
        eq['equilibrium.time_slice.0.profiles_1d.phi'] = phi

    if not include_psi:
        if 'profiles_1d' in eq['equilibrium.time_slice.0'] and 'psi' in eq['equilibrium.time_slice.0.profiles_1d']:
            del eq['equilibrium.time_slice.0.profiles_1d.psi']
        del eq['equilibrium.time_slice.0.profiles_2d.0.psi']
    else:
        eq['equilibrium.time_slice.0.profiles_1d.psi'] = psi

    if not include_q:
        if 'profiles_1d' in eq['equilibrium.time_slice.0'] and 'q' in eq['equilibrium.time_slice.0.profiles_1d']:
            del eq['equilibrium.time_slice.0.profiles_1d.q']
    else:
        eq['equilibrium.time_slice.0.profiles_1d.q'] = q

    if not include_wall:
        del eq['wall']

    if include_xpoint:
        eq['equilibrium.time_slice'][0]['boundary.x_point.0.r'] = 1.304
        eq['equilibrium.time_slice'][0]['boundary.x_point.0.z'] = -1.222

    ods['equilibrium.time_slice'][time_index].update(eq['equilibrium.time_slice.0'])
    ods['equilibrium.time_slice'][time_index]['time'] = float(time_index)
    ods['equilibrium.vacuum_toroidal_field.r0'] = eq['equilibrium.vacuum_toroidal_field.r0']
    ods.set_time_array('equilibrium.vacuum_toroidal_field.b0', time_index, eq['equilibrium.vacuum_toroidal_field.b0'][0])
    ods.set_time_array('equilibrium.time', time_index, float(time_index))

    return ods


@add_to_ODS
def core_profiles(ods, time_index=0, add_junk_ion=False, include_pressure=True):
    """
    Add sample core_profiles data
    This method operates in in-place.

    :param ods: ODS instance

    :param time_index: int

    :param add_junk_ion: bool
        Flag for adding a junk ion for testing how well functions tolerate problems. This will be missing labels, etc.

    :param include_pressure: bool
        Include pressure profiles when temperature and density are added

    :return: ODS instance with profiles added
    """
    from omas import load_omas_json

    pr = load_omas_json(imas_json_dir + '/../samples/sample_core_profiles_ods.json', consistency_check=False)

    ods['core_profiles.profiles_1d'][time_index].update(pr['core_profiles.profiles_1d.0'])
    ods['core_profiles.vacuum_toroidal_field.r0'] = pr['core_profiles.vacuum_toroidal_field.r0']
    ods.set_time_array('core_profiles.vacuum_toroidal_field.b0', time_index, pr['core_profiles.vacuum_toroidal_field.b0'][0])

    if add_junk_ion:
        ions = ods['core_profiles.profiles_1d'][time_index]['ion']
        ions[len(ions)] = copy.deepcopy(ions[len(ions) - 1])
        for item in ions[len(ions) - 1].flat():
            ions[len(ions) - 1][item] *= 0

    if not include_pressure:
        for item in ods.physics_core_profiles_pressures(update=False).flat().keys():
            if p2l(item)[0] == 'core_profiles' and p2l(item)[-1].startswith('pres') and item in ods:
                del ods[item]

    ods['core_profiles.profiles_1d'][time_index]['time'] = float(time_index)
    ods.set_time_array('core_profiles.time', time_index, float(time_index))

    return ods


@add_to_ODS
def ic_antennas(ods):
    """
    Add sample ic_antennas data
    This method operates in in-place.

    :param ods: ODS instance

    :param time_index: int

    :return: ODS instance with profiles added
    """
    from omas import load_omas_json

    ods.update(load_omas_json(imas_json_dir + '/../samples/sample_ic_antennas_ods.json', consistency_check=False))

    return ods


@add_to_ODS
def core_sources(ods, time_index=0):
    """
    Add sample core_profiles data
    This method operates in in-place.

    :param ods: ODS instance

    :param time_index: int

    :return: ODS instance with sources added
    """
    from omas import load_omas_json

    pr = load_omas_json(imas_json_dir + '/../samples/sample_core_sources_ods.json', consistency_check=False)['core_sources']

    if 'core_sources' not in ods:
        ods['core_sources'].update(pr)
    else:
        for item in pr:
            if item not in ods['core_sources']:
                ods['core_sources'][item] = pr[item]
        sources = pr['source']
        for source in sources:
            ods['core_sources.source'][source]['identifier'].update(sources[source]['identifier'])
            ods['core_sources.source'][source]['profiles_1d'][time_index].update(sources[source]['profiles_1d.0'])
    ods.set_time_array('core_sources.time', time_index, float(time_index))
    return ods


@add_to_ODS
def core_transport(ods, time_index=0):
    """
    Add sample core_profiles data
    This method operates in in-place

    :param ods: ODS instance

    :param time_index: int

    :return: ODS instance with sources added
    """
    from omas import load_omas_json

    pr = load_omas_json(imas_json_dir + '/../samples/sample_core_transport_ods.json', consistency_check=False)['core_transport']

    if 'core_transport' not in ods:
        ods['core_transport'].update(pr)
    else:
        for item in pr:
            if item not in ods['core_transport']:
                ods['core_transport'][item] = pr[item]
        models = pr['model']
        for model in models:
            ods['core_transport.model'][model]['identifier'].update(models[model]['identifier'])
            ods['core_transport.model'][model]['profiles_1d'][time_index].update(models[model]['profiles_1d.0'])
    ods.set_time_array('core_transport.time', time_index, float(time_index))
    return ods


@add_to_ODS
def summary(ods):
    """
    Add sample core_profiles data
    This method operates in in-place

    :param ods: ODS instance

    :param time_index: int

    :return: ODS instance with sources added
    """
    from omas import load_omas_json

    pr = load_omas_json(imas_json_dir + '/../samples/sample_summary_ods.json', consistency_check=False)['summary']
    ods['summary'].update(pr)
    return ods


@add_to_ODS
def pf_active(ods, nc_weird=0, nc_undefined=0):
    """
    Adds fake active PF coil locations
    This method operates in in-place

    :param ods: ODS instance

    :param nc_weird: int
        Number of coils with badly defined geometry to include for testing plot overlay robustness

    :param nc_undefined: int
        Number of coils with undefined geometry_type (But valid r, z outlines)
        to include for testing plot overlay robustness.

    :return: ODS instance with PF active hardware information added
    """

    nc_reg = 4
    nc = nc_reg + nc_weird + nc_undefined
    fc_dat = [
        #  R        Z       dR      dZ    tilt1  tilt2
        [0.8608, 0.16830, 0.0508, 0.32106, 0.0, 0.0],
        [1.0041, 1.5169, 0.13920, 0.11940, 45.0, 0.0],
        [2.6124, 0.4376, 0.17320, 0.1946, 0.0, 92.40],
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
            fdat[4] = -fc_dat[i][4] * numpy.pi / 180.0
            fdat[5] = -(fc_dat[i][5] * numpy.pi / 180.0 if fc_dat[i][5] != 0 else numpy.pi / 2.0)
            outline['r'] = [
                fdat[0] - fdat[2] / 2.0 - fdat[3] / 2.0 * numpy.tan((numpy.pi / 2.0 + fdat[5])),
                fdat[0] - fdat[2] / 2.0 + fdat[3] / 2.0 * numpy.tan((numpy.pi / 2.0 + fdat[5])),
                fdat[0] + fdat[2] / 2.0 + fdat[3] / 2.0 * numpy.tan((numpy.pi / 2.0 + fdat[5])),
                fdat[0] + fdat[2] / 2.0 - fdat[3] / 2.0 * numpy.tan((numpy.pi / 2.0 + fdat[5])),
            ]
            outline['z'] = [
                fdat[1] - fdat[3] / 2.0 - fdat[2] / 2.0 * numpy.tan(-fdat[4]),
                fdat[1] + fdat[3] / 2.0 - fdat[2] / 2.0 * numpy.tan(-fdat[4]),
                fdat[1] + fdat[3] / 2.0 + fdat[2] / 2.0 * numpy.tan(-fdat[4]),
                fdat[1] - fdat[3] / 2.0 + fdat[2] / 2.0 * numpy.tan(-fdat[4]),
            ]
            ods['pf_active.coil'][i]['element.0.geometry.geometry_type'] = outline_code

    for i in range(nc_reg, nc_reg + nc_weird):
        # This isn't a real geometry_type, so it should trigger the contingency
        ods['pf_active.coil'][i]['element.0.geometry.geometry_type'] = 99
    for i in range(nc_reg + nc_weird, nc):
        # This one doesn't have geometry_type defined, so the plot overlay will have trouble looking up which type it is
        outline = ods['pf_active.coil'][i]['element.0.geometry.outline']
        outline['r'] = [1.5, 1.6, 1.7, 1.5]
        outline['z'] = [0.1, 0.3, -0.1, 0]

    for i in range(nc):
        # Give the dummy coils identifiers so the plot overlay can try to label them; otherwise the labels aren't tested
        ods['pf_active.coil'][i]['element.0.identifier'] = f'samp{i}'

    # Generate some data that are not time homogeneous
    for i in range(nc_reg):
        n = (1 + i) * 10
        ods['pf_active.coil'][i]['current.data'] = numpy.linspace(0, 1, n)
        ods['pf_active.coil'][i]['current.time'] = numpy.linspace(0, 1, n)

    return ods


@add_to_ODS
def magnetics(ods):
    """
    Adds fake magnetic probe locations
    This method operates in place

    :param ods: ODS instance

    :return: ODS instance with fake magnetics hardware information added
    """

    from .machine_mappings.d3d import magnetics_hardware

    magnetics_hardware(ods)
    ods['magnetics.time'] = [0.0]
    return ods


@add_to_ODS
def thomson_scattering(ods, nc=10):
    """
    Adds fake Thomson scattering channel locations
    This method operates in place

    :param ods: ODS instance

    :param nc: Number of channels to add.

    :return: ODS instance with fake Thomson hardware information added
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

    ods['thomson_scattering.time'] = [0]

    return ods


@add_to_ODS
def charge_exchange(ods, nc=10):
    """
    Adds fake CER channel locations
    This method operates in-place

    :param ods: ODS instance

    :param nc: Number of channels to add

    :return: ODS instance with fake CER hardware information added
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
    ods['charge_exchange.time'] = [0.0]
    return ods


@add_to_ODS
def interferometer(ods):
    """
    Adds fake interferometer locations
    This method operates in place

    :param ods: ODS instance

    :return: ODS instance with fake interferometer hardware information addedd
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

    ods['interferometer.time'] = [0]

    return ods


@add_to_ODS
def bolometer(ods, nc=10):
    """
    Adds fake bolometer chord locations
    This method operates in place

    :param ods: ODS instance

    :param nc: 10  # Number of fake channels to make up for testing

    :return: ODS instance with fake bolometer hardware information added
    """

    angles = numpy.pi + numpy.linspace(-numpy.pi / 4.0, numpy.pi / 4.0, nc)

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

    ods['bolometer.channel'][nc - 1]['identifier'] = 'bolo fan 2 fake'  # This tests separate colors per fan in overlay

    ods['bolometer.time'] = [0]

    return ods


@add_to_ODS
def gas_injection(ods):
    """
    Adds fake gas injection locations
    This method operates in place

    :param ods: ODS instance

    :return: ODS instance with fake gas injection hardware information added
    """

    ods['gas_injection.time'] = [0]

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
    ods['gas_injection.pipe.1.second_point.r'] = 1.63
    ods['gas_injection.pipe.1.second_point.z'] = 1.08
    ods['gas_injection.pipe.1.second_point.phi'] = 6.5

    ods['gas_injection.pipe.2.name'] = 'FAKE_GAS_C'
    ods['gas_injection.pipe.2.exit_position.r'] = 1.65
    ods['gas_injection.pipe.2.exit_position.z'] = 1.1
    ods['gas_injection.pipe.2.exit_position.phi'] = 6.5
    ods['gas_injection.pipe.2.valve.0.identifier'] = 'FAKE_GAS_VALVE_C'
    ods['gas_injection.pipe.2.second_point.phi'] = 6.5
    # This one is at the same R,Z as FAKE_GAS_B, but it doesn't have a second point defined; this supports testing.

    ods['gas_injection.pipe.3.name'] = 'FAKE_GAS_D'
    ods['gas_injection.pipe.3.exit_position.r'] = 2.1
    ods['gas_injection.pipe.3.exit_position.z'] = -0.6
    ods['gas_injection.pipe.3.valve.0.identifier'] = 'FAKE_GAS_VALVE_D'
    # This one deliberately doesn't have a phi angle defined, for testing purposes.

    return ods


@add_to_ODS
def langmuir_probes(ods):
    """
    Adds fake Langmuir probe locations
    This method operates in place

    :param ods: ODS instance

    :return: ODS instance with fake Langmuir probe hardware information added
    """

    ods['langmuir_probes.time'] = numpy.array([0])

    ods['langmuir_probes.embedded.0.identifier'] = 0
    ods['langmuir_probes.embedded.0.name'] = 'p1'
    ods['langmuir_probes.embedded.0.position.r'] = 0.9
    ods['langmuir_probes.embedded.0.position.z'] = 0.1
    ods['langmuir_probes.embedded.0.position.phi'] = 0

    ods['langmuir_probes.embedded.1.identifier'] = -1
    ods['langmuir_probes.embedded.1.name'] = 'p23'
    ods['langmuir_probes.embedded.1.position.r'] = 0.9
    ods['langmuir_probes.embedded.1.position.z'] = -0.9
    ods['langmuir_probes.embedded.1.position.phi'] = numpy.NaN

    ods['langmuir_probes.embedded.2.identifier'] = -2
    ods['langmuir_probes.embedded.2.name'] = 'blah'
    ods['langmuir_probes.embedded.2.position.r'] = 1.5
    ods['langmuir_probes.embedded.2.position.z'] = -1.25
    ods['langmuir_probes.embedded.2.position.phi'] = numpy.NaN

    ods['langmuir_probes.embedded.3.identifier'] = -3
    ods['langmuir_probes.embedded.3.name'] = 'donkey!'
    ods['langmuir_probes.embedded.3.position.r'] = 1.525
    ods['langmuir_probes.embedded.3.position.z'] = -1.25
    ods['langmuir_probes.embedded.3.position.phi'] = numpy.NaN

    ods['langmuir_probes.embedded.4.identifier'] = -4
    ods['langmuir_probes.embedded.4.name'] = 'zzz'
    ods['langmuir_probes.embedded.4.position.r'] = 1.4
    ods['langmuir_probes.embedded.4.position.z'] = 1.4
    ods['langmuir_probes.embedded.4.position.phi'] = numpy.pi

    ods['langmuir_probes.embedded.5.identifier'] = -5
    ods['langmuir_probes.embedded.5.name'] = "it's just a test"
    ods['langmuir_probes.embedded.5.position.r'] = 2.45
    ods['langmuir_probes.embedded.5.position.z'] = 0.25
    ods['langmuir_probes.embedded.5.position.phi'] = numpy.NaN

    return ods


@add_to_ODS
def wall(ods):
    """
    Adds fake wall data
    This method operates in in-place

    :param ods: ODS instance

    :return: ODS instance with added wall description
    """
    ods['wall.time'] = [0.0]
    ods['wall.description_2d[0].limiter.type.description'] = 'first wall'
    ods['wall.description_2d[0].limiter.type.index'] = 0
    ods['wall.description_2d[0].limiter.type.name'] = 'first_wall'
    ods['wall.description_2d[0].limiter.unit[0].outline.r'] = [
        # fmt: off
        1.0, 1.0, 1.3, 1.4, 1.6, 2.15, 2.35, 2.35, 2.15, 1.800, 1.350, 1.35, 1.10, 1.00, 1.0
        # fmt: on
    ]
    ods['wall.description_2d[0].limiter.unit[0].outline.z'] = [
        # fmt: off
        0.0, 1.4, 1.4, 1.3, 1.1, 1.00, 0.50, -0.5, -1.0, -1.25, -1.25, -1.4, -1.4, -1.3, 0.0
        # fmt: on
    ]

    return ods


@add_to_ODS
def pulse_schedule(ods):
    """
    Adds fake control target data to support testing
    This method operates in in-place

    :param ods: ODS instance

    :return: ODS instance with added pulse schedule
    """

    def add_position_control(ods_):
        """Adds sample data for the position control subset"""
        bdry = ods_['pulse_schedule.position_control.boundary_outline']

        # These data are sampled from DIII-D#161558 at the following times:
        t = numpy.array([0.1, 0.52, 0.99, 1.29, 1.46, 2.01, 3.91, 5.97, 6.6, 6.9])  # s
        bdry[0]['r.reference.data'] = numpy.array([2.31, 2.27, 2.27, 2.27, 2.27, 2.27, 2.27, 2.27, 2.25, 2.25])  # m
        bdry[0]['z.reference.data'] = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        bdry[1]['r.reference.data'] = numpy.array([2.21, 2.17, 2.17, 2.16, 2.16, 2.15, 2.16, 2.15, 2.12, 2.12])
        bdry[1]['z.reference.data'] = numpy.array([0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43])
        bdry[2]['r.reference.data'] = numpy.array([1.9, 1.9, 1.9, 1.88, 1.87, 1.87, 1.87, 1.87, 1.83, 1.83])
        bdry[2]['z.reference.data'] = numpy.array([0.81, 0.81, 0.81, 0.78, 0.76, 0.76, 0.76, 0.76, 0.69, 0.69])
        bdry[3]['r.reference.data'] = numpy.array([1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52])
        bdry[3]['z.reference.data'] = numpy.array([0.84, 0.91, 0.91, 0.87, 0.84, 0.83, 0.83, 0.83, 1.35, 1.35])
        bdry[4]['r.reference.data'] = numpy.array([1.44, 1.39, 1.39, 1.41, 1.43, 1.43, 1.43, 1.43, 1.1, 1.1])
        bdry[4]['z.reference.data'] = numpy.array([0.8, 0.86, 0.86, 0.83, 0.81, 0.81, 0.81, 0.81, 1.21, 1.21])
        bdry[5]['r.reference.data'] = numpy.array([1.32, 1.28, 1.28, 1.3, 1.31, 1.32, 1.32, 1.32, 1.07, 1.07])
        bdry[5]['z.reference.data'] = numpy.array([0.73, 0.77, 0.77, 0.75, 0.74, 0.73, 0.73, 0.73, 0.99, 0.99])
        bdry[6]['r.reference.data'] = numpy.array([0.92, 1.16, 1.16, 1.16, 1.17, 1.17, 1.17, 1.17, 0.92, 0.92])
        bdry[6]['z.reference.data'] = numpy.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        bdry[7]['r.reference.data'] = numpy.array([1.15, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2])
        bdry[7]['z.reference.data'] = numpy.array([0.5, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.5, 0.5])
        bdry[8]['r.reference.data'] = numpy.array([1.08, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.08, 1.08])
        bdry[8]['z.reference.data'] = numpy.array([-0.5, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.5, -0.5])
        bdry[9]['r.reference.data'] = numpy.array([1.23, 1.12, 1.12, 1.14, 1.14, 1.14, 1.14, 1.14, 1.23, 1.23])
        bdry[9]['z.reference.data'] = numpy.array([-0.78, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.79, -0.79])
        bdry[10]['r.reference.data'] = numpy.array([1.94, 1.18, 1.18, 1.21, 1.22, 1.22, 1.22, 1.22, 1.92, 1.92])
        bdry[10]['z.reference.data'] = numpy.array([-0.88, -0.81, -0.81, -0.79, -0.79, -0.79, -0.79, -0.79, -0.85, -0.85])
        bdry[11]['r.reference.data'] = numpy.array([2.23, 1.89, 1.89, 1.89, 1.89, 1.89, 1.89, 1.89, 2.18, 2.18])
        bdry[11]['z.reference.data'] = numpy.array([-0.43, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.43, -0.43])

        for i in range(12):
            bdry[i]['r.reference_type'] = bdry[i]['z.reference_type'] = 1
            bdry[i]['r.reference_name'] = bdry[i]['z.reference_name'] = 'seg{}'.format(i)
            bdry[i]['r.reference.time'] = bdry[i]['z.reference.time'] = t

        strk = ods_['pulse_schedule.position_control.strike_point']
        strk[0]['r.reference.data'] = numpy.array([numpy.NaN, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, numpy.NaN, numpy.NaN])
        strk[0]['z.reference.data'] = numpy.array([numpy.NaN, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, numpy.NaN, numpy.NaN])
        strk[1]['r.reference.data'] = numpy.array([numpy.NaN, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, numpy.NaN, numpy.NaN])
        strk[1]['z.reference.data'] = numpy.array([numpy.NaN, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, numpy.NaN, numpy.NaN])

        for i in range(2):
            strk[i]['r.reference_type'] = strk[i]['z.reference_type'] = 1
            strk[i]['r.reference_name'] = strk[i]['z.reference_name'] = 'strk{}'.format(i)
            strk[i]['r.reference.time'] = strk[i]['z.reference.time'] = t

        xpt = ods_['pulse_schedule.position_control.x_point']
        xpt[0]['r.reference.data'] = numpy.array([1.1, 1.34, 1.34, 1.43, 1.43, 1.43, 1.43, 1.43, 1.13, 1.13])
        xpt[0]['z.reference.data'] = numpy.array([-1.6, -1.21, -1.21, -1.15, -1.15, -1.15, -1.15, -1.15, -1.41, -1.41])
        xpt[1]['r.reference.data'] = numpy.array([numpy.NaN] * len(t))
        xpt[1]['z.reference.data'] = numpy.array([numpy.NaN] * len(t))

        for i in range(2):
            xpt[i]['r.reference_type'] = xpt[i]['z.reference_type'] = 1
            xpt[i]['r.reference_name'] = xpt[i]['z.reference_name'] = 'strk{}'.format(i)
            xpt[i]['r.reference.time'] = xpt[i]['z.reference.time'] = t

        ods_['pulse_schedule.time'] = t

    add_position_control(ods)

    return ods


@add_to_ODS
def ec_launchers(ods, ngyros=2, ntimes=6):
    """
    Adds fake ec launchers data to support testing
    This method operates in in-place

    :param ods: ODS instance

    :param ngyros: number of gyrotrons

    :param ntimes: number of times

    :return: ODS instance with added ec_launchers
    """

    times = numpy.linspace(0.0, 1.0, ntimes)
    ones = numpy.ones(ntimes)

    ods['ec_launchers.ids_properties.homogeneous_time'] = 1
    for gyro in range(ngyros):
        ods['ec_launchers']['time'] = times
        ods['ec_launchers']['launcher'][gyro]['identifier'] = 'GYRO_' + str(gyro)
        ods['ec_launchers']['launcher'][gyro]['frequency']['data'] = ones * 110e9

        ods['ec_launchers']['launcher'][gyro]['launching_position']['phi'] = 0.0 * ones
        ods['ec_launchers']['launcher'][gyro]['launching_position']['r'] = 2.4 * ones
        ods['ec_launchers']['launcher'][gyro]['launching_position']['z'] = 0.68 * ones
        ods['ec_launchers']['launcher'][gyro]['mode']['data'] = -1.0 * ones
        ods['ec_launchers']['launcher'][gyro]['power_launched']['data'] = 0.5e6 * (
            ones - 0.5 * numpy.cos(2 * numpy.pi * times + gyro / ngyros)
        )

        ods['ec_launchers']['launcher'][gyro]['steering_angle_pol']['data'] = 0.61 * ones
        ods['ec_launchers']['launcher'][gyro]['steering_angle_tor']['data'] = 0.0 * ones

    return ods


@add_to_ODS
def nbi(ods, nunits=2, ntimes=6):
    """
    Adds fake nbi data to support testing
    This method operates in in-place

    :param ods: ODS instance

    :param nunits: number of times

    :param ntimes: number of times

    :return: ODS instance with added nbi
    """

    times = numpy.linspace(0.0, 1.0, ntimes)
    ones = numpy.ones(ntimes)

    ods['nbi.time'] = times
    ods['nbi.ids_properties.homogeneous_time'] = 1

    for unit in range(nunits):
        ods['nbi']['unit'][unit]['beam_current_fraction']['data'] = [0.27032773 * ones, 0.14438479 * ones, 0.07613747 * ones]
        ods['nbi']['unit'][unit]['beam_power_fraction']['data'] = [0.36067036 * ones, 0.09631886 * ones, 0.03386079 * ones]
        ods['nbi']['unit'][unit]['beamlets_group'][0]['angle'] = -0
        ods['nbi']['unit'][unit]['beamlets_group'][0]['direction'] = 1

        ods['nbi']['unit'][unit]['beamlets_group'][0]['divergence_component'][0]['horizontal'] = 0.00873
        ods['nbi']['unit'][unit]['beamlets_group'][0]['divergence_component'][0]['particles_fraction'] = 1.0
        ods['nbi']['unit'][unit]['beamlets_group'][0]['divergence_component'][0]['vertical'] = 0.227
        ods['nbi']['unit'][unit]['beamlets_group'][0]['focus']['focal_length_horizontal'] = 1e31
        ods['nbi']['unit'][unit]['beamlets_group'][0]['focus']['focal_length_vertical'] = 10.0
        ods['nbi']['unit'][unit]['beamlets_group'][0]['position']['phi'] = -0.773
        ods['nbi']['unit'][unit]['beamlets_group'][0]['position']['r'] = 8.11
        ods['nbi']['unit'][unit]['beamlets_group'][0]['position']['z'] = 0.0
        ods['nbi']['unit'][unit]['beamlets_group'][0]['tangency_radius'] = 1.146
        ods['nbi']['unit'][unit]['beamlets_group'][0]['width_vertical'] = 0.48
        ods['nbi']['unit'][unit]['beamlets_group'][0]['width_horizontal'] = 0.1

        ods['nbi']['unit'][unit]['identifier'] = 'beam_' + str(unit)
        ods['nbi']['unit'][unit]['energy']['data'] = 80e3 * ones

        ods['nbi']['unit'][unit]['power_launched']['data'] = 2.0e6 * (ones - 0.5 * numpy.cos(2 * numpy.pi * times + unit / nunits))

        ods['nbi']['unit'][unit]['species']['a'] = 2.0
        ods['nbi']['unit'][unit]['species']['z_n'] = 1.0

    return ods
