import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd
from omas.machine_mappings._common import *
from omas.utilities.machine_mapping_decorator import machine_mapping_function
from omas.utilities.omas_mds import mdsvalue

__all__ = []
__regression_arguments__ = {'__all__': __all__}

# Reference information
east_divertor_corners = {  # Upper, Lower, Outer, Inner
    'uo': (1.70668, 1.16211),
    'ui': (1.37929, 1.04859),
    'lo': (1.76511, -1.17034),
    'li': (1.33132, -1.01072),
}

# Hardware descriptor functions
@machine_mapping_function(__regression_arguments__)
def setup_pf_active_hardware_description_east(ods):
    r"""
    Adds EAST tokamak poloidal field coil hardware geometry to ODS

    :param ods: ODS instance
    """
    # Coil data from iris:/fusion/usc/src/idl/efitview/diagnoses/EAST/coils_east.dat accessed 2019 December 30 by D. Eldon
    east_pf_coil_data = np.array(
        [
            [6.2866000e-1, 2.5132000e-1, 1.5078000e-1, 4.4260000e-1, 0, 0],
            [6.2866000e-1, -2.513200e-1, 1.5078000e-1, 4.4260000e-1, 0, 0],
            [6.2866000e-1, 7.5396000e-1, 1.5078000e-1, 4.4260000e-1, 0, 0],
            [6.2866000e-1, -7.539600e-1, 1.5078000e-1, 4.4260000e-1, 0, 0],
            [6.2866000e-1, 1.2566000e00, 1.5078000e-1, 4.4260000e-1, 0, 0],
            [6.2866000e-1, -1.256600e00, 1.5078000e-1, 4.4260000e-1, 0, 0],
            [1.0721700e00, 1.7537000e00, 2.3694000e-1, 8.8520000e-2, 0, 0],
            [1.0721700e00, -1.753700e00, 2.3694000e-1, 8.8520000e-2, 0, 0],
            [1.1367900e00, 1.9409200e00, 3.6618000e-1, 2.6556000e-1, 0, 0],
            [1.1367900e00, -1.940920e00, 3.6618000e-1, 2.6556000e-1, 0, 0],
            [2.9455800e00, 1.5907300e00, 1.1844000e-1, 2.0340000e-1, 0, 0],
            [2.9455800e00, -1.590730e00, 1.1844000e-1, 2.0340000e-1, 0, 0],
            [3.2698000e00, 9.0419000e-1, 8.8960000e-2, 1.6272000e-1, 0, 0],
            [3.2698000e00, -9.041900e-1, 7.8960000e-2, 1.6272000e-1, 0, 0],
            [2.4500000e00, 6.0000000e-1, 5.0000000e-2, 1.0000000e-1, 0, 0],
            [2.4500000e00, -6.000000e-1, 5.0000000e-2, 1.0000000e-1, 0, 0],
        ]
    )

    pf_coils_to_ods(ods, east_pf_coil_data)

    for i in range(len(east_pf_coil_data[:, 0])):
        # Coil names can be found in figures 2, 3 and 4 of Xiao, et al. FED 2019
        # https://www.sciencedirect.com/science/article/pii/S0920379619304594
        if i < 14:
            coilnum = i + 1
            fcid = 'PF{}'.format(coilnum)
        else:
            coilnum = i - 13
            fcid = 'IC{}'.format(coilnum)
        ods['pf_active.coil'][i]['name'] = ods['pf_active.coil'][i]['identifier'] = fcid
        ods['pf_active.coil'][i]['element.0.identifier'] = fcid


def east_coords_along_wall(s, rlim, zlim, surface):
    """
    Transforms s into R, Z. Useful for finding LP locations

    :param s: numeric
        Distance along the wall from a reference point (m)
    :param rlim: 1D array
        R coordinates along the limiting surface (m)
    :param zlim:
        Z coordinates along the limiting surface (m)
    :param surface: str
        Which surface / reference should be used?
        'uo', 'ui', 'lo', or 'li'
    :return: (R, Z)
        R value(s) corresponding to s value(s) in m
        Z value(s)
        slim (S values corresponding to rlim and zlim)
    """

    # Find corner and pick direction around limiter to go
    r_corner, z_corner = east_divertor_corners[surface]
    wc = np.where((r_corner == rlim) & (z_corner == zlim))[0]
    dr_fwd, dz_fwd = rlim[wc + 1] - rlim[wc], zlim[wc + 1] - zlim[wc]
    dr_rev, dz_rev = rlim[wc - 1] - rlim[wc], zlim[wc - 1] - zlim[wc]
    theta_fwd, theta_rev = np.arctan2(dz_fwd, dr_fwd), np.arctan2(dz_rev, dr_rev)
    theta_vert = np.pi / 2 if z_corner > 0 else -np.pi / 2
    direction = 1 if abs(theta_fwd - theta_vert) < abs(theta_rev - theta_vert) else -1
    # Get limiter path starting at the relevant corner and going toward the most vertical direction
    rlim = np.roll(rlim, -wc - int(direction == -1))[::direction]
    zlim = np.roll(zlim, -wc - int(direction == -1))[::direction]
    # Get distance along limiter starting from the corner
    ds = np.append(0, np.sqrt(np.diff(rlim) ** 2 + np.diff(zlim) ** 2))
    slim = np.cumsum(ds)
    # Make sure the path is set up correctly
    assert rlim[0] == r_corner
    assert zlim[0] == z_corner
    assert abs(np.arctan2(zlim[1] - zlim[0], rlim[1] - rlim[0]) - (theta_fwd if direction == 1 else theta_rev)) < np.pi / 500
    assert slim[0] == 0
    # Interpolate R, Z of limiter vs. distance from corner onto the distances for the probes
    r = np.interp(s, slim, rlim)
    z = np.interp(s, slim, zlim)

    slim2 = np.roll(slim, wc + int(direction == -1))[::direction]

    return r, z, slim2


@machine_mapping_function(__regression_arguments__, pulse=85282)
def setup_langmuir_probes_hardware_description_east(ods, pulse):
    """
    Load EAST Langmuir probe locations into an ODS

    :param ods: ODS instance

    :param pulse: int
        Will try to fill in from ODS machine data if None
    """
    try:
        rlim = ods['wall.description_2d[0].limiter.unit[0].outline.r']
        zlim = ods['wall.description_2d[0].limiter.unit[0].outline.z']
    except (ValueError, KeyError):
        # Don't have rlim and zlim yet
        printd('No limiter in ods. Gathering it from MDSplus to support ', topic='machine')
        lim = mdsvalue('east', 'EFIT_EAST', pulse, TDI=r'\top.results.geqdsk.lim').data()
        rlim = lim[:, 0]
        zlim = lim[:, 1]

    j = 0
    for ul in ['upper', 'lower']:
        for oi in ['outer', 'inner']:
            corner = '{}{}'.format(ul[0], oi[0])
            pointname = r'\{}LPS'.format(corner.upper())
            # These are distance along the wall from a reference point in m for a group of probes
            m = mdsvalue('east', 'ANALYSIS', pulse, TDI=pointname)
            try:
                s = m.data()
            except Exception:
                printd('Failed MDSplus data check for {}; data invalid. Halting.'.format(pointname), topic='machine')
            else:
                # Data appear to be valid; proceed
                printd('Processing data for probe group {}; {}'.format(corner.upper(), pointname), topic='machine')

                r, z, _ = east_coords_along_wall(s, rlim, zlim, corner)

                numbering_starts_at = 1
                for i in range(len(r)):
                    # Probe numbering scheme confirmed with EAST handbook using data access 2018 July 16 by D. Eldon
                    probe_number = i + numbering_starts_at
                    identifier = '{}{:02d}'.format(corner.upper(), probe_number)
                    ods['langmuir_probes.embedded'][j]['position.r'] = r[i]
                    ods['langmuir_probes.embedded'][j]['position.z'] = z[i]
                    ods['langmuir_probes.embedded'][j]['position.phi'] = np.nan  # Didn't find this in MDSplus
                    ods['langmuir_probes.embedded'][j]['identifier'] = identifier
                    ods['langmuir_probes.embedded'][j]['name'] = identifier
                    j += 1


@machine_mapping_function(__regression_arguments__, pulse=85282)
def setup_gas_injection_hardware_description_east(ods, pulse):
    """
    Sets up APPROXIMATE EAST gas injector data for some systems.

    Data sources:
    Figure downloaded from EAST handbook into notes
    """

    i = 0

    def port_angle(port):
        """
        Converts a port letter into a toroidal angle
        EAST has 16 segments. If A is segment 0, P is 15. Assumes port centers are equally spaced, which they appear to
        be, or at least nearly so, based on a drawing from the EAST handbook.
        :return: float
            Angle in radians
        """
        # I am guessing a toroidal angle coordinate system. I could be wrong by an offset and a direction.
        offset = 0  # radians
        direction = 1  # +/- 1
        import string

        return string.ascii_lowercase.find(port.lower()) / 16.0 * 2 * np.pi * direction + offset
        return (ord(port.lower()) - ord('a')) / 16.0 * 2 * np.pi * direction + offset

    # OUPEV2
    # I think it's between probes 8 & 9. I am guessing. This gives R, Z
    # I think it's in port O
    pipe = ods['gas_injection']['pipe'][i]
    phi = port_angle('o')
    pipe['name'] = 'OUPEV2_{:03d}'.format(int(round(phi * 180 / np.pi)))
    pipe['exit_position']['r'] = 1.73  # m
    pipe['exit_position']['z'] = 1.057  # m
    pipe['exit_position']['phi'] = phi
    pipe['valve'][0]['identifier'] = 'OUPEV2'
    pipe['second_point']['phi'] = phi
    pipe['second_point']['r'] = 1.729
    pipe['second_point']['z'] = 1.05675
    i += 1

    # ODPEV2
    # It's in the lower divertor. I'll have to eyeball from a drawing. Also, I am guessing which tip it is.
    # I think it's in port O
    pipe = ods['gas_injection']['pipe'][i]
    phi = port_angle('o')
    pipe['name'] = 'ODPEV2_{:03d}'.format(int(round(phi * 180 / np.pi)))
    pipe['exit_position']['r'] = 1.811  # m
    pipe['exit_position']['z'] = -0.972  # m
    pipe['exit_position']['phi'] = phi
    pipe['valve'][0]['identifier'] = 'ODPEV2'
    pipe['second_point']['phi'] = phi
    pipe['second_point']['r'] = 1.806
    pipe['second_point']['z'] = -0.9715
    i += 1


if __name__ == '__main__':
    test_machine_mapping_functions(__all__, globals(), locals())
