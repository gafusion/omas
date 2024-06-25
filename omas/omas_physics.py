'''physics-based ODS methods and utilities

-------
'''

from scipy.interpolate import RectBivariateSpline
from .omas_utils import *
from .omas_core import ODS

__all__ = []
__ods__ = []


def add_to__ODS__(f):
    """
    anything wrapped here will be available as a ODS method with name 'physics_'+f.__name__
    """
    __ods__.append(f.__name__)
    return f


def add_to__ALL__(f):
    __all__.append(f.__name__)
    return f


def preprocess_ods(*require, require_mode=['warn_through', 'warn_skip', 'raise'][0]):
    """
    Decorator function that:
     * checks that required quantities are there
    """

    def _req(f):
        from functools import wraps

        @wraps(f)
        def wrapper(*args1, **kw1):
            args, kw = args_as_kw(f, args1, kw1)

            # handle missing required quantities
            missing = []
            for k in require:
                if k not in kw['ods']:
                    missing.append(k)
            if len(missing):
                txt = 'could not evaluate %s because of missing %s ODS' % (f.__name__, missing)
                if require_mode == 'warn_through':
                    printe(txt)
                elif require_mode == 'warn_skip':
                    printe(txt)
                    return kw['ods']
                elif require_mode == 'raise':
                    raise RuntimeError(txt)

            # run function
            return f(*args, **kw)

        return wrapper

    return _req


# constants class that mimics scipy.constants
class constants(object):
    e = 1.6021766208e-19


@add_to__ODS__
def consistent_times(ods, attempt_fix=True, raise_errors=True):
    """
    Assign .time and .ids_properties.homogeneous_time info for top-level structures since these are required for writing an IDS to IMAS

    :param attempt_fix: fix dataset_description and wall IDS to have 0 times if none is set

    :param raise_errors: raise errors if could not satisfy IMAS requirements

    :return: `True` if all is good, `False` if requirements are not satisfied, `None` if fixes were applied
    """

    # if called at top level, loop over all data structures
    if not len(ods.location):
        out = {}
        for ds in ods:
            out[ds] = ods.getraw(ds).physics_consistent_times(attempt_fix=attempt_fix, raise_errors=raise_errors)
        if any(k is False for k in out.values()):
            return False
        elif any(k is None for k in out.values()):
            return None
        else:
            return True

    ds = p2l(ods.location)[0]

    extra_info = {}
    time = ods.time(extra_info=extra_info)
    if extra_info['homogeneous_time'] is False:
        ods['ids_properties']['homogeneous_time'] = extra_info['homogeneous_time']
    elif time is not None and len(time):
        ods['time'] = time
        ods['ids_properties']['homogeneous_time'] = extra_info['homogeneous_time']
    elif attempt_fix:
        ods['time'] = [-1.0]
        extra_info['homogeneous_time'] = True
        ods['ids_properties']['homogeneous_time'] = extra_info['homogeneous_time']
        return None
    elif raise_errors:
        raise ValueError(ods.location + '.time cannot be automatically filled! Missing time information in the data structure.')
    else:
        return False
    return True


@add_to__ODS__
def imas_info(ods):
    """
    add ids_properties.version_put... information

    :return: updated ods
    """
    # if called at top level, loop over all data structures
    if not len(ods.location):
        for ds in ods:
            ods.getraw(ds).physics_imas_info()
        return
    else:
        ods['ids_properties.version_put.access_layer'] = 'N/A'
        ods['ids_properties.version_put.access_layer_language'] = 'OMAS'
        ods['ids_properties.version_put.data_dictionary'] = ods.imas_version

    return ods


@add_to__ODS__
@preprocess_ods('equilibrium')
def equilibrium_stored_energy(ods, update=True):
    """
    Calculate MHD stored energy from equilibrium pressure and volume

    :param ods: input ods

    :param update: operate in place

    :return: updated ods
    """
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    for time_index in ods['equilibrium']['time_slice']:
        pressure_equil = ods['equilibrium']['time_slice'][time_index]['profiles_1d']['pressure']
        volume_equil = ods['equilibrium']['time_slice'][time_index]['profiles_1d']['volume']

        ods_n['equilibrium.time_slice'][time_index]['.global_quantities.energy_mhd'] = (
            3.0 / 2.0 * numpy.trapz(pressure_equil, x=volume_equil)
        )  # [J]

    return ods_n


@add_to__ODS__
@preprocess_ods('equilibrium')
def equilibrium_ggd_to_rectangular(ods, time_index=None, resolution=None, method='linear', update=True):
    """
    Convert GGD data to profiles 2D

    :param ods: input ods

    :param time_index: time slices to process

    :param resolution: integer or tuple for rectangular grid resolution

    :param method: one of 'nearest', 'linear', 'cubic', 'extrapolate'

    :param update: operate in place

    :return: updated ods
    """
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    points = ods['equilibrium.grids_ggd[0].grid[0].space[0].objects_per_dimension[0].object[:].geometry']

    if resolution is None:
        resolution = int(numpy.sqrt(len(points[:, 0])))
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    if time_index is None:
        time_index = range(len(ods['equilibrium.time_slice']))
    elif isinstance(time_index, int):
        time_index = [time_index]

    cache = True
    for itime in time_index:
        ods_n[f'equilibrium.time_slice.{itime}.profiles_2d.0.grid_type'].setdefault('index', 1)
        for k in ods_n[f'equilibrium.time_slice.{itime}.profiles_2d']:
            profiles_2d = ods_n[f'equilibrium.time_slice.{itime}.profiles_2d.{k}']
            if 'grid_type.index' in profiles_2d and profiles_2d['grid_type.index'] == 1:
                break
        ggd = ods[f'equilibrium.time_slice.{itime}.ggd.0']
        for what in ggd:
            quantity = ggd[what + '.0.values']
            r, z, interpolated, cache = scatter_to_rectangular(
                points[:, 0], points[:, 1], quantity, resolution[0], resolution[1], method=method, return_cache=cache
            )
            profiles_2d[what] = interpolated.T
        profiles_2d['grid.dim1'] = r
        profiles_2d['grid.dim2'] = z
    return ods_n


def map_flux_coordinate_to_pol_flux(ods, time_index, origin, values):
    """
    Maps from one magnetic coordinate system to psi
    :param ods: input ods

    :param time_index: time slices to process

    :param origin: Specifier for original coordinate system

    :param values: Values to transform to poloidal flux

    :return: Transformed values
    """
    if origin == "rho_pol":
        return (
            values**2
            * (
                ods["equilibrium"]["time_slice"][time_index]["global_quantities"]["psi_sep"]
                - ods["equilibrium"]["time_slice"][time_index]["global_quantities"]["psi_axis"]
            )
            + ods["equilibrium"]["time_slice"][time_index]["global_quantities"]["psi_axis"]
        )
    else:
        raise NotImplementedError(f"Conversion from {origin} not yet implemented.")


def map_pol_flux_to_flux_coordinate(ods, time_index, destination, values):
    import numpy as np

    """
        Maps from one magnetic coordinate system to psi
        :param ods: input ods

        :param time_index: time slices to process

        :param destination: Target coordinate system for output

        :param values: Values to transform to poloidal flux

        :return: Transformed values
    """
    if destination == "rho_pol":
        return np.sqrt(
            (values - ods["equilibrium"]["time_slice"][time_index]["global_quantities"]["psi_axis"])
            / (
                ods["equilibrium"]["time_slice"][time_index]["global_quantities"]["psi_boundary"]
                - ods["equilibrium"]["time_slice"][time_index]["global_quantities"]["psi_axis"]
            )
        )
    else:
        raise NotImplementedError(f"Conversion to {destination} not yet implemented.")


@add_to__ODS__
@preprocess_ods('equilibrium')
def remap_flux_coordinates(ods, time_index, origin, destination, values):
    """
    Maps from one magnetic coordinate system to another. At the moment only supports
    psi <-> rho_pol
    :param ods: input ods

    :param time_index: time slices to process

    :param origin: Specifier for original coordinate system

    :param destination: Target coordinate system for output

    :param values: Values to transform

    :return: Transformed values
    """
    if origin != "psi":
        psi = map_flux_coordinate_to_pol_flux(ods, time_index, origin, values)
    else:
        psi = values
    if destination != "psi":
        return map_pol_flux_to_flux_coordinate(ods, time_index, destination, psi)
    return psi


@add_to__ODS__
@preprocess_ods('equilibrium')
def resolve_equilibrium_profiles_2d_grid_index(ods, time_index, grid_identifier):
    """
    Convenience function to identify which of profiles_2d[:].grid_type.index
    matches the specified grid_identifier

    :param ods: input ods

    :param time_index: time index to search

    :param grid_identifier: grid type to be resolved

    :return: Index of grid the requested grid, not to be confused with
        profiles_2d[:].grid_type.index
    """
    grid_type = {'grid_type.index': grid_identifier}
    try:
        grid_index = search_in_array_structure(
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d'], grid_type, no_matches_raise_error=True
        )[0]
    except IndexError:
        raise ValueError("Requested equilibrium.profiles_2d_grid_type not present")
    return grid_index


@add_to__ODS__
@preprocess_ods('equilibrium')
def derive_equilibrium_profiles_2d_quantity(ods, time_index, grid_index, quantity):
    """
    This function derives values of empty fields in prpfiles_2d from other parameters in the equilibrium ods
    Currently only the magnetic field components are supported

    :param ods: input ods

    :param time_index: time slice to process

    :param grid_index: Index of grid to map

    :param quantity: Member of profiles_2d to be derived

    :return: updated ods
    """
    from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline

    r, z = numpy.meshgrid(
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim1'],
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim2'],
        indexing="ij",
    )
    psi_spl = RectBivariateSpline(
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim1'],
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim2'],
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.psi'],
    )
    cocos = define_cocos(11)
    if quantity == "b_r":
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.b_r'] = (
            psi_spl(r, z, dy=1, grid=False) * cocos['sigma_RpZ'] * cocos['sigma_Bp'] / ((2.0 * numpy.pi) ** cocos['exp_Bp'] * r)
        )
        return ods
    elif quantity == "b_z":
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.b_z'] = (
            -psi_spl(r, z, dx=1, grid=False) * cocos['sigma_RpZ'] * cocos['sigma_Bp'] / ((2.0 * numpy.pi) ** cocos['exp_Bp'] * r)
        )
        return ods
    elif quantity == "b_tor":
        mask = numpy.logical_and(
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.psi']
            < numpy.max(ods[f'equilibrium.time_slice.{time_index}.profiles_1d.psi']),
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.psi']
            > numpy.min(ods[f'equilibrium.time_slice.{time_index}.profiles_1d.psi']),
        )
        f_spl = InterpolatedUnivariateSpline(
            ods[f'equilibrium.time_slice.{time_index}.profiles_1d.psi'], ods[f'equilibrium.time_slice.{time_index}.profiles_1d.f']
        )
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.b_tor'] = numpy.zeros(r.shape)
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.b_tor'][mask] = (
            f_spl(psi_spl(r[mask], z[mask], grid=False)) / r[mask]
        )
        ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.b_tor'][mask == False] = (
            ods[f'equilibrium.time_slice.{time_index}.profiles_1d.f'][-1] / r[mask == False]
        )
        return ods
    raise NotImplementedError(f"Cannot add {quantity}. Not yet implemented.")


def cache_interpolator(cache, time_index, grid_index, quantity, interpolator):
    """
    Utility function for equilibrium_profiles_2d_map. Creates a tree dictionary structure to store interpolators.

    :param cache: cache object to add tree entry to

    :param time_index: time slices to process

    :param grid_index: Index of grid to map

    :param quantity: Member of profiles_2d[:]

    :param interpolator: Interpolator to store

    :return: updated cache

    """
    if cache is None:
        cache = {}
    if time_index not in cache:
        cache[time_index] = {}
    if grid_index not in cache[time_index]:
        cache[time_index][grid_index] = {}
    cache[time_index][grid_index][quantity] = interpolator
    return cache


@add_to__ODS__
@preprocess_ods('equilibrium')
def equilibrium_profiles_2d_map(
    ods, time_index, grid_index, quantity, dim1=None, dim2=None, cache=None, return_cache=False, out_of_bounds_value=numpy.nan
):
    """
    This routines creates interpolators for quantities and stores them in the cache for future use.
    It can also be used to just return the current profile_2d quantity by omitting dim1 and dim2.
    At the moment this routine always extrapolates for data outside the defined grid range.

    :param ods: input ods

    :param time_index: time slices to process

    :param grid_index: Index of grid to map

    :param quantity: Member of profiles_2d[:] to map

    :param dim1: First coordinate of the points to map to

    :param dim2: Second coordinate of the points to map to

    :param cache: Cache to store interpolants in

    :param return_cache: Toggles return of cache

    :return: mapped positions (and cahce if return_cache)
    """
    if quantity not in ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}']:
        ods.physics_derive_equilibrium_profiles_2d_quantity(time_index, grid_index, quantity)
    if dim1 is None or dim2 is None:
        return ods[f'equilibrium.time_slice.{time_index}.profiles_2d.0.{quantity}']
    # Try to use an interpolator from the cache
    if cache is not None:
        try:
            if return_cache:
                return cache[time_index][quantity](dim1, dim2, grid=False), cache
            else:
                return cache[time_index][quantity](dim1, dim2, grid=False)
        except KeyError:
            pass
    if ods[f'equilibrium.time_slice[{time_index}].profiles_2d[{grid_index}].grid_type.index'] == 91:
        interpolator = create_scatter_interpolator(
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.0.grid.dim1'],
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim2'],
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.{quantity}'],
            method='cubic',
            return_cache=False,
        )
    else:
        interpolator = RectBivariateSpline(
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim1'],
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim2'],
            ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.{quantity}'],
        )
    mapped_values = numpy.zeros(dim1.shape)
    mapped_values[:] = numpy.nan
    mask = numpy.logical_and(
        numpy.logical_and(
            dim1 > numpy.min(ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim1']),
            dim1 < numpy.max(ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim1']),
        ),
        numpy.logical_and(
            dim2 > numpy.min(ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim2']),
            dim2 < numpy.max(ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.grid.dim2']),
        ),
    )
    if return_cache:
        cache = cache_interpolator(cache, time_index, grid_index, quantity, interpolator)
        mapped_values[mask] = cache[time_index][grid_index][quantity](dim1[mask], dim2[mask], grid=False)
        return mapped_values, cache
    mapped_values[mask] = interpolator(dim1[mask], dim2[mask], grid=False)
    return mapped_values


def remove_integrator_drift(time, data, time_after_shot):
    # assume that the drift is zero at time[0]
    ind = time > time_after_shot
    return data - (time - time[0]) / (time[ind].mean() - time[0]) * data[ind].mean(0)


@add_to__ODS__
def equilibrium_form_constraints(
    ods,
    times=None,
    default_average=0.02,
    constraints=None,
    averages=None,
    cutoff_hz=None,
    rm_integr_drift_after=None,
    update=True,
    **nuconv_kw,
):
    """
    generate equilibrium constraints from experimental data in ODS

    :param ods: input ODS

    :param times: list of times at which to generate the constraints

    :param default_average: default averaging time

    :param constraints: list of constraints to be formed (if experimental data is available)
                        NOTE: only the constraints marked with `OK` are supported at this time::

                         OK b_field_tor_vacuum_r
                         OK bpol_probe
                         OK diamagnetic_flux
                          * faraday_angle
                         OK flux_loop
                         OK ip
                          * iron_core_segment
                          * mse_polarisation_angle
                          * n_e
                          * n_e_line
                         OK pf_current
                          * pf_passive_current
                          * pressure
                          * q
                          * strike_point
                          * x_point

    :param averages: dictionary with average times for individual constraints
               Smoothed using Gaussian, sigma=averages/4. and the convolution is integrated across +/-4.*sigma.

    :param cutoff_hz: a list of two elements with low and high cutoff frequencies [lowFreq, highFreq]

    :param rm_integr_drift_after: time in ms after which is assumed thet all currents are zero and signal should be equal to zero. Used for removing of the integrators drift

    :param update: operate in place

    :return: updated ods
    """
    from omfit_classes.utils_math import smooth_by_convolution, firFilter

    if averages is None:
        averages = {}

    # identify possible constraints
    possible_constraints = omas_info('equilibrium')['equilibrium.time_slice.0.constraints'].keys()
    if constraints is None:
        constraints = possible_constraints
    else:
        for constraint in constraints:
            if constraint not in possible_constraints:
                raise ValueError(f'Constraint `{constraint}` not recognized: possible options are {possible_constraints}')

    # instantiate new ODS if not operating in place
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    if times is None:
        if 'equilibrium.time' in ods:
            times = ods['equilibrium.time']
        else:
            raise ValueError('Must specify times at which to apply equilibrium constraint')

    times = numpy.atleast_1d(times)
    ods_n['equilibrium.time'] = times

    nuconv_kw.setdefault('window_function', 'boxcar')
    # pf_current
    if 'pressure' in constraints and 'thompson' in ods:
        raise Exception('Not implemented yet!!')

    # pf_current
    if 'pf_current' in constraints and 'pf_active.coil' in ods:
        average = averages.get('pf_active', default_average)
        for channel in ods['pf_active.coil']:
            printd(f'Processing pf_active.coil.{channel}', topic='machine')
            try:
                # get
                label = ods[f'pf_active.coil.{channel}.name']
                turns = ods[f'pf_active.coil.{channel}.element[0].turns_with_sign']
                data = ods[f'pf_active.coil.{channel}.current.data']
                time = ods[f'pf_active.coil.{channel}.current.time']
                if f'pf_active.coil.{channel}.current.data_error_upper' in ods:
                    error = ods[f'pf_active.coil.{channel}.current.data_error_upper']
                else:
                    error = None
                # process
                if rm_integr_drift_after is not None:
                    data = remove_integrator_drift(time, data, rm_integr_drift_after)
                if cutoff_hz is not None:
                    data = firFilter(time, data, cutoff_hz)
                # Don't average for length=2 arrays or smaller
                if len(data) >2:
                    const = smooth_by_convolution(data * turns, time, times, average, **nuconv_kw)
                    if error is not None:
                        const_error = smooth_by_convolution(error * turns, time, times, average, **nuconv_kw)
                else:
                    const = smooth_by_convolution(data * turns, time, times)
                    if error is not None:
                         const_error = smooth_by_convolution(error * turns, time, times)

                # assign
                for time_index in range(len(times)):
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.pf_current.{channel}.measured'] = const[time_index]
                    if error is not None:
                        ods_n[f'equilibrium.time_slice.{time_index}.constraints.pf_current.{channel}.measured_error_upper'] = const_error[
                            time_index
                        ]
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.pf_current.{channel}.source'] = label
            except Exception as _excp:
                raise _excp.__class__(f'Problem with pf_current channel {channel} :' + str(_excp))

    # bpol_probe
    if 'bpol_probe' in constraints and 'magnetics.b_field_pol_probe' in ods:
        average = averages.get('bpol_probe', default_average)
        for channel in ods[f'magnetics.b_field_pol_probe']:
            printd(f'Processing magnetics.b_field_pol_probe.{channel}', topic='machine')
            try:
                # get
                label = ods[f'magnetics.b_field_pol_probe.{channel}.identifier']
                for time_index in range(len(times)):
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.bpol_probe.{channel}.source'] = label
                valid = ods.get(
                    f'magnetics.b_field_pol_probe.{channel}.field.validity',
                    1 - int(f'magnetics.b_field_pol_probe.{channel}.field.data' in ods),
                )
                if valid == 0:  # 0 means that the data is good
                    data = ods[f'magnetics.b_field_pol_probe.{channel}.field.data']
                    time = ods[f'magnetics.b_field_pol_probe.{channel}.field.time']
                    if f'magnetics.b_field_pol_probe.{channel}.field.data_error_upper' in ods:
                        error = ods[f'magnetics.b_field_pol_probe.{channel}.field.data_error_upper']
                    else:
                        error = None
                    # process
                    if rm_integr_drift_after is not None:
                        data = remove_integrator_drift(time, data, rm_integr_drift_after)
                    if cutoff_hz is not None:
                        data = firFilter(time, data, cutoff_hz)
                    const = smooth_by_convolution(data, time, times, average, **nuconv_kw)
                    if error is not None:
                        const_error = smooth_by_convolution(error, time, times, average, **nuconv_kw)
                    # assign
                    for time_index in range(len(times)):
                        ods_n[f'equilibrium.time_slice.{time_index}.constraints.bpol_probe.{channel}.measured'] = const[time_index]
                        if error is not None:
                            ods_n[
                                f'equilibrium.time_slice.{time_index}.constraints.bpol_probe.{channel}.measured_error_upper'
                            ] = const_error[time_index]
                else:
                    for time_index in range(len(times)):
                        ods_n[f'equilibrium.time_slice.{time_index}.constraints.bpol_probe.{channel}.measured'] = numpy.nan
                        if error is not None:
                            ods_n[f'equilibrium.time_slice.{time_index}.constraints.bpol_probe.{channel}.measured_error_upper'] = numpy.nan
            except Exception as _excp:
                raise _excp.__class__(f'Problem with bpol_probe channel {channel}: ' + str(_excp))

    # flux_loop
    if 'flux_loop' in constraints and 'magnetics.flux_loop' in ods:
        average = averages.get('flux_loop', default_average)
        for channel in ods[f'magnetics.flux_loop']:
            printd(f'Processing magnetics.flux_loop.{channel}', topic='machine')
            try:
                # get
                label = ods[f'magnetics.flux_loop.{channel}.identifier']
                for time_index in range(len(times)):
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.flux_loop.{channel}.source'] = label
                valid = ods.get(f'magnetics.flux_loop.{channel}.flux.validity', 1 - int(f'magnetics.flux_loop.{channel}.flux.data' in ods))
                if valid == 0:  # 0 means that the data is good
                    data = ods[f'magnetics.flux_loop.{channel}.flux.data']
                    time = ods[f'magnetics.flux_loop.{channel}.flux.time']
                    if f'magnetics.flux_loop.{channel}.flux.data_error_upper' in ods:
                        error = ods[f'magnetics.flux_loop.{channel}.flux.data_error_upper']
                    else:
                        error = None
                    # process
                    if rm_integr_drift_after is not None:
                        data = remove_integrator_drift(time, data, rm_integr_drift_after)
                    if cutoff_hz is not None:
                        data = firFilter(time, data, cutoff_hz)
                    const = smooth_by_convolution(data, time, times, average, **nuconv_kw)
                    if error is not None:
                        const_error = smooth_by_convolution(error, time, times, average, **nuconv_kw)
                    # assign
                    for time_index in range(len(times)):
                        ods_n[f'equilibrium.time_slice.{time_index}.constraints.flux_loop.{channel}.measured'] = const[time_index]
                        if error is not None:
                            ods_n[
                                f'equilibrium.time_slice.{time_index}.constraints.flux_loop.{channel}.measured_error_upper'
                            ] = const_error[time_index]
                else:
                    for time_index in range(len(times)):
                        ods_n[f'equilibrium.time_slice.{time_index}.constraints.flux_loop.{channel}.measured'] = numpy.nan
                        if error is not None:
                            ods_n[f'equilibrium.time_slice.{time_index}.constraints.flux_loop.{channel}.measured_error_upper'] = numpy.nan
            except Exception as _excp:
                raise _excp.__class__(f'Problem with flux_loop channel {channel}: {_excp}')

    # ip
    if 'ip' in constraints and 'magnetics.ip.0.data' in ods:
        average = averages.get('ip', default_average)
        printd(f'Processing magnetics.ip', topic='machine')
        try:
            # get
            data = ods['magnetics.ip.0.data']
            time = ods['magnetics.ip.0.time']
            if 'magnetics.ip.0.data_error_upper' in ods:
                error = ods['magnetics.ip.0.data_error_upper']
            else:
                error = None
            # process
            if rm_integr_drift_after is not None:
                data = remove_integrator_drift(time, data, rm_integr_drift_after)
            if cutoff_hz is not None:
                data = firFilter(time, data, cutoff_hz)
            const = smooth_by_convolution(data, time, times, average, **nuconv_kw)
            if error is not None:
                const_error = smooth_by_convolution(error, time, times, average, **nuconv_kw)
            # assign
            for time_index in range(len(times)):
                ods_n[f'equilibrium.time_slice.{time_index}.constraints.ip.measured'] = const[time_index]
                if error is not None:
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.ip.measured_error_upper'] = const_error[time_index]
        except Exception as _excp:
            raise _excp.__class__(f'Problem with ip: {_excp}')

    # diamagnetic_flux
    if 'diamagnetic_flux' in constraints and 'magnetics.diamagnetic_flux.0.data' in ods:
        average = averages.get('diamagnetic_flux', default_average)
        printd(f'Processing magnetics.diamagnetic_flux', topic='machine')
        try:
            # get
            data = ods['magnetics.diamagnetic_flux.0.data']
            time = ods['magnetics.diamagnetic_flux.0.time']
            if 'magnetics.diamagnetic_flux.0.data_error_upper' in ods:
                error = ods['magnetics.diamagnetic_flux.0.data_error_upper']
            else:
                error = None
            # process
            # if rm_integr_drift_after is not None:
            # drift is already removed?
            # data = remove_integrator_drift(time, data, rm_integr_drift_after)
            if cutoff_hz is not None:
                data = firFilter(time, data, cutoff_hz)
            const = smooth_by_convolution(data, time, times, average, **nuconv_kw)
            if error is not None:
                const_error = smooth_by_convolution(error, time, times, average, **nuconv_kw)
            # assign
            for time_index in range(len(times)):
                ods_n[f'equilibrium.time_slice.{time_index}.constraints.diamagnetic_flux.measured'] = const[time_index]
                if error is not None:
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.diamagnetic_flux.measured_error_upper'] = const_error[
                        time_index
                    ]
        except Exception as _excp:
            raise _excp.__class__(f'Problem with diamagnetic_flux: {_excp}')

    # b_field_tor_vacuum_r
    if 'b_field_tor_vacuum_r' in constraints and 'tf.b_field_tor_vacuum_r.data' in ods:
        printd(f'Processing tf.b_field_tor_vacuum_r', topic='machine')
        average = averages.get('b_field_tor_vacuum_r', default_average)
        try:
            # get
            data = ods['tf.b_field_tor_vacuum_r.data']
            time = ods['tf.b_field_tor_vacuum_r.time']
            if 'tf.b_field_tor_vacuum_r.data_error_upper' in ods:
                error = ods['tf.b_field_tor_vacuum_r.data_error_upper']
            else:
                error = None
            # process
            if rm_integr_drift_after is not None:
                data = remove_integrator_drift(time, data, rm_integr_drift_after)
            if cutoff_hz is not None:
                data = firFilter(time, data, cutoff_hz)
            const = smooth_by_convolution(data, time, times, average, **nuconv_kw)
            if error is not None:
                const_error = smooth_by_convolution(error, time, times, average, **nuconv_kw)
            # assign
            for time_index in range(len(times)):
                ods_n[f'equilibrium.time_slice.{time_index}.constraints.b_field_tor_vacuum_r.measured'] = const[time_index]
                if error is not None:
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.b_field_tor_vacuum_r.measured_error_upper'] = const_error[
                        time_index
                    ]
        except Exception as _excp:
            raise _excp.__class__(f'Problem with b_field_tor_vacuum_r: {_excp}')

    # mse
    if 'mse_polarisation_angle' in constraints and 'mse.channel.0.polarisation_angle.data' in ods:
        average = averages.get('mse_polarisation_angle', default_average)
        for channel in ods[f'mse.channel']:
            printd(f'Processing mse.channel.{channel}', topic='machine')

            try:
                # get
                label = ods[f'mse.channel.{channel}.name']
                for time_index in range(len(times)):
                    ods_n[f'equilibrium.time_slice.{time_index}.constraints.mse_polarisation_angle.{channel}.source'] = label
                valid = ods.get(
                    f'mse.channel.{channel}.polarisation_angle.validity', 1 - int(f'mse.channel.{channel}.polarisation_angle.data' in ods)
                )
                if valid == 0:  # 0 means that the data is good
                    data = copy.deepcopy(ods[f'mse.channel.{channel}.polarisation_angle.data'])
                    time = ods[f'mse.channel.{channel}.polarisation_angle.time']
                    if f'mse.channel.{channel}.polarisation_angle.data_error_upper' in ods:
                        error = ods[f'mse.channel.{channel}.polarisation_angle.data_error_upper']
                    else:
                        error = None
                    # process
                    if cutoff_hz is not None:
                        data = firFilter(time, data, cutoff_hz)
                    if f'mse.channel.{channel}.polarisation_angle.validity_timed' in ods:
                        data[ods[f'mse.channel.{channel}.polarisation_angle.validity_timed'] != 0] = numpy.nan
                        error[ods[f'mse.channel.{channel}.polarisation_angle.validity_timed'] != 0] = numpy.nan
                    const = smooth_by_convolution(data, time, times, average, **nuconv_kw)
                    if error is not None:
                        const_error = smooth_by_convolution(error, time, times, average, **nuconv_kw)
                    # assign
                    for time_index in range(len(times)):
                        ods_n[f'equilibrium.time_slice.{time_index}.constraints.mse_polarisation_angle.{channel}.measured'] = const[
                            time_index
                        ]
                        if error is not None:
                            ods_n[
                                f'equilibrium.time_slice.{time_index}.constraints.mse_polarisation_angle.{channel}.measured_error_upper'
                            ] = const_error[time_index]
                else:
                    for time_index in range(len(times)):
                        ods_n[f'equilibrium.time_slice.{time_index}.constraints.mse_polarisation_angle.{channel}.measured'] = numpy.nan
                        if error is not None:
                            ods_n[
                                f'equilibrium.time_slice.{time_index}.constraints.mse_polarisation_angle.{channel}.measured_error_upper'
                            ] = numpy.nan
               
                ods_n['mse.channel.:.active_spatial_resolution[0].geometric_coefficients']
                ods_n['mse.channel.:.active_spatial_resolution[0].centre.r']
                ods_n['mse.channel.:.active_spatial_resolution[0].centre.z']
                ods_n['mse.channel.:.active_spatial_resolution[0].centre.phi']

            except Exception as _excp:
                raise _excp.__class__(f'Problem with mse channel {channel}: {_excp}')

    return ods_n


@add_to__ODS__
@preprocess_ods('core_profiles', 'equilibrium')
def summary_greenwald(ods, update=True):
    """
    Calculates Greenwald Fraction for each time slice and stores them in the summary ods.

    :param ods: input ods

    :param update: operate in place

    :return: updated ods
    """

    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    a = (ods['equilibrium.time_slice.:.profiles_1d.r_outboard'][:, -1] - ods['equilibrium.time_slice.:.profiles_1d.r_inboard'][:, -1]) / 2
    ip = ods['equilibrium.time_slice.:.global_quantities.ip']
    nel = []
    for time_index in ods['equilibrium.time_slice']:
        with omas_environment(
            ods,
            coordsio={
                'core_profiles.profiles_1d.%d.grid.rho_tor_norm'
                % time_index: ods['equilibrium.time_slice.%s.profiles_1d.rho_tor_norm' % time_index]
            },
        ):
            ne = ods['core_profiles.profiles_1d.%d.electrons.density_thermal' % time_index]
            volume = ods['equilibrium.time_slice.%d.profiles_1d.volume' % time_index]
            ne_vol_avg = numpy.trapz(ne, x=volume) / volume[-1]

            if 'interferometer' in ods:
                ods.physics_summary_lineaverage_density()
                nel.append(ods['interferometer.channel.0.n_e_line_average.data'][time_index])
            else:
                print(
                    "Warning: greenwald fraction calculation used volume average density instead of line average fill in ods['interferometer'] to use nel"
                )
                nel.append(ne_vol_avg)
    ods_n['summary.global_quantities.greenwald_fraction.value'] = abs(numpy.array(nel) / 1e20 / ip * 1e6 * numpy.pi * a**2)
    ods_n['summary.time'] = ods['equilibrium.time']

    return ods_n


@add_to__ODS__
@preprocess_ods('core_profiles', 'equilibrium', 'interferometer')
def summary_lineaverage_density(ods, line_grid=2000, time_index=None, update=True, doPlot=False):
    """
    Calculates line-average electron density for each time slice and stores them in the summary ods

    :param ods: input ods

    :param line_grid: number of points to calculate line average density over (includes point outside of boundary)

    :param time_index: time slices to process

    :param update: operate in place

    :param doPlot: plots the interferometer lines on top of the equilibrium boundary shape

    :return: updated ods
    """
    import scipy

    if doPlot:
        from matplotlib import pyplot as plt

    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    if time_index is None:
        ods_n['summary.line_average.n_e.value'] = numpy.zeros(len(ods['core_profiles']['profiles_1d']))
        for time_index in range(len(ods['core_profiles']['profiles_1d'])):
            line_average_ne = summary_lineaverage_density(ods_n, line_grid=line_grid, time_index=time_index, update=update, doPlot=doPlot)[
                'interferometer'
            ]['channel'][0]['n_e_line_average']['data'][time_index]
            ods_n['summary.line_average.n_e.value'][time_index] = line_average_ne
        ods_n['summary.time'] = ods['equilibrium.time']
        return ods_n

    Rb = ods['equilibrium']['time_slice'][time_index]['boundary']['outline']['r']
    Zb = ods['equilibrium']['time_slice'][time_index]['boundary']['outline']['z']

    Rgrid = ods['equilibrium']['time_slice'][time_index]['profiles_2d'][0]['grid']['dim1']
    Zgrid = ods['equilibrium']['time_slice'][time_index]['profiles_2d'][0]['grid']['dim2']

    psi2d = ods['equilibrium']['time_slice'][time_index]['profiles_2d'][0]['psi']
    psi_spl = RectBivariateSpline(Rgrid, Zgrid, psi2d)
    psi_eq = ods['equilibrium']['time_slice'][time_index]['profiles_1d']['psi']
    rhon_eq = ods['equilibrium']['time_slice'][time_index]['profiles_1d']['rho_tor_norm']
    rhon_cp = ods['core_profiles']['profiles_1d'][time_index]['grid']['rho_tor_norm']
    ne = ods['core_profiles']['profiles_1d'][time_index]['electrons']['density_thermal']
    ne = numpy.interp(rhon_eq, rhon_cp, ne)
    tck = scipy.interpolate.splrep(psi_eq, ne, k=3)

    if 'time' not in ods['interferometer']:
        ods_n['interferometer.ids_properties.homogeneous_time'] = 1
        ods_n['interferometer']['time'] = copy.copy(ods['core_profiles']['time'])

    ifpaths = [['first_point', 'second_point'], ['second_point', 'third_point']]

    if doPlot:
        plt.plot(
            ods['equilibrium.time_slice[0].boundary.outline.r'], ods['equilibrium.time_slice[0].boundary.outline.z'], label='Boundary shape'
        )
        plt.xlabel('r [m]')
        plt.ylabel('z [m]')

    for channel in ods['interferometer']['channel']:
        ne_line_paths = []
        dist_paths = []
        for ifpath in ifpaths:
            R1 = ods['interferometer']['channel'][channel]['line_of_sight'][ifpath[0]]['r']
            Z1 = ods['interferometer']['channel'][channel]['line_of_sight'][ifpath[0]]['z']
            phi1 = ods['interferometer']['channel'][channel]['line_of_sight'][ifpath[0]]['phi']
            x1 = R1 * numpy.cos(phi1)
            y1 = R1 * numpy.sin(phi1)

            R2 = ods['interferometer']['channel'][channel]['line_of_sight'][ifpath[1]]['r']
            Z2 = ods['interferometer']['channel'][channel]['line_of_sight'][ifpath[1]]['z']
            phi2 = ods['interferometer']['channel'][channel]['line_of_sight'][ifpath[1]]['phi']
            x2 = R2 * numpy.cos(phi2)
            y2 = R2 * numpy.sin(phi2)

            xline = numpy.linspace(x1, x2, line_grid)
            yline = numpy.linspace(y1, y2, line_grid)
            Rline = numpy.linspace(R1, R2, line_grid)
            Zline = numpy.linspace(Z1, Z2, line_grid)
            dist = numpy.zeros(line_grid)

            if doPlot:
                plt.plot(Rline, Zline, label=f'interferometer path : {"-".join(ifpath)} channel:{channel}')
                plt.legend()

            for i, Rval in enumerate(Rline):
                dist[i] = numpy.min((Rline[i] - Rb) ** 2 + (Zline[i] - Zb) ** 2)

            zero_crossings = numpy.where(numpy.diff(numpy.sign(numpy.gradient(dist))))[0]
            i1 = zero_crossings[0]
            i2 = zero_crossings[-1]

            psival = [psi_spl(Rline[i], Zline[i], grid=False).item() for i in range(i1, i2, numpy.sign(i2 - i1))]
            ne_interp = scipy.interpolate.splev(psival, tck)
            ne_line = numpy.trapz(ne_interp)
            ne_line /= abs(i2 - i1)
            ne_line_paths.append(ne_line)
            dist_paths.append(numpy.sqrt((xline[i2] - xline[i1]) ** 2 + (yline[i2] - yline[i1]) ** 2 + (Zline[i2] - Zline[i1]) ** 2))

        ne_line = numpy.average(ne_line_paths, weights=dist_paths)
        if f'interferometer.channel.{channel}.n_e_line_average.data' not in ods_n:
            ods_n['interferometer']['channel'][channel]['n_e_line_average']['data'] = numpy.zeros(len(ods['interferometer']['time']))

        ods_n['interferometer']['channel'][channel]['n_e_line_average']['data'][time_index] = ne_line

    return ods_n


@add_to__ODS__
@preprocess_ods('core_profiles', 'equilibrium')
def summary_currents(ods, time_index=None, update=True):
    """
    Calculatess plasma currents from core_profiles for each time slice and stores them in the summary ods

    :param ods: input ods

    :param time_index: time slices to process

    :param update: operate in place

    :return: updated ods
    """

    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    current_names = [
        ('j_bootstrap', 'current_bootstrap.value'),
        ('j_non_inductive', 'current_non_inductive.value'),
        ('j_ohmic', 'current_ohm.value'),
    ]
    rho = ods['equilibrium.time_slice[0].profiles_1d.rho_tor_norm']
    time_index = 0
    coordsio = {'core_profiles.profiles_1d.%d.grid.rho_tor_norm' % time_index: rho}

    with omas_environment(ods, coordsio=coordsio):

        for (jname_cp, jname_sum) in current_names:
            if f'core_profiles.profiles_1d.{time_index}.{jname_cp}' in ods:
                Bt = ods['equilibrium.vacuum_toroidal_field.b0']
                JtoR = transform_current(
                    rho=rho,
                    JparB=ods['core_profiles']['profiles_1d'][time_index][jname_cp] * Bt,
                    equilibrium=ods['equilibrium.time_slice'][time_index],
                )
                Ip = numpy.trapz(JtoR, ods['equilibrium.time_slice[0].profiles_1d.volume']) / 2 / numpy.pi

                if f'summary.global_quantities.{jname_sum}' not in ods:
                    ods_n['summary.global_quantities'][jname_sum] = numpy.zeros(time_index + 1)

                    ods_n['summary.global_quantities'][jname_sum][time_index] = Ip

    return ods_n


@add_to__ODS__
@preprocess_ods('core_profiles', 'equilibrium')
def summary_thermal_stored_energy(ods, update=True):
    """
    Calculates the stored energy based on the contents of core_profiles for all time-slices

    :param ods: input ods

    :param update: operate in place

    :return: updated ods
    """
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    ods.physics_core_profiles_pressures()
    thermal_energy = []
    for time_index in ods['core_profiles.profiles_1d']:
        eq = ods[f'equilibrium.time_slice[{time_index}].profiles_1d']
        volume = numpy.interp(x=ods[f'core_profiles.profiles_1d.{time_index}.grid.rho_tor_norm'], xp=eq['rho_tor_norm'], fp=eq['volume'])
        thermal_energy.append(numpy.trapz(3 / 2 * ods['core_profiles.profiles_1d[0].pressure_thermal'], x=volume))

    ods_n['summary.global_quantities.energy_thermal.value'] = numpy.array(thermal_energy)
    ods_n['summary.time'] = ods['equilibrium.time']

    return ods_n


@add_to__ODS__
@preprocess_ods('core_profiles', 'core_sources', 'equilibrium')
def summary_taue(ods, thermal=True, update=True):
    """
    Calculates Energy confinement time estimated from the IPB98(y,2) scaling for each time slice and stores them in the summary ods

    :param ods: input ods

    :thermal: calculates the thermal part of the energy confinement time from core_profiles if True, otherwise use the stored energy MHD from the equilibrium ods

    :param update: operate in place

    :return: updated ods
    """
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    tau_e_scaling = []
    tau_e_MHD = []
    for time_index in ods['equilibrium']['time_slice']:
        equilibrium_ods = ods['equilibrium']['time_slice'][time_index]
        a = (equilibrium_ods['profiles_1d']['r_outboard'][-1] - equilibrium_ods['profiles_1d']['r_inboard'][-1]) / 2
        r_major = (equilibrium_ods['profiles_1d']['r_outboard'][-1] + equilibrium_ods['profiles_1d']['r_inboard'][-1]) / 2
        bt = ods['equilibrium']['vacuum_toroidal_field']['b0'][time_index] * ods['equilibrium']['vacuum_toroidal_field']['r0'] / r_major
        ip = equilibrium_ods['global_quantities']['ip']
        aspect = r_major / a
        psi = ods['equilibrium']['time_slice'][time_index]['profiles_1d']['psi']
        rho_tor_norm_equi = equilibrium_ods['profiles_1d']['rho_tor_norm']
        rho_tor_norm_core = ods['core_profiles']['profiles_1d'][time_index]['grid']['rho_tor_norm']
        psi = numpy.interp(rho_tor_norm_core, rho_tor_norm_equi, psi)
        # Making Equilibrium grid same grid as core_sources and core_profiles
        with omas_environment(ods, coordsio={'equilibrium.time_slice.0.profiles_1d.psi': psi}):
            volume = equilibrium_ods['profiles_1d']['volume']
            kappa = volume[-1] / 2 / numpy.pi / numpy.pi / a / a / r_major
            ne = ods['core_profiles']['profiles_1d'][time_index]['electrons']['density_thermal']
            ne_vol_avg = numpy.trapz(ne, x=volume) / volume[-1]

            if 'interferometer' in ods:
                ods.physics_summary_lineaverage_density()
                nel = ods['interferometer.channel.0.n_e_line_average.data'][time_index]
            else:
                print(
                    "Warning: taue calculation used volume average density instead of line average fill in ods['interferometer'] to use nel"
                )
                nel = ne_vol_avg
            # Naive weighted isotope average:
            n_deuterium_avg = 0.0
            n_tritium_avg = 0.0
            ions = ods['core_profiles']['profiles_1d'][time_index]['ion']
            for ion in ods['core_profiles']['profiles_1d'][time_index]['ion']:
                if ions[ion]['label'] == 'D' or 'd':
                    n_deuterium_avg = numpy.trapz(ions[ion]['density_thermal'], x=volume)
                elif ions[ion]['label'] == 'T' or 't':
                    n_tritium_avg = numpy.trapz(ions[ion]['density_thermal'], x=volume)
            isotope_factor = (2.014102 * n_deuterium_avg + 3.016049 * n_tritium_avg) / (n_deuterium_avg + n_tritium_avg)

            info_string = ''
            # Get total power from ods function:
            if 'power_loss' in ods['summary.global_quantities']:
                power_loss = ods['summary.global_quantities.power_loss.value'][time_index]
                info_string += "Power from: summary.global_quantities.power_loss.value,  "
            elif 'power_steady' in ods['summary.global_quantities']:
                print("Warning: taue calculation used power steady instead of power_loss")
                ods.physics_summary_heating_power()
                power_loss = ods['summary.global_quantities.power_steady.value'][time_index]
                info_string += "INACCURATE Power from: summary.global_quantities.power_steady.value,  "
            else:
                return ods_n
            # Stored energy from profiles or equilibrium
            if 'summary.global_quantities.energy_thermal' in ods and thermal:
                stored_energy = ods['summary.global_quantities.energy_thermal.value'][time_index]
                info_string += "Stored energy from: summary.global_quantities.energy_thermal.value"
            elif 'global_quantities.energy_mhd' in equilibrium_ods:
                if thermal:
                    print("Warning, tau_e calculated with stored energy MHD")
                stored_energy = equilibrium_ods['global_quantities']['energy_mhd']
                info_string += "Stored energy from: 'global_quantities']['energy_mhd"
            else:
                return ods_n
            # Calculate tau_e
            tau_e = abs(
                56.2e-3
                * (abs(ip) / 1e6) ** 0.93
                * abs(bt) ** 0.15
                * (nel / 1e19) ** 0.41
                * (power_loss / 1e6) ** -0.69
                * r_major**1.97
                * kappa**0.78
                * aspect**-0.58
                * isotope_factor**0.19
            )  # [s]
            for k in ['kappa', 'bt', 'ip', 'nel', 'power_loss', 'aspect', 'isotope_factor', 'tau_e']:
                printd(f'{k}: {eval(k)}', topic='summary_taue')
            tau_e_scaling.append(tau_e)
            tau_e_MHD.append(stored_energy / power_loss)

    # assign quantities in the ODS
    ods_n['summary']['global_quantities']['tau_energy_98']['value'] = numpy.array(tau_e_scaling)
    ods_n['summary']['global_quantities']['tau_energy']['value'] = numpy.array(tau_e_MHD)

    ods_n['summary.global_quantities.tau_energy.source'] = info_string
    ods_n['summary.global_quantities.tau_energy_98.source'] = "h98y2 scaling law"

    ods_n['summary.time'] = ods['equilibrium.time']

    return ods_n


@add_to__ODS__
@preprocess_ods('core_sources')
def summary_heating_power(ods, update=True):
    """
    Integrate power densities to the total and heating and current drive systems and fills summary.global_quantities

    :param ods: input ods

    :param update: operate in place

    :return: updated ods
    """
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    if 'core_sources' not in ods:
        return ods_n
    sources = ods_n['core_sources']['source']
    index_dict = {2: 'nbi', 3: 'ec', 4: 'lh', 5: 'ic', 6: 'fusion', 7: 'ohmic'}
    power_dict = {'total_heating': [], 'nbi': [], 'ec': [], 'lh': [], 'ic': [], 'fusion': []}
    if 'core_sources.source.0' not in ods_n:
        return ods_n
    q_init = numpy.zeros([len(ods['core_sources']['time']),
                         len(sources[0]['profiles_1d'][0]['grid']['rho_tor_norm'])])

    q_dict = {
        'total_heating': copy.deepcopy(q_init),
        'nbi': copy.deepcopy(q_init),
        'ec': copy.deepcopy(q_init),
        'lh': copy.deepcopy(q_init),
        'ic': copy.deepcopy(q_init),
        'fusion': copy.deepcopy(q_init),
    }

    for time_index in sources[0]['profiles_1d']:
        vol = sources[0]['profiles_1d'][time_index]['grid']['volume']
        for source in sources:
            source_1d = sources[source]['profiles_1d'][time_index]
            if sources[source]['identifier.index'] in index_dict:
                if 'electrons' in source_1d and 'energy' in source_1d['electrons']:
                    q_dict['total_heating'][time_index,:] += source_1d['electrons']['energy']
                    if sources[source]['identifier.index'] in index_dict and index_dict[sources[source]['identifier.index']] in q_dict:
                        q_dict[index_dict[sources[source]['identifier.index']]][time_index,:]  += source_1d['electrons']['energy']
                if 'total_ion_energy' in source_1d:
                    q_dict['total_heating'][time_index,:] += source_1d['total_ion_energy']
                    if sources[source]['identifier.index'] in index_dict and index_dict[sources[source]['identifier.index']] in q_dict:
                        q_dict[index_dict[sources[source]['identifier.index']]][time_index,:] += source_1d['total_ion_energy']

    for key, value in power_dict.items():
        power_dict[key] = numpy.trapz(q_dict[key], x=vol,axis=1)
        if numpy.sum(power_dict[key]) > 0:
            if key == 'total_heating':
                ods_n['summary.global_quantities.power_steady.value'] = numpy.array(power_dict[key])
                continue
            elif key == 'fusion':
                ods_n['summary.fusion.power.value'] = numpy.array(power_dict[key])
                ods_n['summary.fusion.neutron_power_total.value'] = (14.1 / 3.5) * numpy.array(power_dict[key])
                continue
            ods_n[f'summary.heating_current_drive.{key}[0].power.value'] = numpy.array(power_dict[key])

    ods_n['summary.time'] = ods['equilibrium.time']
    return ods_n


@add_to__ODS__
@preprocess_ods()
def summary_global_quantities(ods, update=True):
    """
    Calculates global quantities for each time slice and stores them in the summary ods:
     - Greenwald Fraction
     - Energy confinement time estimated from the IPB98(y,2) scaling
     - Integrate power densities to the totals
     - Generate summary.global_quantities from global_quantities of other IDSs

    :param ods: input ods

    :param update: operate in place

    :return: updated ods
    """
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    ods_n.update(ods.physics_summary_greenwald(update=update))
    ods_n.update(ods.physics_summary_currents(update=update))
    ods_n.update(ods.physics_summary_thermal_stored_energy(update=update))
    ods_n.update(ods.physics_summary_heating_power(update=update))
    ods_n.update(ods.physics_summary_taue(update=update))
    ods_n.update(ods.physics_summary_consistent_global_quantities(update=update))

    return ods_n


@add_to__ODS__
def summary_consistent_global_quantities(ods, ds=None, update=True):
    """
    Generate summary.global_quantities from global_quantities of other IDSs

    :param ods: input ods

    :param ds: IDS from which to update summary.global_quantities. All IDSs if `None`.

    :param update: operate in place

    :return: updated ods
    """
    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    global_quantities = copy.copy(omas_global_quantities(ods.imas_version))

    if ds is None:
        ds = set(map(lambda x: x.split('.')[0], global_quantities))
    ds = set(ds)
    if 'summary' in ds:
        ds.remove('summary')

    # global_quantities destinations
    dst = []
    for item in global_quantities:
        path = item.split('.')
        if path[0] == 'summary':
            dst.append(path[-1])

    # global_quantities sources
    src = []
    for item in global_quantities:
        path = item.split('.')
        if path[0] in ds and path[0] in ods and path[-1] in dst:
            src.append(item)

    # copy global_quantities from other IDSs to summary
    for item in src:
        if item.replace(':', '0') in ods:
            path = item.split('.')
            ods_n[f'summary.global_quantities.{path[-1]}.value'] = ods[item]
            ods_n[f'summary.global_quantities.{path[-1]}.source'] = 'Consistency with ' + path[0]

    return ods_n


@add_to__ODS__
@preprocess_ods()
def core_profiles_consistent(ods, update=True, use_electrons_density=False, enforce_quasineutrality=False):
    """
    Calls all core_profiles consistency functions including
      - core_profiles_densities
      - core_profiles_pressures
      - core_profiles_zeff

    :param ods: input ods

    :param update: operate in place

    :param use_electrons_density:
            denominator is core_profiles.profiles_1d.:.electrons.density
            instead of sum Z*n_i in Z_eff calculation

    :param enforce_quasineutrality: update electron density to be quasineutral with ions

    :return: updated ods
    """
    ods = core_profiles_densities(ods, update=update, enforce_quasineutrality=enforce_quasineutrality)
    core_profiles_pressures(ods)
    core_profiles_zeff(ods, use_electrons_density=use_electrons_density, enforce_quasineutrality=enforce_quasineutrality)
    return ods


@add_to__ODS__
@preprocess_ods('core_profiles')
def core_profiles_pressures(ods, update=True):
    """
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
    """
    ods_p = ods
    if not update:
        from omas import ODS

        ods_p = ODS().copy_attrs_from(ods)

    for time_index in ods['core_profiles']['profiles_1d']:
        prof1d = ods['core_profiles']['profiles_1d'][time_index]
        prof1d_p = ods_p['core_profiles']['profiles_1d'][time_index]

        if not update:
            prof1d_p['grid']['rho_tor_norm'] = prof1d['grid']['rho_tor_norm']

        __zeros__ = 0.0 * prof1d['grid']['rho_tor_norm']

        prof1d_p['pressure_thermal'] = copy.deepcopy(__zeros__)
        prof1d_p['pressure_ion_total'] = copy.deepcopy(__zeros__)
        prof1d_p['pressure_perpendicular'] = copy.deepcopy(__zeros__)
        prof1d_p['pressure_parallel'] = copy.deepcopy(__zeros__)

        # electrons
        prof1d_p['electrons']['pressure'] = copy.deepcopy(__zeros__)

        __p__ = None
        if 'density_thermal' in prof1d['electrons'] and 'temperature' in prof1d['electrons']:
            __p__ = nominal_values(prof1d['electrons']['density_thermal'] * prof1d['electrons']['temperature'] * constants.e)
        elif 'pressure_thermal' in prof1d['electrons']:
            __p__ = nominal_values(prof1d['electrons']['pressure_thermal'])

        if __p__ is not None:
            prof1d_p['electrons']['pressure_thermal'] = __p__
            prof1d_p['electrons']['pressure'] += __p__
            prof1d_p['pressure_thermal'] += __p__
            prof1d_p['pressure_perpendicular'] += __p__ / 3.0
            prof1d_p['pressure_parallel'] += __p__ / 3.0

        if 'pressure_fast_perpendicular' in prof1d['electrons']:
            __p__ = nominal_values(prof1d['electrons']['pressure_fast_perpendicular'])
            if not update:
                prof1d_p['electrons']['pressure_fast_perpendicular'] = __p__
            prof1d_p['electrons']['pressure'] += 2.0 * __p__
            prof1d_p['pressure_perpendicular'] += __p__

        if 'pressure_fast_parallel' in prof1d['electrons']:
            __p__ = nominal_values(prof1d['electrons']['pressure_fast_parallel'])
            if not update:
                prof1d_p['electrons']['pressure_fast_parallel'] = __p__
            prof1d_p['electrons']['pressure'] += __p__
            prof1d_p['pressure_parallel'] += __p__

        # ions
        for k in range(len(prof1d['ion'])):

            prof1d_p['ion'][k]['pressure'] = copy.deepcopy(__zeros__)

            __p__ = None
            if 'density_thermal' in prof1d['ion'][k] and 'temperature' in prof1d['ion'][k]:
                __p__ = nominal_values(prof1d['ion'][k]['density_thermal'] * prof1d['ion'][k]['temperature'] * constants.e)
            elif 'pressure_thermal' in prof1d['ion'][k]:
                __p__ = nominal_values(prof1d['ion'][k]['pressure_thermal'])

            if __p__ is not None:
                prof1d_p['ion'][k]['pressure_thermal'] = __p__
                prof1d_p['ion'][k]['pressure'] += __p__
                prof1d_p['pressure_thermal'] += __p__
                prof1d_p['pressure_perpendicular'] += __p__ / 3.0
                prof1d_p['pressure_parallel'] += __p__ / 3.0
                prof1d_p['pressure_ion_total'] += __p__

            if 'pressure_fast_perpendicular' in prof1d['ion'][k]:
                __p__ = nominal_values(prof1d['ion'][k]['pressure_fast_perpendicular'])
                if not update:
                    prof1d_p['ion'][k]['pressure_fast_perpendicular'] = __p__
                prof1d_p['ion'][k]['pressure'] += 2.0 * __p__
                prof1d_p['pressure_perpendicular'] += __p__

            if 'pressure_fast_parallel' in prof1d['ion'][k]:
                __p__ = nominal_values(prof1d['ion'][k]['pressure_fast_parallel'])
                if not update:
                    prof1d_p['ion'][k]['pressure_fast_parallel'] = __p__
                prof1d_p['ion'][k]['pressure'] += __p__
                prof1d_p['pressure_parallel'] += __p__

        # extra pressure information that is not within IMAS structure is set only if consistency_check is False
        if ods_p.consistency_check is False:
            prof1d_p['pressure'] = prof1d_p['pressure_perpendicular'] * 2 + prof1d_p['pressure_parallel']
            prof1d_p['pressure_electron_total'] = prof1d_p['pressure_thermal'] - prof1d_p['pressure_ion_total']
            prof1d_p['pressure_fast'] = prof1d_p['pressure'] - prof1d_p['pressure_thermal']

    return ods_p


@add_to__ODS__
@preprocess_ods('core_profiles')
def core_profiles_densities(ods, update=True, enforce_quasineutrality=False):
    """
    Density, density_thermal, and density_fast for electrons and ions are filled and are self-consistent

    :param ods: input ods

    :param update: operate in place

    :param enforce_quasineutrality: update electron density to be quasineutral with ions

    :return: updated ods
    """

    ods_n = ods
    if not update:
        from omas import ODS

        ods_n = ODS().copy_attrs_from(ods)

    def consistent_density(loc):
        if 'density' in loc:

            # if there is no thermal nor fast, assume it is thermal
            if 'density_thermal' not in loc and 'density_fast' not in loc:
                loc['density_thermal'] = loc['density']

            # if there is no thermal calculate it
            elif 'density_thermal' not in loc and 'density_fast' in loc:
                loc['density_thermal'] = loc['density'] - loc['density_fast']

            # if there is no fast calculate it
            elif 'density_thermal' in loc and 'density_fast' not in loc:
                loc['density_fast'] = loc['density'] - loc['density_thermal']

        # enforce self-consistency
        loc['density'] = copy.deepcopy(__zeros__)
        for density in ['density_thermal', 'density_fast']:
            if density in loc:
                loc['density'] += loc[density]
            else:
                loc[density] = copy.deepcopy(__zeros__)

    for time_index in ods['core_profiles']['profiles_1d']:
        prof1d = ods['core_profiles']['profiles_1d'][time_index]
        prof1d_n = ods_n['core_profiles']['profiles_1d'][time_index]

        if not update:
            prof1d_n['grid']['rho_tor_norm'] = prof1d['grid']['rho_tor_norm']

        __zeros__ = 0.0 * prof1d['grid']['rho_tor_norm']

        # electrons
        consistent_density(prof1d_n['electrons'])

        # ions
        for k in range(len(prof1d['ion'])):
            consistent_density(prof1d_n['ion'][k])

        if enforce_quasineutrality:
            ne_q = copy.deepcopy(__zeros__)
            for k in range(len(prof1d_n['ion'])):
                ne_q += prof1d_n[f'ion[{k}].element[0].z_n'] * prof1d_n[f'ion[{k}].density']
            qnfac = ne_q / (prof1d_n[f'electrons.density'] + numpy.finfo(numpy.float64).tiny)
            for den in ['density', 'density_fast', 'density_thermal']:
                prof1d_n['electrons'][den] *= qnfac

    return ods_n


@add_to__ODS__
@preprocess_ods('core_profiles')
def core_profiles_zeff(ods, update=True, use_electrons_density=False, enforce_quasineutrality=False):
    """
    calculates effective charge

    :param ods: input ods

    :param update: operate in place

    :param use_electrons_density:
            denominator core_profiles.profiles_1d.:.electrons.density
            instead of sum Z*n_i

    :param enforce_quasineutrality: update electron density to be quasineutral with ions

    :return: updated ods
    """

    ods_z = core_profiles_densities(ods, update=update, enforce_quasineutrality=enforce_quasineutrality)

    for time_index in ods['core_profiles']['profiles_1d']:
        prof1d = ods['core_profiles']['profiles_1d'][time_index]
        prof1d_z = ods_z['core_profiles']['profiles_1d'][time_index]

        Z2n = 0.0 * prof1d_z['grid']['rho_tor_norm']
        Zn = 0.0 * prof1d_z['grid']['rho_tor_norm']

        for k in range(len(prof1d['ion'])):
            Z = prof1d['ion'][k]['element'][0]['z_n']  # from old ODS
            n = prof1d_z['ion'][k]['density']  # from new ODS
            Z2n += n * Z**2
            Zn += n * Z
        if use_electrons_density:
            prof1d_z['zeff'] = Z2n / prof1d_z['electrons']['density']
        else:
            prof1d_z['zeff'] = Z2n / Zn
    return ods_z


@add_to__ODS__
@preprocess_ods('equilibrium', 'core_profiles')
def current_from_eq(ods, time_index):
    """
    This function sets the currents in ods['core_profiles']['profiles_1d'][time_index]
    using ods['equilibrium']['time_slice'][time_index]['profiles_1d']['j_tor']

    :param ods: ODS to update in-place

    :param time_index: ODS time index to updated
    if None, all times are updated
    """

    # run an all time slices if time_index is None
    if time_index is None:
        for itime in ods['equilibrium.time_slice']:
            current_from_eq(ods, time_index=itime)
        return

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

        JparB_tot = transform_current(rho, JtoR=JtoR_tot, equilibrium=ods['equilibrium']['time_slice'][time_index], includes_bootstrap=True)

    try:
        core_profiles_currents(ods, time_index, rho, j_total=JparB_tot / B0)
    except AssertionError:
        # redo but wipe out old current components since we can't make it consistent
        core_profiles_currents(
            ods, time_index, rho, j_actuator=None, j_bootstrap=None, j_ohmic=None, j_non_inductive=None, j_total=JparB_tot / B0
        )

    return


@add_to__ODS__
@preprocess_ods('equilibrium', 'core_profiles')
def core_profiles_currents(
    ods,
    time_index=None,
    rho_tor_norm=None,
    j_actuator='default',
    j_bootstrap='default',
    j_ohmic='default',
    j_non_inductive='default',
    j_total='default',
    warn=True,
):
    """
    This function sets currents in ods['core_profiles']['profiles_1d'][time_index]

    If provided currents are inconsistent with each other or ods, ods is not updated and an error is thrown.

    Updates integrated currents in ods['core_profiles']['global_quantities']
    (N.B.: `equilibrium` IDS is required for evaluating j_tor and integrated currents)

    :param ods: ODS to update in-place

    :param time_index: ODS time index to updated
    if None, all times are updated

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

    # run an all time slices if time_index is None
    if time_index is None:
        for itime in ods['core_profiles.profiles_1d']:
            core_profiles_currents(
                ods,
                time_index=itime,
                rho_tor_norm=rho_tor_norm,
                j_actuator=j_actuator,
                j_bootstrap=j_bootstrap,
                j_ohmic=j_ohmic,
                j_non_inductive=j_non_inductive,
                j_total=j_total,
                warn=warn,
            )
        return

    try:
        from scipy.integrate import cumulative_trapezoid as cumtrapz
    except ImportError:
        from scipy.integrate import cumtrapz

    prof1d = ods['core_profiles.profiles_1d'][time_index]

    if rho_tor_norm is None:
        rho_tor_norm = prof1d['grid.rho_tor_norm']

    # SETUP DEFAULTS
    data = {}
    with omas_environment(ods, coordsio={'core_profiles.profiles_1d.%d.grid.rho_tor_norm' % time_index: rho_tor_norm}):
        for j in ['j_actuator', 'j_bootstrap', 'j_non_inductive', 'j_ohmic', 'j_total']:
            if isinstance(eval(j), str) and eval(j) == 'default':
                if j in prof1d:
                    data[j] = copy.deepcopy(prof1d[j])
                elif (j == 'j_actuator') and 'core_sources' in ods:
                    data[j] = core_sources_j_parallel_sum(ods)
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
        fsa_invR = omas_interp1d(rho_tor_norm, eq['profiles_1d']['rho_tor_norm'], eq['profiles_1d']['gm9'])
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

    err = "Cannot set j_actuator without j_bootstrap provided or calculable"
    if j_actuator is not None:
        assert j_bootstrap is not None, err

    # j_non_inductive
    err = 'j_non_inductive inconsistent with j_actuator and j_bootstrap'
    if (j_non_inductive is not None) and ((j_actuator is not None) or (j_bootstrap is not None)):
        assert numpy.allclose(j_non_inductive, j_actuator + j_bootstrap), err

    # j_total
    err = 'j_total inconsistent with j_ohmic and j_non_inductive'
    if (j_total is not None) and ((j_ohmic is not None) or (j_non_inductive is not None)):
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

        currents = [('j_bootstrap', 'current_bootstrap', True), ('j_non_inductive', 'current_non_inductive', True), ('j_tor', 'ip', False)]

        for Jname, Iname, transform in currents:
            if Jname in prof1d:
                J = prof1d[Jname]
                if transform:
                    # transform <J.B>/B0 to <Jt/R>
                    J = transform_current(rho_eq, JparB=J * B0, equilibrium=eq, includes_bootstrap=True)
                else:
                    # already <Jt/R>/<1/R>
                    J *= fsa_invR
                ods.set_time_array('core_profiles.global_quantities.%s' % Iname, time_index, cumtrapz(vp * J, psi)[-1] / (2.0 * numpy.pi))
            elif 'core_profiles.global_quantities.%s' % Iname in ods:
                # set current to zero if this time_index exists already
                if time_index < len(ods['core_profiles.global_quantities.%s' % Iname]):
                    ods['core_profiles.global_quantities.%s' % Iname][time_index] = 0.0

    return


@add_to__ODS__
@preprocess_ods()
def wall_add(ods, machine=None):
    """
    Add wall information to the ODS

    :param ods: ODS to update in-place

    :param machine: machine of which to load the wall (if None it is taken from ods['dataset_description.data_entry.machine'])
    """
    if machine is None:
        if 'machine' in ods['dataset_description.data_entry']:
            machine = ods['dataset_description.data_entry.machine']
        else:
            raise LookupError('Could not figure out what machine wall to use: dataset_description.data_entry.machine is not set')

    # fmt: off
    walls = {}
    walls['iter'] = {'RLIM': [6.267, 7.283, 7.899, 8.306, 8.395, 8.27, 7.904, 7.4,
                              6.587, 5.753, 4.904, 4.311, 4.126, 4.076, 4.046, 4.046,
                              4.067, 4.097, 4.178, 3.9579, 4.0034, 4.1742, 4.3257, 4.4408,
                              4.5066, 4.5157, 4.467, 4.4064, 4.4062, 4.3773, 4.3115, 4.2457,
                              4.1799, 4.4918, 4.5687, 4.6456, 4.8215, 4.9982, 5.1496, 5.2529,
                              5.2628, 5.2727, 5.565, 5.565, 5.565, 5.565, 5.572, 5.572,
                              5.572, 5.572, 5.6008, 5.6842, 5.815, 5.9821, 6.171, 6.3655,
                              6.267],
                     'ZLIM': [-3.036, -2.247, -1.332, -0.411, 0.643, 1.691, 2.474,
                              3.189, 3.904, 4.542, 4.722, 4.334, 3.592, 2.576,
                              1.559, 0.543, -0.474, -1.49, -2.496, -2.5284, -2.5284,
                              -2.5574, -2.6414, -2.7708, -2.931, -3.1039, -3.2701, -3.3943,
                              -3.3948, -3.4699, -3.6048, -3.7397, -3.8747, -3.8992, -3.8176,
                              -3.736, -3.699, -3.7314, -3.8282, -3.9752, -4.1144, -4.2536,
                              -4.5459, -4.3926, -4.2394, -4.0862, -3.9861, -3.9856, -3.886,
                              -3.885, -3.6924, -3.5165, -3.3723, -3.2722, -3.225, -3.2346,
                              -3.036]}
    walls['west'] = {'RLIM': [2.86135614, 2.87861924, 2.90016384, 2.91997554, 2.93800414,
                              2.95420434, 2.96853494, 2.98095994, 2.99144794, 2.99997234,
                              3.00651164, 3.01104944, 3.01357414, 3.01407934, 3.01256394,
                              3.00903154, 3.00349134, 2.99595704, 2.98644784, 2.97498774,
                              2.96160574, 2.94633544, 2.92921564, 2.91028954, 2.88960484,
                              2.86721394, 3.1298712, 3.1117572, 3.0928301, 3.0731123,
                              3.0526269, 3.031398, 3.0094508, 2.9868111, 2.9635054,
                              2.9395614, 2.9150072, 2.8898718, 2.8641847, 2.8379762,
                              2.8112773, 2.766, 2.735, 2.704, 2.673,
                              2.642, 2.611, 2.58, 2.549, 2.518,
                              2.487, 2.456, 2.4446, 2.4199, 2.3952,
                              2.3705, 2.3458, 2.3211, 2.2965, 2.2718,
                              2.2471, 2.2224, 2.1977, 2.173, 2.1483,
                              2.1236, 2.0989, 2.0742, 2.0495, 2.0249,
                              2.0002, 1.9755, 1.9508, 1.9261, 1.9014,
                              1.8966603, 1.8901254, 1.8839482, 1.8781297, 1.8726708,
                              1.8675721, 1.8628345, 1.8584587, 1.8544453, 1.8507948,
                              1.8475079, 1.844585, 1.8420265, 1.8398328, 1.8380042,
                              1.836541, 1.8354435, 1.8347117, 1.8343457, 1.8343457,
                              1.8347117, 1.8354435, 1.836541, 1.8380042, 1.8398328,
                              1.8420265, 1.844585, 1.8475079, 1.8507948, 1.8544453,
                              1.8584587, 1.8628345, 1.8675721, 1.8726708, 1.8781297,
                              1.8839482, 1.8901254, 1.8966603, 1.9091, 1.928,
                              1.9468, 1.9657, 1.9845, 2.0034, 2.0222,
                              2.0411, 2.0599, 2.0788, 2.0977, 2.1165,
                              2.1354, 2.1542, 2.1731, 2.1919, 2.2108,
                              2.2297, 2.2485, 2.2674, 2.2862, 2.3051,
                              2.3239, 2.3428, 2.3616, 2.3805, 2.3895,
                              2.43351111, 2.47752222, 2.52153333, 2.56554444, 2.60955556,
                              2.65356667, 2.69757778, 2.74158889, 2.7856, 2.802274,
                              2.8291295, 2.855505, 2.8813692, 2.9066919, 2.9314431,
                              2.9555937, 2.9791152, 3.0019799, 3.0241609, 3.045632,
                              3.0663679, 3.0863443, 3.1055375, 3.1239249, 2.86135614],
                     'ZLIM': [-0.4702282, -0.44550049, -0.41155163, -0.37656315, -0.34062343,
                              -0.30382328, -0.26625564, -0.22801541, -0.1891992, -0.14990505,
                              -0.11023223, -0.07028096, -0.03015215, 0.01005283, 0.05023242,
                              0.09028511, 0.13010973, 0.16960569, 0.20867321, 0.2472136,
                              0.2851295, 0.32232515, 0.35870657, 0.39418187, 0.42866144,
                              0.46205816, 0.51562535, 0.53962674, 0.56299231, 0.58569451,
                              0.60770659, 0.62900263, 0.64955753, 0.66934707, 0.68834792,
                              0.70653771, 0.723895, 0.74039934, 0.75603128, 0.77077241,
                              0.78460535, 0.7492, 0.7492, 0.7492, 0.7492,
                              0.7492, 0.7492, 0.7492, 0.7492, 0.7492,
                              0.7492, 0.7492, 0.7986, 0.7886, 0.7787,
                              0.7687, 0.7587, 0.7487, 0.7388, 0.7288,
                              0.7188, 0.7088, 0.6989, 0.6889, 0.6789,
                              0.6689, 0.659, 0.649, 0.639, 0.6291,
                              0.6191, 0.6091, 0.5991, 0.5892, 0.5792,
                              0.555, 0.52546258, 0.49584828, 0.46616143, 0.43640637,
                              0.40658745, 0.37670903, 0.3467755, 0.31679123, 0.28676061,
                              0.25668802, 0.22657788, 0.19643458, 0.16626254, 0.13606618,
                              0.1058499, 0.07561814, 0.04537531, 0.01512584, -0.01512584,
                              -0.04537531, -0.07561814, -0.1058499, -0.13606618, -0.16626254,
                              -0.19643458, -0.22657788, -0.25668802, -0.28676061, -0.31679123,
                              -0.3467755, -0.37670903, -0.40658745, -0.43640637, -0.46616143,
                              -0.49584828, -0.52546258, -0.555, -0.5798, -0.5874,
                              -0.595, -0.6026, -0.6102, -0.6178, -0.6254,
                              -0.633, -0.6406, -0.6482, -0.6558, -0.6634,
                              -0.671, -0.6786, -0.6862, -0.6938, -0.7014,
                              -0.709, -0.7166, -0.7242, -0.7318, -0.7394,
                              -0.747, -0.7546, -0.7622, -0.7698, -0.6754,
                              -0.6754, -0.6754, -0.6754, -0.6754, -0.6754,
                              -0.6754, -0.6754, -0.6754, -0.6754, -0.78901166,
                              -0.77548512, -0.76104485, -0.74570785, -0.7294922, -0.71241701,
                              -0.69450238, -0.67576944, -0.65624025, -0.63593783, -0.61488609,
                              -0.59310985, -0.57063475, -0.54748729, -0.52369473, -0.4702282]}
    # fmt: on

    if machine.lower() not in walls:
        raise LookupError('OMAS wall information only available for: %s' % walls.keys())

    ods['wall.description_2d.+.limiter.type.name'] = 'first_wall'
    ods['wall.description_2d.-1.limiter.type.index'] = 0
    ods['wall.description_2d.-1.limiter.type.description'] = 'first wall'
    ods['wall.description_2d.-1.limiter.unit.0.outline.r'] = walls[machine.lower()]['RLIM']
    ods['wall.description_2d.-1.limiter.unit.0.outline.z'] = walls[machine.lower()]['ZLIM']


@add_to__ODS__
@preprocess_ods('equilibrium')
def equilibrium_consistent(ods):
    """
    Calculate missing derived quantities for equilibrium IDS

    :param ods: ODS to update in-place

    :return: updated ods
    """
    for time_index in ods['equilibrium.time_slice']:
        eq = ods['equilibrium.time_slice'][time_index]
        eq1d = ods['equilibrium.time_slice'][time_index]['profiles_1d']

        psi = eq1d['psi']
        psi_norm = abs(psi - psi[0]) / abs(psi[-1] - psi[0])
        psirz = eq['profiles_2d.0.psi']
        psirz_norm = abs(psirz - psi[0]) / abs(psi[-1] - psi[0])
        fpol = eq1d['f']

        # extend functions in PSI to be clamped at edge value when outside of PSI range (i.e. outside of LCFS)
        ext_psi_norm_mesh = numpy.hstack((psi_norm[0] - 1e6, psi_norm, psi_norm[-1] + 1e6))

        def ext_arr(inv):
            return numpy.hstack((inv[0], inv, inv[-1]))

        fpol = omas_interp1d(psirz_norm, ext_psi_norm_mesh, ext_arr(fpol))
        Z, R = numpy.meshgrid(eq['profiles_2d.0.grid.dim2'], eq['profiles_2d.0.grid.dim1'])
        eq['profiles_2d.0.r'] = R
        eq['profiles_2d.0.z'] = Z
        eq['profiles_2d.0.b_field_tor'] = fpol / R

    equilibrium_stored_energy(ods, update=True)

    return ods


@add_to__ODS__
@preprocess_ods('equilibrium')
def equilibrium_transpose_RZ(ods, flip_dims=False):
    """
    Transpose 2D grid values for RZ grids under equilibrium.time_slice.:.profiles_2d.:.

    :param ods: ODS to update in-place

    :param flip_dims: whether to switch the equilibrium.time_slice.:.profiles_2d.:.grid.dim1 and dim1

    :return: updated ods
    """
    for time_index in ods['equilibrium.time_slice']:
        for grid in ods['equilibrium.time_slice'][time_index]['profiles_2d']:
            if ods['equilibrium.time_slice'][time_index]['profiles_2d'][grid]['grid_type.index'] == 1:
                eq2D = ods['equilibrium.time_slice'][time_index]['profiles_2d'][grid]
                for item in eq2D:
                    if isinstance(eq2D[item], numpy.ndarray) and len(eq2D[item].shape) == 2:
                        eq2D[item] = eq2D[item].T

                if flip_dims:
                    tmp = eq2D['grid.dim1']
                    eq2D['grid.dim1'] = ed2D['grid.dim2']
                    eq2D['grid.dim1'] = tmp
    return ods


@add_to__ODS__
@preprocess_ods('magnetics')
def magnetics_sanitize(ods, remove_bpol_probe=True):
    """
    Take data in legacy magnetics.bpol_probe and store it in current magnetics.b_field_pol_probe and magnetics.b_field_tor_probe

    :param ods: ODS to update in-place

    :return: updated ods
    """
    if 'magnetics.bpol_probe' not in ods:
        return ods

    if 'magnetics.b_field_pol_probe' in ods:
        ods['magnetics.b_field_pol_probe'].clear()
    if 'magnetics.b_field_tor_probe' in ods:
        ods['magnetics.b_field_tor_probe'].clear()

    tor_angle = ods['magnetics.bpol_probe[:].toroidal_angle']
    for k in ods['magnetics.bpol_probe']:
        if abs(numpy.sin(ods.get(f'magnetics.bpol_probe.{k}.toroidal_angle', 0.0))) < numpy.sin(numpy.pi / 4):
            ods[f'magnetics.b_field_pol_probe.+'] = ods['magnetics.bpol_probe'][k]
        else:
            ods[f'magnetics.b_field_tor_probe.+'] = ods['magnetics.bpol_probe'][k]

    if remove_bpol_probe:
        del ods['magnetics.bpol_probe']

    return ods


def delete_ggd(ods, ds=None):
    """
    delete all .ggd and .grids_ggd entries

    :param ods: input ods

    :param ds: string or list of strings where to limit the deletion process

    :return: list of strings with deleted entries
    """
    if ds is None:
        ds = ods.keys()
    elif isinstance(ds, str):
        ds = [ds]

    from .omas_structure import extract_ggd

    ggds = extract_ggd()

    deleted = []
    for ggd in ggds:
        if not any(ggd.startswith(structure + '.') for structure in ds):
            continue

        if ':' not in ggd:
            if ggd in ods:
                del ods[ggd]
                deleted.append(ggd)
        else:
            dir, base = ggd.split('[:]')
            if dir in ods:
                for k in ods[dir].keys():
                    if 'ggd' in ods[dir + '[%d]' % k]:
                        del ods[dir][k][base]
                        deleted.append(ggd)

    return deleted


def grids_ggd_points_triangles(grid):
    """
    Return points and triangles in grids_ggd structure

    :param grid: a ggd grid such as 'equilibrium.grids_ggd[0].grid[0]'

    :return: tuple with points and triangles
    """
    # objects_per_dimension: 0 = nodes, 1 = edges, 2 = faces, 3 = cells / volumes
    points = grid['space[0].objects_per_dimension[0].object[:].geometry']
    triangles = grid['space[0].objects_per_dimension[2].object[:].nodes']
    return points, triangles


class ScatterInterpolator(object):
    """
    Interface class for a unified call function for the scipy interpolators:
        - NearestNDInterpolator
        - LinearNDInterpolator
        - CloughTocher2DInterpolator
        - Rbf
    Created by factory create_scatter_interpolator
    """

    def __init__(self, interpolant, method):
        """
        Constructor

        :param interpolant: Instance of the scipy interpolator

        :param method: Interpolation method
        """
        self.interpolant = interpolant
        self.method = method

    def __call__(self, R, Z, **kwargs):
        """
        Call the interpolator

        :param interpolant: Instance of the scipy interpolator

        :param method: Interpolation method

        :param kwargs: Catch extra arguments (needed for polymorphism in equilibrium_profiles_2d_map)
        """
        if self.method == 'extrapolate':
            return numpy.reshape(self.interpolant(R.flat, Z.flat), R.shape)
        else:
            return numpy.reshape(self.interpolant(numpy.vstack((R.flat, Z.flat)).T), R.shape)


def create_scatter_interpolator(r, z, data, method=['nearest', 'linear', 'cubic', 'extrapolate'][1], return_cache=False):
    """
    Create an interpolator for scattered data points. Utility function for scatter_to_rectangular and equilibrium_profiles_2d_map

    :param r: r coordinate of data points

    :param z: z coordinate of data points

    :param data: data

    :param method: one of 'nearest', 'linear', 'cubic', 'extrapolate'

    :param sanitize: avoid NaNs in regions where data is missing

    :return: interpolator(R,Z) -> interpolated data (and cache if return_cache)
    """
    import scipy
    from scipy import interpolate

    cache = None
    if method == 'nearest':
        interpolant = scipy.interpolate.NearestNDInterpolator((r, z), data)
    elif method == 'linear':
        if cache is None:
            cache = scipy.spatial.Delaunay(numpy.vstack((r, z)).T)
        interpolant = scipy.interpolate.LinearNDInterpolator(cache, data)
    elif method == 'cubic':
        if cache is None:
            cache = scipy.spatial.Delaunay(numpy.vstack((r, z)).T)
        interpolant = scipy.interpolate.CloughTocher2DInterpolator(cache, data)
    elif method == 'extrapolate':
        if cache is None:
            cache = True
        interpolant = scipy.interpolate.Rbf(r, z, data)
    else:
        raise ValueError('Interpolation method %s is not recognized' % method)
    if return_cache:
        return ScatterInterpolator(interpolant, method), cache
    return ScatterInterpolator(interpolant, method)


def scatter_to_rectangular(r, z, data, R, Z, method=['nearest', 'linear', 'cubic', 'extrapolate'][1], sanitize=True, return_cache=False):
    """
    Interpolate scattered data points to rectangular grid

    :param r: r coordinate of data points

    :param z: z coordinate of data points

    :param data: data

    :param R: scalars, 1D arrays, or 2D arrays

    :param Z: scalars, 1D arrays, or 2D arrays

    :param method: one of 'nearest', 'linear', 'cubic', 'extrapolate'

    :param sanitize: avoid NaNs in regions where data is missing

    :param return_cache: cache object or boolean to return cache object for faster interpolaton

    :return: R, Z, interpolated_data (and cache if return_cache)
    """
    import scipy

    if isinstance(R, int) and isinstance(Z, int):
        R, Z = numpy.meshgrid(numpy.linspace(numpy.min(r), numpy.max(r), R), numpy.linspace(numpy.min(z), numpy.max(z), Z))
    elif len(numpy.atleast_1d(R).shape) == 1 and len(numpy.atleast_1d(Z).shape) == 1:
        R, Z = numpy.meshgrid(R, Z)
    elif len(numpy.atleast_1d(R).shape) == 2 and len(numpy.atleast_1d(Z).shape) == 2:
        pass
    else:
        raise ValueError('R and Z must both be either scalars, 1D arrays, or 2D arrays')

    cache = None
    if return_cache:
        scatter_interpolator, cache = create_scatter_interpolator(r, z, data, method=method, return_cache=return_cache)
    else:
        scatter_interpolator = create_scatter_interpolator(r, z, data, method=method, return_cache=return_cache)
    intepolated_data = scatter_interpolator(R, Z)
    # remove any NaNs using a rough nearest interpolation
    index = ~numpy.isnan(intepolated_data.flat)
    if sanitize and sum(1 - index):
        intepolated_data.flat[~index] = scipy.interpolate.NearestNDInterpolator(
            (R.flatten()[index], Z.flatten()[index]), intepolated_data.flatten()[index]
        )((R.flatten()[~index], Z.flatten()[~index]))

    if return_cache:
        return R[0, :], Z[:, 0], intepolated_data, cache
    return R[0, :], Z[:, 0], intepolated_data


@add_to__ODS__
def check_iter_scenario_requirements(ods):
    """
    Check that the current ODS satisfies the ITER scenario database requirements as defined in https://confluence.iter.org/x/kQqOE

    :return: list of elements that are missing to satisfy the ITER scenario requirements
    """
    from .omas_imas import iter_scenario_requirements

    fail = []
    for item in iter_scenario_requirements:
        try:
            ods[item]  # acccessing a leaf that has no data will raise an error
        except Exception:
            fail.append(item)
    return fail


@add_to__ALL__
def probe_endpoints(r0, z0, a0, l0, cocos):
    """
    Transform r,z,a,l arrays commonly used to describe poloidal magnetic
    probes geometry to actual r,z coordinates of the end-points of the probes.
    This is useful for plotting purposes.

    :param r0: r coordinates [m]

    :param z0: Z coordinates [m]

    :param a0: poloidal angles [radiants]

    :param l0: lenght [m]

    :param cocos: cocos convention

    :return: list of 2-points r and z coordinates of individual probes
    """
    theta_convention = 1
    if cocos in [1, 4, 6, 7, 11, 14, 16, 17]:
        theta_convention = -1

    boo = (1 - numpy.sign(l0)) / 2.0
    cor = boo * numpy.pi / 2.0
    # then, compute the two-point arrays to build the partial rogowskis
    # as segments rather than single points, applying the correction
    px = r0 - l0 / 2.0 * numpy.cos(theta_convention * (a0 + cor))
    py = z0 - l0 / 2.0 * numpy.sin(theta_convention * (a0 + cor))
    qx = r0 + l0 / 2.0 * numpy.cos(theta_convention * (a0 + cor))
    qy = z0 + l0 / 2.0 * numpy.sin(theta_convention * (a0 + cor))
    segx = []
    segy = []
    for k in range(len(r0)):
        segx.append([px[k], qx[k]])
        segy.append([py[k], qy[k]])

    return segx, segy


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
    fsa_B2 = omas_interp1d(rho, rho_eq, equilibrium['profiles_1d.gm5'])
    fsa_invR2 = omas_interp1d(rho, rho_eq, equilibrium['profiles_1d.gm1'])
    f = omas_interp1d(rho, rho_eq, equilibrium['profiles_1d.f'])
    dpdpsi = omas_interp1d(rho, rho_eq, equilibrium['profiles_1d.dpressure_dpsi'])

    # diamagnetic term to get included with bootstrap currrent
    JtoR_dia = dpdpsi * (1.0 - fsa_invR2 * f**2 / fsa_B2)
    JtoR_dia *= cocos['sigma_Bp'] * (2.0 * numpy.pi) ** cocos['exp_Bp']

    if JtoR is not None:
        Jout = fsa_B2 * (JtoR + includes_bootstrap * JtoR_dia) / (f * fsa_invR2)
    elif JparB is not None:
        Jout = f * fsa_invR2 * JparB / fsa_B2 - includes_bootstrap * JtoR_dia

    return Jout


@add_to__ODS__
def core_sources_j_parallel_sum(ods, time_index=0):
    """
    ods function used to sum all j_parallel contributions from core_sources (j_actuator)

    :param ods: input ods

    :param time_index: time slice to process

    :return: sum of j_parallel in [A/m^2]
    """

    rho = ods[f'core_profiles.profiles_1d.{time_index}.grid.rho_tor_norm']
    j_act = numpy.zeros(len(rho))

    for source in ods['core_sources.source']:
        if 'j_parallel' in ods[f'core_sources.source[{source}].profiles_1d.{time_index}']:
            with omas_environment(ods, coordsio={f'core_sources.source.{source}.profiles_1d.{time_index}.grid.rho_tor_norm': rho}):
                j_act += ods[f'core_sources.source[{source}].profiles_1d[{time_index}].j_parallel']

    return j_act


@add_to__ALL__
def search_ion(ion_ods, label=None, Z=None, A=None, no_matches_raise_error=True, multiple_matches_raise_error=True):
    """
    utility function used to identify the ion number and element numbers given the ion label and or their Z and/or A

    :param ion_ods: ODS location that ends with .ion

    :param label: ion label

    :param Z: ion element charge

    :param A: ion element mass

    :parame no_matches_raise_error: whether to raise a IndexError when no ion matches are found

    :parame multiple_matches_raise_error: whether to raise a IndexError when multiple ion matches are found

    :return: dictionary with matching ions labels, each with list of matching ion elements
    """
    if not ion_ods.location.endswith('.ion'):
        raise ValueError('ods location must end with `.ion`')
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
        raise IndexError('Multiple ion match query: label=%s  Z=%s  A=%s' % (label, Z, A))
    if no_matches_raise_error and len(match) == 0:
        raise IndexError('No ion match query: label=%s  Z=%s  A=%s' % (label, Z, A))
    return match


@add_to__ALL__
def search_in_array_structure(ods, conditions, no_matches_return=0, no_matches_raise_error=False, multiple_matches_raise_error=True):
    """
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
    """

    if ods.omas_data is not None and not isinstance(ods.omas_data, list):
        raise Exception('ods location must be an array of structures')

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
            raise IndexError('no matches for conditions: %s' % conditions)
        match = [no_matches_return]

    if multiple_matches_raise_error and len(match) > 1:
        raise IndexError('multiple matches for conditions: %s' % conditions)

    return match

@add_to__ALL__
def get_plot_scale_and_unit(phys_quant, species=None):
    """
    Returns normalizing scale for a physical quantity.
    E.g. "temprerature" returns 1.e-3 and keV
    :param phys_qaunt: str with a physical quantity. Uses IMAS scheme names where possible
    :return: scale, unit
    """
    if "temperature" in phys_quant:
        return 1.e-3, r"\mathrm{keV}"
    elif "density" in phys_quant :
        if species is not None and species not in ["H", "D", "He"]:
            return 1.e-18, r"\times 10^{18}\,\mathrm{m}^{-3}"
        else:
            return 1.e-19, r"\times 10^{19}\,\mathrm{m}^{-3}"
    elif "velocity" in  phys_quant:
        return 1.e-6, r"\mathrm{Mm}\,\mathrm{s}^{-1}"
    elif "e_field" in phys_quant:
        return 1.e-3, r"\mathrm{kV}\,\mathrm{m}^{-1}"
    

@add_to__ALL__
def define_cocos(cocos_ind):
    """
    Returns dictionary with COCOS coefficients given a COCOS index

    https://docs.google.com/document/d/1-efimTbI55SjxL_yE_GKSmV4GEvdzai7mAj5UYLLUXw/edit

    :param cocos_ind: COCOS index

    :return: dictionary with COCOS coefficients
    """

    cocos = dict.fromkeys(['sigma_Bp', 'sigma_RpZ', 'sigma_rhotp', 'sign_q_pos', 'sign_pprime_pos', 'exp_Bp'])

    # all multipliers shouldn't change input values if cocos_ind is None
    if cocos_ind is None:
        cocos['exp_Bp'] = 0
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = 0
        cocos['sign_pprime_pos'] = 0
        return cocos

    # if COCOS>=10, this should be 1
    cocos['exp_Bp'] = 0
    if cocos_ind >= 10:
        cocos['exp_Bp'] = +1

    if cocos_ind in [1, 11]:
        # These cocos are for
        # (1)  psitbx(various options), Toray-GA
        # (11) ITER, Boozer
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [2, 12, -12]:
        # These cocos are for
        # (2)  CHEASE, ONETWO, HintonHazeltine, LION, XTOR, MEUDAS, MARS, MARS-F
        # (12) GENE
        # (-12) ASTRA
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [3, 13]:
        # These cocos are for
        # (3) Freidberg*, CAXE and KINX*, GRAY, CQL3D^, CarMa, EFIT* with : ORB5, GBSwith : GT5D
        # (13)  CLISTE, EQUAL, GEC, HELENA, EU ITM-TF up to end of 2011
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [4, 14]:
        # These cocos are for
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [5, 15]:
        # These cocos are for
        # (5) TORBEAM, GENRAY^
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [6, 16]:
        # These cocos are for
        cocos['sigma_Bp'] = +1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = -1
        cocos['sign_q_pos'] = -1
        cocos['sign_pprime_pos'] = -1

    elif cocos_ind in [7, 17]:
        # These cocos are for
        # (17) LIUQE*, psitbx(TCV standard output)
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = +1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
        cocos['sign_pprime_pos'] = +1

    elif cocos_ind in [8, 18]:
        # These cocos are for
        cocos['sigma_Bp'] = -1
        cocos['sigma_RpZ'] = -1
        cocos['sigma_rhotp'] = +1
        cocos['sign_q_pos'] = +1
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
        printd("No COCOS tranformation for " + str(cocosin_index) + " to " + str(cocosout_index), topic='cocos')
        sigma_Ip_eff = 1
        sigma_B0_eff = 1
        sigma_Bp_eff = 1
        exp_Bp_eff = 0
        sigma_rhotp_eff = 1
    else:
        printd("COCOS tranformation from " + str(cocosin_index) + " to " + str(cocosout_index), topic='cocos')
        cocosin = define_cocos(cocosin_index)
        cocosout = define_cocos(cocosout_index)

        sigma_Ip_eff = cocosin['sigma_RpZ'] * cocosout['sigma_RpZ']
        sigma_B0_eff = cocosin['sigma_RpZ'] * cocosout['sigma_RpZ']
        sigma_Bp_eff = cocosin['sigma_Bp'] * cocosout['sigma_Bp']
        exp_Bp_eff = cocosout['exp_Bp'] - cocosin['exp_Bp']
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
def identify_cocos(B0, Ip, q, psi, clockwise_phi=None, a=None):
    """
    Utility function to identify COCOS coordinate system
    If multiple COCOS are possible, then all are returned.

    :param B0: toroidal magnetic field (with sign)

    :param Ip: plasma current (with sign)

    :param q: safety factor profile (with sign) as function of psi

    :param psi: poloidal flux as function of psi(with sign)

    :param clockwise_phi: (optional) [True, False] if phi angle is defined clockwise or not
                          This is required to identify odd Vs even COCOS
                          Note that this cannot be determined from the output of a code.
                          An easy way to determine this is to answer the question: is positive B0 clockwise?

    :param a: (optional) flux surfaces minor radius as function of psi
              This is required to identify 2*pi term in psi definition

    :return: list with possible COCOS
    """

    if clockwise_phi is None:
        sigma_rpz = clockwise_phi
    elif clockwise_phi:
        sigma_rpz = -1
    else:
        sigma_rpz = +1

    # return both even and odd COCOS if clockwise_phi is not provided
    if sigma_rpz is None:
        tmp = identify_cocos(B0, Ip, q, psi, True, a)
        tmp.extend(identify_cocos(B0, Ip, q, psi, False, a))
        return tmp

    sigma_Ip = numpy.sign(Ip)
    sigma_B0 = numpy.sign(B0)
    sign_dpsi_pos = numpy.sign(numpy.gradient(psi))[0]
    sign_q_pos = numpy.sign(q)[0]

    sigma_Bp = sign_dpsi_pos / sigma_Ip
    sigma_rhotp = sign_q_pos / (sigma_Ip * sigma_B0)

    sigma2cocos = {
        (+1, +1, +1): 1,  # +Bp, +rpz, +rtp
        (+1, -1, +1): 2,  # +Bp, -rpz, +rtp
        (-1, +1, -1): 3,  # -Bp, +rpz, -rtp
        (-1, -1, -1): 4,  # -Bp, -rpz, -rtp
        (+1, +1, -1): 5,  # +Bp, +rpz, -rtp
        (+1, -1, -1): 6,  # +Bp, -rpz, -rtp
        (-1, +1, +1): 7,  # -Bp, +rpz, +rtp
        (-1, -1, +1): 8,  # -Bp, -rpz, +rtp
    }

    # identify 2*pi term in psi definition based on q estimate
    if a is not None:
        index = numpy.argmin(numpy.abs(q))
        if index == 0:
            index += 1
        q_estimate = abs((numpy.pi * B0 * (a[index] - a[0]) ** 2) / (psi[index] - psi[0]))
        q_actual = abs(q[index])
        if abs(q_estimate - q_actual) < abs(q_estimate / (2 * numpy.pi) - q_actual):
            eBp = 1
        else:
            eBp = 0

        return [sigma2cocos[(sigma_Bp, sigma_rpz, sigma_rhotp)] + 10 * eBp]

    # return COCOS<10 as well as COCOS>10 if a is not provided
    else:
        return [sigma2cocos[(sigma_Bp, sigma_rpz, sigma_rhotp)], sigma2cocos[(sigma_Bp, sigma_rpz, sigma_rhotp)] + 10]


@add_to__ALL__
@contextmanager
def omas_environment(
    ods,
    cocosio=None,
    coordsio=None,
    unitsio=None,
    uncertainio=None,
    input_data_process_functions=None,
    xmlcodeparams=False,
    dynamic_path_creation=None,
    **kw,
):
    """
    Provides environment for data input/output to/from OMAS

    :param ods: ODS on which to operate

    :param cocosio: COCOS convention

    :param coordsio: dictionary/ODS with coordinates for data interpolation

    :param unitsio: True/False whether data read from OMAS should have units

    :param uncertainio: True/False whether data read from OMAS should have uncertainties

    :param input_data_process_functions: list of functions that are used to process data that is passed to the ODS

    :param xmlcodeparams: view code.parameters as an XML string while in this environment

    :param dynamic_path_creation: whether to dynamically create the path when setting an item
                                  * False: raise an error when trying to access a structure element that does not exists
                                  * True (default): arrays of structures can be incrementally extended by accessing at the next element in the array
                                  * 'dynamic_array_structures': arrays of structures can be dynamically extended

    :param kw: extra keywords set attributes of the ods (eg. 'consistency_check', 'dynamic_path_creation', 'imas_version')

    :return: ODS with environment set
    """

    # turn simple coordsio dictionary into an ODS
    if isinstance(coordsio, dict):
        from omas import ODS

        tmp = ODS(cocos=ods.cocos)
        with omas_environment(tmp, cocosio=cocosio, dynamic_path_creation='dynamic_array_structures'):
            tmp.update(coordsio)
        coordsio = tmp

    if cocosio is not None and not isinstance(cocosio, int):
        raise ValueError('cocosio can only be an integer')

    # backup attributes
    bkp_dynamic_path_creation = omas_rcparams['dynamic_path_creation']
    bkp_cocosio = ods.cocosio
    bkp_coordsio = ods.coordsio
    bkp_unitsio = ods.unitsio
    bkp_uncertainio = ods.uncertainio
    bkp_args = {}
    for item in kw:
        bkp_args[item] = getattr(ods, item)

    # set attributes
    for item in kw:
        setattr(ods, item, kw[item])
    if cocosio is not None:
        ods.cocosio = cocosio
    if coordsio is not None:
        ods.coordsio = coordsio
    if unitsio is not None:
        ods.unitsio = unitsio
    if uncertainio is not None:
        ods.uncertainio = uncertainio
    if dynamic_path_creation is not None:
        omas_rcparams['dynamic_path_creation'] = dynamic_path_creation

    # set input_data_process_functions
    if input_data_process_functions is not None:
        from . import omas_core

        bkp_input_data_process_functions = copy.copy(omas_core.input_data_process_functions)
        omas_core.input_data_process_functions[:] = input_data_process_functions

    # set code.parameters as XML string
    if xmlcodeparams:
        ods.codeparams2xml()

    try:
        if coordsio is not None:
            with omas_environment(coordsio, cocosio=cocosio):
                yield ods
        else:
            yield ods

    finally:
        # restore code.parameters as dictionary
        if xmlcodeparams:
            ods.codeparams2dict()
        # restore attributes
        omas_rcparams['dynamic_path_creation'] = bkp_dynamic_path_creation
        ods.cocosio = bkp_cocosio
        ods.coordsio = bkp_coordsio
        ods.unitsio = bkp_unitsio
        ods.uncertainio = bkp_uncertainio
        for item in kw:
            try:
                setattr(ods, item, bkp_args[item])
            except Exception as _excp:
                # Add more user feedback, since use of consistency_check in an omas_environment can be confusing
                if item == 'consistency_check':
                    raise _excp.__class__(str(_excp) + '\nThe IMAS consistency was violated getting out of the omas_environment')
                raise
        # restore input_data_process_functions
        if input_data_process_functions is not None:
            omas_core.input_data_process_functions[:] = bkp_input_data_process_functions


def generate_cocos_signals(structures=[], threshold=0, write=True, verbose=False):
    """
    This is a utility function for generating the omas_cocos.py Python file

    :param structures: list of structures for which to generate COCOS signals

    :param threshold: score threshold below which singals entries will not be written in omas_cocos.py
    * 0 is a reasonable threshold for catching signals that should have an associated COCOS transform
    * 10000 (or any high number) is a way to hide signals in omas_cocos.py that are unassigned

    :param write: update omas_cocos.py file

    :param verbose: print cocos signals to screen

    :return: dictionary structure with tally of score and reason for scoring for every entry
    """
    # cocos_signals contains the IMAS locations and the corresponding `cocos_transform` function
    from .omas_cocos import _cocos_signals

    # update OMAS cocos information with the one stored in IMAS
    from .omas_structure import extract_cocos

    _cocos_signals.update(extract_cocos())

    # units of entries currently in cocos_singals
    cocos_units = []
    for item in _cocos_signals:
        if _cocos_signals[item] == '?':
            continue
        info = omas_info_node(item)
        if len(info) and 'units' in info:  # info may have no length if nodes are deleted between IMAS versions
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

    if isinstance(structures, str):
        structures = [structures]
    structures += cocos_structures
    structures = numpy.unique(structures)

    from .omas_utils import _structures, _extra_structures
    from .omas_utils import i2o
    from .omas_core import ODS

    # if generate_cocos_signals is run after omas has been used for something else
    # (eg. when running test_examples) it may be that _extra_structures is not empty
    # Thus, we clear _structures and _extra_structures to make sure that
    # structures_filenames() is not polluted by _extra_structures
    _structures_bkp = copy.deepcopy(_structures)
    _extra_structures_bkp = copy.deepcopy(_extra_structures)
    try:
        _structures.clear()
        _extra_structures.clear()

        ods = ODS()
        out = {}
        text = []
        csig = [
            """
'''List of automatic COCOS transformations

-------
'''
# COCOS signals candidates are generated by running utilities/generate_cocos_signals.py
# Running this script is useful to keep track of new signals that IMAS adds in new data structure releases
#
# In this file you are only allowed to edit/add entries to the `_cocos_signals` dictionary
# The comments indicate `#[ADD_OR_DELETE_SUGGESTION]# MATCHING_SCORE # RATIONALE_FOR_ADD_OR_DELETE`
#
# Proceed as follows:
# 1. Edit transformations in this file (if a signal is missing, it can be added here)
# 2. Run `utilities/generate_cocos_signals.py` (which will update this same file)
# 3. Commit changes
#
# Valid transformations are defined in the `cocos_transform()` function and they are:
#
#        transforms = {}
#        transforms['1/PSI'] = sigma_Ip_eff * sigma_Bp_eff / (2 * numpy.pi) ** exp_Bp_eff
#        transforms['invPSI'] = transforms['1/PSI']
#        transforms['dPSI'] = transforms['1/PSI']
#        transforms['F_FPRIME'] = transforms['dPSI']
#        transforms['PPRIME'] = transforms['dPSI']
#        transforms['PSI'] = sigma_Ip_eff * sigma_Bp_eff * (2 * numpy.pi) ** exp_Bp_eff
#        transforms['Q'] = sigma_Ip_eff * sigma_B0_eff * sigma_rhotp_eff
#        transforms['TOR'] = sigma_B0_eff
#        transforms['BT'] = transforms['TOR']
#        transforms['IP'] = transforms['TOR']
#        transforms['F'] = transforms['TOR']
#        transforms['POL'] = sigma_B0_eff * sigma_rhotp_eff
#        transforms['BP'] = transforms['POL']
#        transforms[None] = 1
#

_cocos_signals = {}
"""
        ]

        skip_signals = [
            'chi_squared',
            'standard_deviation',
            'weight',
            'coefficients',
            'beta_tor',
            'beta_pol',
            'radial',
            'rho_tor_norm',
            'darea_drho_tor',
            'dvolume_drho_tor',
            'ratio',
            'fraction',
            'rate',
            'd',
            'flux',
            'v',
            'b_field_max',
            'b_field_r',
            'b_field_z',
            'b_r',
            'b_z',
            'width_tor',
        ]

        # loop over structures
        for structure in structures:
            print('Updating COCOS info for: ' + structure)
            text.extend(['', '# ' + structure.upper()])
            csig.extend(['', '# ' + structure.upper()])

            out[structure] = {}
            ods[structure]
            m = 0
            # generate score and add reason for scoring
            for item in sorted(list(load_structure(structure, omas_rcparams['default_imas_version'])[0].keys())):
                item = i2o(item)
                item_ = item
                if any(item.endswith(k) for k in [':.values', ':.value', ':.data']):
                    item_ = l2o(p2l(item)[:-2])
                elif any(item.endswith(k) for k in ['.values', '.value', '.data']):
                    item_ = l2o(p2l(item)[:-1])
                m = max(m, len(item))
                score = 0
                rationale = []
                if item.startswith(structure) and '_error_' not in item:
                    entry = "_cocos_signals['%s']=" % i2o(item)
                    info = omas_info_node(item)
                    units = info.get('units', None)
                    data_type = info.get('data_type', None)
                    documentation = info.get('documentation', '')
                    if data_type in ['STRUCTURE', 'STR_0D', 'STRUCT_ARRAY']:
                        continue
                    elif units in [None, 's']:
                        out[structure].setdefault(-1, []).append((item, '[%s]' % units))
                        continue
                    elif re.match('.*\.[rz]$', item):
                        continue
                    elif any((item_.endswith('.' + k) or item_.endswith('_' + k) or '.' + k + '.' in item) for k in skip_signals):
                        out[structure].setdefault(-1, []).append((item, p2l(item_)[-1]))
                        continue
                    elif any(k in documentation for k in ['always positive']):
                        out[structure].setdefault(-1, []).append((item, documentation))
                        continue
                    n = item.count('.')
                    for pnt, key in enumerate(p2l(item)):
                        pnt = pnt / n
                        for case in ['q', 'ip', 'b0', 'phi', 'psi', 'f', 'f_df']:
                            if key == case:
                                rationale += [case]
                                score += pnt
                                break
                        for case in ['q', 'j', 'phi', 'psi', 'ip', 'b', 'f', 'v', 'f_df']:
                            if key.startswith('%s_' % case) and not any(key.startswith(k) for k in ['psi_norm']):
                                rationale += [case]
                                score += pnt
                                break
                        for case in ['velocity', 'current', 'b_field', 'e_field', 'torque', 'momentum']:
                            if case in key and key not in ['heating_current_drive']:
                                rationale += [case]
                                score += pnt
                                break
                        for case in ['_dpsi']:
                            if case in key and case + '_norm' not in key:
                                rationale += [case]
                                score += pnt
                                break
                        for case in ['poloidal', 'toroidal', 'parallel', '_tor', '_pol', '_par', 'tor_', 'pol_', 'par_']:
                            if (key.endswith(case) or key.startswith(case)) and not any(
                                [key.startswith(k) for k in ['conductivity_', 'pressure_', 'rho_', 'length_']]
                            ):
                                rationale += [case]
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
                    if _cocos_signals.get(item, '?') == '?':
                        if score > 0:
                            message = '#[ADD?]'
                        else:
                            message = '#[DEL?]'
                    elif score < 0:
                        message = '#[DEL?]'

                    transform = _cocos_signals.get(item, '?')
                    if isinstance(transform, str):
                        transform = repr(transform)
                    txt = ("_cocos_signals['%s']=%s" % (item, transform)).ljust(m + 20) + message + '# %f # %s' % (score, rationale)
                    text.append(txt)
                    if score > threshold or (item in _cocos_signals and _cocos_signals[item] != '?'):
                        csig.append(txt)

            # print to screen (note that this prints ALL the entries, whereas omas_cocos.py only contains entries that score above a give threshold)
            if verbose:
                print('\n'.join(text) + '\n\n' + '-' * 20)

        filename = os.path.abspath(str(os.path.dirname(__file__)) + '/omas_cocos.py')
        if write:
            # update omas_cocos.py
            with open(filename, 'w') as f:
                f.write(str('\n'.join(csig)))
        else:
            # check that omas_cocos.py file is up-to-date
            with open(filename, 'r') as f:
                original = str(f.read())
            new = str('\n'.join(csig))
            diff = difflib.unified_diff(original, new)
            assert original == new, 'COCOS signals are not up-to-date! Run `make cocos` to update the omas_cocos.py file.\n' + ''.join(diff)

        return out

    finally:
        _structures.clear()
        _structures.update(_structures_bkp)
        _extra_structures.clear()
        _extra_structures.update(_extra_structures_bkp)


# The CocosSignals class is just a dictionary that raises warnings when users access
# entries that are likely to need a COCOS transformation, but do not have one.
class CocosSignals(dict):
    def __init__(self):
        self.reload()

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if not isinstance(value, list) and value == '?':
            warnings.warn(
                f'''
`{key}` may require defining its COCOS transform in {os.path.split(__file__)[0] + os.sep}omas_cocos.py
Once done, you can reload the cocos definitions with:
> from omas.omas_physics import cocos_signals; cocos_signals.reload()
'''.strip()
            )
        return value

    def reload(self):
        namespace = {}
        with open(os.path.split(__file__)[0] + os.sep + 'omas_cocos.py', 'r') as f:
            exec(f.read(), namespace)
        self.clear()
        self.update(namespace['_cocos_signals'])


# cocos_signals is the actual dictionary
cocos_signals = CocosSignals()
