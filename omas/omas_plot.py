'''plotting ODS methods and utilities

-------
'''

from .omas_utils import *
from .omas_physics import cocos_transform
from .omas_symbols import latexit

__all__ = []
__ods__ = []


def add_to__ODS__(f):
    """
    anything wrapped here will be available as a ODS method with name 'plot_'+f.__name__
    """
    __ods__.append(f.__name__)
    return f


def add_to__ALL__(f):
    __all__.append(f.__name__)
    return f


# ================================
# plotting helper functions
# ================================


def uerrorbar(x, y, ax=None, **kwargs):
    r"""
    Given arguments y or x,y where x and/or y have uncertainties, feed the
    appropriate terms to matplotlib's errorbar function.

    If y or x is more than 1D, it is flattened along every dimension but the last.

    :param x: array of independent axis values

    :param y: array of values with uncertainties, for which shaded error band is plotted

    :param ax: The axes instance into which to plot (default: gca())

    :param \**kwargs: Passed to ax.errorbar

    :return: list. A list of ErrorbarContainer objects containing the line, bars, and caps of each (x,y) along the last dimension.
    """
    result = []

    # set default key word arguments
    if ax is None:
        from matplotlib import pyplot

        ax = pyplot.gca()
    kwargs.setdefault('marker', 'o')
    if 'linestyle' not in kwargs and 'ls' not in kwargs:
        kwargs['linestyle'] = ''
    if numpy.all(std_devs(y) == 0) and numpy.all(std_devs(x) == 0):
        kwargs.setdefault('capsize', 0)

    # enable combinations of 1D and 2D x's and y's
    y = numpy.array(y)
    y = y.reshape(-1, y.shape[-1])
    x = numpy.array(x)
    x = x.reshape(-1, x.shape[-1])
    if x.shape[0] == 1 and y.shape[0] > 1:  # one x for all y's
        x = numpy.tile(x[0, :], y.shape[0]).reshape(-1, x.shape[-1])

    # plot each (x,y) and collect container objects
    for xi, yi in zip(x, y):
        tmp = ax.errorbar(nominal_values(xi), nominal_values(yi), xerr=std_devs(xi), yerr=std_devs(yi), **kwargs)
        result.append(tmp)

    return result


class Uband(object):
    """
    This class wraps the line and PollyCollection(s) associated with a banded
    errorbar plot for use in the uband function.

    """

    def __init__(self, line, bands):
        """
        :param line: Line2D
            A line of the x,y nominal values

        :param bands: list of PolyCollections
            The fill_between and/or fill_betweenx PollyCollections spanning the std_devs of the x,y data

        """
        from matplotlib.cbook import flatten

        self.line = line  # matplotlib.lines.Line2D
        self.bands = list(flatten([bands]))  # matplotlib.collections.PolyCollection(s)

    def __getattr__(self, attr):
        if attr in ['set_color', 'set_lw', 'set_linewidth', 'set_dashes', 'set_linestyle']:

            def _band_line_method(self, method, *args, **kw):
                """
                Call the same method for line and band.
                Returns Line2D method call result.
                """
                for band in self.bands:
                    getattr(band, method)(*args, **kw)
                return getattr(self.line, method)(*args, **kw)

            return lambda method=attr, *args, **kw: _band_line_method(method, *args, **kw)
        else:
            return getattr(self.line, attr)


def uband(x, y, ax=None, fill_kw={'alpha': 0.25}, **kw):
    r"""
    Given arguments x,y where either or both have uncertainties, plot x,y using pyplt.plot
    of the nominal values and surround it with with a shaded error band using matplotlib's
    fill_between and/or fill_betweenx.

    If y or x is more than 1D, it is flattened along every dimension but the last.

    :param x: array of independent axis values

    :param y: array of values with uncertainties, for which shaded error band is plotted

    :param ax: axes instance into which to plot (default: gca())

    :param fill_kw: dict. Passed to pyplot.fill_between

    :param \**kw: Passed to pyplot.plot

    :return: list. A list of Uband objects containing the line and bands of each (x,y) along the last dimension.

    """

    from matplotlib import pyplot

    result = []
    if ax is None:
        ax = pyplot.gca()

    # enable combinations of 1D and 2D x's and y's
    y = numpy.array(y)
    y = y.reshape(-1, y.shape[-1])
    x = numpy.array(x)
    x = x.reshape(-1, x.shape[-1])
    if x.shape[0] == 1 and y.shape[0] > 1:  # one x for all y's
        x = numpy.tile(x[0, :], y.shape[0]).reshape(-1, x.shape[-1])

    # plot each (x,y) and collect the lines/bands into a single object
    for xi, yi in zip(x, y):
        xnom = numpy.atleast_1d(numpy.squeeze(nominal_values(xi)))
        xerr = numpy.atleast_1d(numpy.squeeze(std_devs(xi)))
        ynom = numpy.atleast_1d(numpy.squeeze(nominal_values(yi)))
        yerr = numpy.atleast_1d(numpy.squeeze(std_devs(yi)))

        (l,) = ax.plot(xnom, ynom, **kw)

        fkw = copy.copy(fill_kw)  # changes to fill_kw propagate to the next call of uband!
        fkw.setdefault('color', l.get_color())
        bands = []
        if numpy.any(yerr != 0):
            bandy = ax.fill_between(xnom, ynom - yerr, ynom + yerr, **fkw)
            bands.append(bandy)
        if numpy.any(xerr != 0):
            bandx = ax.fill_betweenx(ynom, xnom - xerr, xnom + xerr, **fkw)
            bands.append(bandx)

        tmp = Uband(l, bands)
        result.append(tmp)

    return result


def imas_units_to_latex(unit):
    """
    converts units to a nice latex format for plot labels

    :param unit: string with unit in imas format

    :return: string with unit in latex format
    """
    unit = re.sub('(\-?[0-9]+)', r'{\1}', unit)
    unit = re.sub('\.', r'\,', unit)
    return f' [${unit}$]'


@add_to__ALL__
def get_channel_count(ods, hw_sys, check_loc=None, test_checker=None, channels_name='channel'):
    """
    Utility function for CX hardware overlays.
    Gets a channel count for some hardware systems.
    Provide check_loc to make sure some data exist.

    :param ods: OMAS ODS instance

    :param hw_sys: string
        Hardware system to check. Must be a valid top level IDS name, like 'thomson_scattering'

    :param check_loc: [optional] string
        If provided, an additional check will be made to ensure that some data exist.
        If this check fails, channel count will be set to 0
        Example: 'thomson_scattering.channel.0.position.r'

    :param test_checker: [optional] string to evaluate into bool
        Like "checker > 0", where checker = ods[check_loc]. If this test fails, nc will be set to 0

    :param channels_name: string
        Use if you need to generalize to something that doesn't have real channels but has something analogous,
        like how 'gas_injection' has 'pipe' that's shaped like 'channel' is in 'thomson_scattering'.

    :return: Number of channels for this hardware system. 0 indicates empty.
    """
    try:
        nc = len(ods[hw_sys][channels_name])
        if check_loc is not None:
            checker = ods[check_loc]
            if test_checker is not None:
                assert eval(test_checker)
    except (TypeError, AssertionError, ValueError, IndexError, KeyError):
        nc = 0

    if nc == 0:
        printd('{} overlay could not find sufficient data to make a plot'.format(hw_sys))
    return nc


def gas_filter(label, which_gas):
    """
    Utility: processes the mask / which_gas selector for gas_injection_overlay
    :param label: string
        Label for a gas valve to be tested

    :param which_gas: string or list
        See gas_injection_overlay docstring

    :return: bool
        Flag indicating whether or not a valve with this label should be shown
    """
    include = False
    if isinstance(which_gas, str):
        if which_gas == 'all':
            include = True
    elif isinstance(which_gas, list):
        include = any(wg in label for wg in which_gas)
    return include


def gas_arrow(ods, r, z, direction=None, r2=None, z2=None, snap_to=numpy.pi / 4.0, ax=None, color=None, pad=1.0, **kw):
    """
    Draws an arrow pointing in from the gas valve
    :param ods: ODS instance

    :param r: float
        R position of gas injector (m)

    :param z: float
        Z position of gas injector (m)

    :param r2: float [optional]
        R coordinate of second point, at which the gas injector is aiming inside the vessel

    :param z2: float [optional]
        Z coordinate of second point, at which the gas injector is aiming inside the vessel

    :param direction: float
        Direction of injection (radians, COCOS should match ods.cocos). None = try to guess.

    :param snap_to: float
        Snap direction angle to nearest value. Set snap to pi/4 to snap to 0, pi/4, pi/2, 3pi/4, etc. No in-between.

    :param ax: axes instance into which to plot (default: gca())

    :param color: matplotlib color specification

    :param pad: float
        Padding between arrow tip and specified (r,z)
    """

    from matplotlib import pyplot

    def pick_direction():
        """Guesses the direction for the arrow (from injector toward machine) in case you don't know"""
        dr = ods['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['r'] - r
        dz = ods['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['z'] - z
        theta = numpy.arctan2(dz, -dr)
        if snap_to > 0:
            theta = snap_to * round(theta / snap_to)
        return theta

    if (r2 is not None) and (z2 is not None):
        direction = numpy.arctan2(z2 - z, r - r2)
    elif direction is None:
        direction = pick_direction()
    else:
        direction = cocos_transform(ods.cocos, 11)['BP'] * direction

    if ax is None:
        ax = pyplot.gca()

    shaft_len = 3.5 * (1 + pad) / 2.0

    da = numpy.pi / 10  # Angular half width of the arrow head
    x0 = numpy.cos(-direction) * pad
    y0 = numpy.sin(-direction) * pad
    head_mark = [
        (x0, y0),
        (x0 + numpy.cos(-direction + da), y0 + numpy.sin(-direction + da)),
        (x0 + numpy.cos(-direction), y0 + numpy.sin(-direction)),
        (x0 + shaft_len * numpy.cos(-direction), y0 + shaft_len * numpy.sin(-direction)),
        (x0 + numpy.cos(-direction), y0 + numpy.sin(-direction)),
        (x0 + numpy.cos(-direction - da), y0 + numpy.sin(-direction - da)),
    ]

    kw.pop('marker', None)  # Ignore this
    return ax.plot(r, z, marker=head_mark, color=color, markersize=100 * (pad + shaft_len) / 5, **kw)


def geo_type_lookup(geometry_type, subsys, imas_version=omas_rcparams['default_imas_version'], reverse=False):
    """
    Given a geometry type code

    :param geometry_type: int (or string if reverse=True)
        Geometry type code (or geometry name if reverse)

    :param subsys: string
        Name of subsystem or ODS, like 'pf_active'

    :param imas_version: string
        IMAS version to use when mapping

    :param reverse: bool
        Switches the roles of param geometry_type and return

    :return: string (or int if reverse=True)
        Name of the field indicated by geometry_type (or type code if reverse=True).
        For example: In IMAS 3.19.0, `pf_active.coil[:].element[:].geometry.geometry_type = 1` means 'outline'.
        In version 3.19.0 the following geometry types exist {1: 'outline', 2: 'rectangle', 4: 'arcs of circle'}
    """

    # Fetch information from IMAS data description of geometry_type for the relevant subsys
    lookup = {
        'ic_antennas': 'ic_antennas.antenna.:.strap.:.geometry.geometry_type',
        'pf_active': 'pf_active.coil.:.element.:.geometry.geometry_type',
    }
    if subsys not in lookup.keys():
        printe('Warning: unrecognized IMAS substructure ({})'.format(subsys))
        return None

    try:
        doc = omas_info_node(lookup[subsys], imas_version=imas_version)['documentation']
    except ValueError as _excp:
        printe(repr(_excp))
        return None

    geo_map = eval('{%s}' % doc.split('(')[-1][:-2])
    if 3 not in geo_map:
        geo_map[3] = 'oblique'  # For backward compatibility

    if reverse:
        # https://stackoverflow.com/a/13149770/6605826
        return list(geo_map.keys())[list(geo_map.values()).index(geometry_type)]
    else:
        return geo_map.get(geometry_type, None)


def padded_extension(values_in, n, fill_value):
    """
    Forces values_in to be at least length n by appending copies of fill_value as needed
    :param values_in: scalar or 1D iterable

    :param n: int

    :param fill_value: scalar

    :return: 1D array with length >= n
    """
    x = numpy.atleast_1d(values_in).tolist()
    if len(x) < n:
        x += [fill_value] * (n - len(x))
    return numpy.array(x)


def text_alignment_setup(n, default_ha='left', default_va='baseline', **kw):
    """
    Interprets text alignment instructions
    :param n: int
        Number of labels that need alignment instructions

    :param default_ha: string or list of n strings
        Default horizontal alignment. If one is supplied, it will be copied n times.

    :param default_va: string or list of n strings
        Default vertical alignment. If one is supplied, it will be copied n times.

    :param kw: keywords caught by overlay method

    :return: (list of n strings, list of n strings, kw)
        Horizontal alignment instructions
        Vertical alignment instructions
        Updated keywords
    """
    label_ha = padded_extension(kw.pop('label_ha', None), n, None)
    label_va = padded_extension(kw.pop('label_va', None), n, None)

    default_ha = numpy.atleast_1d(default_ha).tolist()
    default_va = numpy.atleast_1d(default_va).tolist()
    if len(default_ha) == 1:
        default_ha *= n
    if len(default_va) == 1:
        default_va *= n

    for i in range(n):
        label_ha[i] = default_ha[i] if label_ha[i] is None else label_ha[i]
        label_va[i] = default_va[i] if label_va[i] is None else label_va[i]

    return label_ha, label_va, kw


def label_shifter(n, kw):
    """
    Interprets label shift instructions

    :param n: int
        Number of labels that need shift instructions

    :param kw: dict
        Keywords passed to main plot script; label shifting keywords will be removed

    :return: (1D array with length >= n, 1D array with length >= n)
        R shifts
        Z shifts
    """
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)
    label_dr = padded_extension(label_dr, n, fill_value=label_dr if numpy.isscalar(label_dr) else 0)
    label_dz = padded_extension(label_dz, n, fill_value=label_dz if numpy.isscalar(label_dz) else 0)
    return label_dr, label_dz


# hold last 100 references of matplotlib.widgets.Slider
_stimes = []


def ods_time_plot(ods_plot_function, ods, time_index, time, **kw):
    r"""
    Utility function for generating time dependent plots

    :param ods_plot_function: ods plot function to be called
    this function must accept ax (either a single or a list of axes)
    and must return the axes (or list of axes) it used

    :param ods: ods

    :param time_index: time indexes to be scanned

    :param time: array of times

    :param \**kw: extra aruments to passed to ods_plot_function

    :return: slider instance and list of axes used
    """
    from matplotlib import pyplot
    from matplotlib.widgets import Slider

    time_index = numpy.atleast_1d(time_index)
    time = time[time_index]
    axs = {}

    def do_clean(time0):
        if axs is not None:
            for ax in axs:
                if axs[ax] is not None:
                    axs[ax].cla()

    def update(time0):
        if 'ax' in kw:
            ax = kw.pop('ax')
        elif not len(axs):
            ax = None
        elif len(axs) == 1:
            ax = list(axs.values())[0]
        else:
            ax = axs
        time_index0 = time_index[numpy.argmin(abs(time - time0))]
        tmp = ods_plot_function(ods, time_index0, ax=ax, **kw)['ax']
        if isinstance(tmp, dict):
            axs.update(tmp)
        else:
            axs[1, 1, 1] = tmp

    stime, axtime = kw.pop('stime', (None, None))

    update(time[0])

    if stime is None:
        axtime = pyplot.axes([0.1, 0.96, 0.75, 0.03])
        min_time = min(time)
        max_time = max(time)
        if min_time == max_time:
            min_time = min_time - 1
            max_time = max_time + 1

        stime = Slider(axtime, 'Time[s]', min_time, max_time, valinit=min(time), valstep=min(numpy.diff(time)))
        if stime not in _stimes:
            _stimes.append(stime)
            if len(_stimes) > 100:
                _stimes.pop(0)
        stime.on_changed(do_clean)
    stime.on_changed(update)
    for time0 in time:
        axtime.axvline(time0, color=['r', 'y', 'c', 'm'][stime.cnt - 2])
    return {'stime': (stime, axtime), 'ax': axs}


def cached_add_subplot(fig, ax_cache, *args, **kw):
    r"""
    Utility function that works like matplotlib add_subplot
    but reuses axes if these were already used before

    :param fig: matplotlib figure

    :param ax_cache: caching dictionary

    :param \*args: arguments passed to matplotlib add_subplot

    :param \**kw: keywords arguments passed to matplotlib add_subplot

    :return: matplotlib axes
    """
    if args in ax_cache:
        return ax_cache[args]
    else:
        ax = fig.add_subplot(*args, **kw)
        ax_cache[args] = ax
        return ax


# ================================
# ODSs' plotting methods
# ================================
def handle_time(ods, time_location, time_index, time):
    '''
    Given either time_index or time returns both time_index and time consistent with one another
    NOTE: time takes precedence over time_index

    :param time_location: location of which to get the time

    :param time_index: int or list of ints

    :param time: float or list of floats

    :return: time_index, time
    '''
    if time is not None:
        tds = ods.time(time_location)
        time_index = []
        for t in numpy.atleast_1d(time):
            time_index.append(numpy.argmin(abs(tds - t)))
    if time_index is None:
        time = ods.time(time_location)
        if time is None:
            time_index = 0
        else:
            time_index = numpy.arange(len(time))
    return time_index, numpy.atleast_1d(time)


@add_to__ODS__
def equilibrium_CX(
    ods,
    time_index=None,
    time=None,
    levels=None,
    contour_quantity='rho_tor_norm',
    allow_fallback=True,
    ax=None,
    sf=3,
    label_contours=None,
    show_wall=True,
    xkw={},
    ggd_points_triangles=None,
    **kw,
):
    r"""
    Plot equilibrium cross-section
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: ODS instance
        input ods containing equilibrium data

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param levels: sorted numeric iterable
        values to pass to 2D plot as contour levels

    :param contour_quantity: string
        quantity to contour, anything in eq['profiles_1d'] or eq['profiles_2d'] or psi_norm

    :param allow_fallback: bool
        If rho/phi is requested but not available, plot on psi instead if allowed. Otherwise, raise ValueError.

    :param ax: Axes instance
        axes to plot in (active axes is generated if `ax is None`)

    :param sf: int
        Resample scaling factor. For example, set to 3 to resample to 3x higher resolution. Makes contours smoother.

    :param label_contours: bool or None
        True/False: do(n't) label contours
        None: only label if contours are of q

    :param show_wall: bool
        Plot the inner wall or limiting surface, if available

    :param xkw: dict
        Keywords to pass to plot call to draw X-point(s). Disable X-points by setting xkw={'marker': ''}

    :param ggd_points_triangles:
        Caching of ggd data structure as generated by omas_physics.grids_ggd_points_triangles() method

    :param \**kw: keywords passed to matplotlib plot statements

    :return: Axes instance
    """

    # caching of ggd data
    if ggd_points_triangles is None and 'equilibrium.grids_ggd' in ods:
        from .omas_physics import grids_ggd_points_triangles

        ggd_points_triangles = grids_ggd_points_triangles(ods['equilibrium.grids_ggd[0].grid[0]'])

    if allow_fallback is True:
        allow_fallback = 'psi'

    # time animation
    time_index, time = handle_time(ods, 'equilibrium', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(
                equilibrium_CX,
                ods,
                time_index,
                time,
                levels=levels,
                contour_quantity=contour_quantity,
                allow_fallback=allow_fallback,
                ax=ax,
                sf=sf,
                label_contours=label_contours,
                show_wall=show_wall,
                xkw=xkw,
                ggd_points_triangles=ggd_points_triangles,
                **kw,
            )

    import matplotlib
    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    return_dict = {'ax': ax}

    wall = None
    eq = ods['equilibrium']['time_slice'][time_index]
    if 'wall' in ods:
        if time_index in ods['wall']['description_2d']:
            wall = ods['wall']['description_2d'][time_index]['limiter']['unit']
        elif 0 in ods['wall']['description_2d']:
            wall = ods['wall']['description_2d'][0]['limiter']['unit']

    # Plotting style
    kw.setdefault('linewidth', 1)
    label = kw.pop('label', '')
    kw1 = copy.deepcopy(kw)
    kw1['linewidth'] = kw['linewidth'] + 1

    # Boundary
    ax.plot(eq['boundary.outline.r'], eq['boundary.outline.z'], label=label, **kw1)
    kw1.setdefault('color', ax.lines[-1].get_color())

    # Magnetic axis
    if 'global_quantities.magnetic_axis.r' in eq and 'global_quantities.magnetic_axis.z':
        ax.plot(eq['global_quantities']['magnetic_axis']['r'], eq['global_quantities']['magnetic_axis']['z'], '+', **kw1)

    # get 2d data either from grid or ggd
    def get2d(contour_quantity):
        pr2d = None
        if 'profiles_2d' in eq and 'profiles_2d.0.%s' % contour_quantity in eq:
            pr2d = eq['profiles_2d.0.%s' % contour_quantity]
        elif 'ggd.0.%s.0.values' % contour_quantity in eq:
            pr2d = eq['ggd.0.%s.0.values' % contour_quantity]
        return pr2d

    # Choose quantity to plot
    for fallback in range(2):
        # Most robust thing is to use PSI2D and interpolate 1D quantities over it
        if (
            get2d('psi') is not None
            and 'psi' in eq['profiles_1d']
            and contour_quantity in eq['profiles_1d']
            or contour_quantity == 'psi_norm'
        ):
            x_value_1d = eq['profiles_1d']['psi']
            m = x_value_1d[0]
            M = x_value_1d[-1]
            x_value_1d = (x_value_1d - m) / (M - m)
            x_value_2d = (get2d('psi') - m) / (M - m)
            if contour_quantity == 'psi_norm':
                value_1d = x_value_1d
            else:
                value_1d = eq['profiles_1d'][contour_quantity]
            value_2d = omas_interp1d(x_value_2d, x_value_1d, value_1d)
            break
        # Next get 2D quantity
        elif get2d(contour_quantity) is not None:
            value_1d = None
            value_2d = get2d(contour_quantity)
            break
        elif allow_fallback and not fallback:
            print('No %s equilibrium CX data to plot. Fallback on %s.' % (contour_quantity, allow_fallback))
            contour_quantity = allow_fallback
        # allow fallback
        elif fallback:
            txt = 'No %s equilibrium CX data to plot. Aborting.' % contour_quantity
            if allow_fallback:
                print(txt)
                return ax
            else:
                raise ValueError(txt)
    return_dict['contour_quantity'] = contour_quantity

    # handle levels
    if levels is None and value_1d is not None:
        if contour_quantity == 'q':
            max_q = int(numpy.round(omas_interp1d(0.95, x_value_1d, value_1d)))
            levels = numpy.arange(max_q)
        else:
            levels = numpy.linspace(numpy.min(value_1d), numpy.max(value_1d), 11)[1:-1]
            levels = numpy.hstack((levels, levels[-1] + (levels[1] - levels[0]) * numpy.arange(100)[1:]))

    # Wall clipping
    if wall is not None:
        path = matplotlib.path.Path(numpy.transpose(numpy.array([wall[0]['outline']['r'], wall[0]['outline']['z']])))
        wall_path = matplotlib.patches.PathPatch(path, facecolor='none', edgecolor='none')
        ax.add_patch(wall_path)

    kw.setdefault('colors', kw1['color'])
    kw.pop('color', '')
    kw['linewidths'] = kw.pop('linewidth')

    if 'profiles_2d.0' in eq:
        # Contours
        if 'r' in eq['profiles_2d.0'] and 'z' in eq['profiles_2d.0']:
            r = eq['profiles_2d.0.r']
            z = eq['profiles_2d.0.z']
        else:
            z, r = numpy.meshgrid(eq['profiles_2d.0.grid.dim2'], eq['profiles_2d.0.grid.dim1'])

        # sanitize
        value_2d = value_2d.copy()
        value_2d[:, -1] = value_2d[:, -2]
        value_2d[-1, :] = value_2d[-2, :]
        value_2d[-1, -1] = value_2d[-2, -2]

        # Resample
        if sf > 1:
            value_2d[numpy.isnan(value_2d)] = numpy.nanmean(value_2d)
            import scipy.ndimage

            r = scipy.ndimage.zoom(r, sf)
            z = scipy.ndimage.zoom(z, sf)
            value_2d = scipy.ndimage.zoom(value_2d, sf)

        cs = ax.contour(r, z, value_2d, levels, **kw)

        if label_contours or ((label_contours is None) and (contour_quantity == 'q')):
            ax.clabel(cs)

    elif 'ggd' in eq:
        cs = ax.tricontour(
            ggd_points_triangles[0][:, 0], ggd_points_triangles[0][:, 1], ggd_points_triangles[1], value_2d, levels=levels, **kw
        )
    else:
        raise Exception('No 2D equilibrium data to plot')

    if contour_quantity == 'q':
        ax.clabel(cs, cs.levels, inline=True, fontsize=10)

    # X-point(s)
    xkw.setdefault('marker', 'x')
    if xkw['marker'] not in ['', ' ']:
        from matplotlib import rcParams

        xkw.setdefault('color', cs.colors)
        xkw.setdefault('linestyle', '')
        xkw.setdefault('markersize', rcParams['lines.markersize'] * 1.5)
        xkw.setdefault('mew', rcParams['lines.markeredgewidth'] * 1.25 + 1.25)
        xp = eq['boundary']['x_point']
        for i in range(len(xp)):
            try:
                xr, xz = xp[i]['r'], xp[i]['z']
            except ValueError:
                pass
            else:
                ax.plot(xr, xz, **xkw)

    # Internal flux surfaces w/ or w/o masking
    if wall is not None:
        for collection in cs.collections:
            collection.set_clip_path(wall_path)

    # Wall
    if wall is not None and show_wall:
        ax.plot(wall[0]['outline']['r'], wall[0]['outline']['z'], 'k', linewidth=2)
        ax.axis([min(wall[0]['outline']['r']), max(wall[0]['outline']['r']), min(wall[0]['outline']['z']), max(wall[0]['outline']['z'])])

    # Axes
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    return return_dict


@add_to__ODS__
def equilibrium_CX_topview(ods, time_index=None, time=None, ax=None, **kw):
    r"""
    Plot equilibrium toroidal cross-section
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: ODS instance
        input ods containing equilibrium data

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param \**kw: arguments passed to matplotlib plot statements

    :return: Axes instance
    """
    # time animation
    time_index, time = handle_time(ods, 'equilibrium', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(equilibrium_CX_topview, time, ods, time_index, ax=ax, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    wall = None
    eq = ods['equilibrium']['time_slice'][time_index]
    if 'wall' in ods:
        if time_index in ods['wall']['description_2d']:
            wall = ods['wall']['description_2d'][time_index]['limiter']['unit']
        elif 0 in ods['wall']['description_2d']:
            wall = ods['wall']['description_2d'][0]['limiter']['unit']

    # Plotting style
    kw.setdefault('linewidth', 1)
    label = kw.pop('label', '')
    kw1 = copy.deepcopy(kw)

    t_angle = numpy.linspace(0.0, 2.0 * numpy.pi, 100)
    sint = numpy.sin(t_angle)
    cost = numpy.cos(t_angle)

    Rout = numpy.max(eq['boundary']['outline']['r'])
    Rin = numpy.min(eq['boundary']['outline']['r'])
    Xout = Rout * cost
    Yout = Rout * sint
    Xin = Rin * cost
    Yin = Rin * sint

    ax.plot(Xin, Yin, **kw1)
    kw1.setdefault('color', ax.lines[-1].get_color())
    ax.plot(Xout, Yout, **kw1)

    # Wall
    if wall is not None:
        Rout = numpy.max(wall[0]['outline']['r'])
        Rin = numpy.min(wall[0]['outline']['r'])
        Xout = Rout * cost
        Yout = Rout * sint
        Xin = Rin * cost
        Yin = Rin * sint

        ax.plot(Xin, Yin, 'k', label=label, linewidth=2)
        ax.plot(Xout, Yout, 'k', label=label, linewidth=2)
        ax.axis('equal')

    # Axes
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    return {'ax': ax}


nice_names = {
    'rho_tor_norm': '$\\rho$',
    'rho_tor': '$\\rho [m]$',
    'rho_volume_norm': '$\\rho_{\\rm vol}$',
    'psi_norm': '$\\psi$',
    'psi': '$\\psi$ [Wb]',
    'phi': '$\\phi$ [Wb]',
    'phi_norm': '$\\phi$',
    'q': '$q$',
}


@add_to__ODS__
def equilibrium_summary(ods, time_index=None, time=None, fig=None, ggd_points_triangles=None, **kw):
    """
    Plot equilibrium cross-section and P, q, P', FF' profiles
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param ggd_points_triangles:
        Caching of ggd data structure as generated by omas_physics.grids_ggd_points_triangles() method

    :param kw: arguments passed to matplotlib plot statements

    :return: figure handler
    """

    # caching of ggd data
    if ggd_points_triangles is None and 'equilibrium.grids_ggd' in ods:
        from .omas_physics import grids_ggd_points_triangles

        ggd_points_triangles = grids_ggd_points_triangles(ods['equilibrium.grids_ggd[0].grid[0]'])

    from matplotlib import pyplot

    axs = kw.pop('ax', {})
    if axs is None:
        axs = {}
    if not len(axs) and fig is None:
        fig = pyplot.figure()

    # time animation
    time_index, time = handle_time(ods, 'equilibrium', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(
                equilibrium_summary, ods, time_index, time, fig=fig, ggd_points_triangles=ggd_points_triangles, ax=axs, **kw
            )

    ax = cached_add_subplot(fig, axs, 1, 3, 1)
    contour_quantity = kw.pop('contour_quantity', 'rho_tor_norm')
    tmp = equilibrium_CX(
        ods, time_index=time_index, ax=ax, contour_quantity=contour_quantity, ggd_points_triangles=ggd_points_triangles, **kw
    )
    eq = ods['equilibrium']['time_slice'][time_index]

    # x
    if tmp['contour_quantity'] in eq['profiles_1d']:
        raw_xName = tmp['contour_quantity']
        x = eq['profiles_1d'][raw_xName]
    else:
        raw_xName = 'psi'
        x = eq['profiles_1d']['psi_norm']
        x = (x - min(x)) / (max(x) - min(x))
    xName = nice_names.get(raw_xName, raw_xName)

    # pressure
    ax = cached_add_subplot(fig, axs, 2, 3, 2)
    ax.plot(x, eq['profiles_1d']['pressure'], **kw)
    kw.setdefault('color', ax.lines[-1].get_color())
    ax.set_title(r'$\,$ Pressure')
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    pyplot.setp(ax.get_xticklabels(), visible=False)

    # q
    ax = cached_add_subplot(fig, axs, 2, 3, 3, sharex=ax)
    ax.plot(x, eq['profiles_1d']['q'], **kw)
    ax.set_title('$q$ Safety factor')
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    if 'label' in kw:
        leg = ax.legend(loc=0)
        import matplotlib

        if compare_version(matplotlib.__version__, '3.1.0') >= 0:
            leg.set_draggable(True)
        else:
            leg.draggable(True)
    pyplot.setp(ax.get_xticklabels(), visible=False)

    # dP_dpsi
    ax = cached_add_subplot(fig, axs, 2, 3, 5, sharex=ax)
    ax.plot(x, eq['profiles_1d']['dpressure_dpsi'], **kw)
    ax.set_title(r"$P\,^\prime$ source function")
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    pyplot.xlabel(xName)

    # FdF_dpsi
    ax = cached_add_subplot(fig, axs, 2, 3, 6, sharex=ax)
    ax.plot(x, eq['profiles_1d']['f_df_dpsi'], **kw)
    ax.set_title(r"$FF\,^\prime$ source function")
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    pyplot.xlabel(xName)

    if raw_xName.endswith('norm'):
        ax.set_xlim([0, 1])

    return {'ax': axs}


@add_to__ODS__
def core_profiles_summary(ods, time_index=None, time=None, fig=None, ods_species=None, quantities=['density_thermal', 'temperature'], **kw):
    """
    Plot densities and temperature profiles for electrons and all ion species
    as per `ods['core_profiles']['profiles_1d'][time_index]`

    :param ods: input ods

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ods_species: list of ion specie indices as listed in the core_profiles ods (electron index = -1)
        if None selected plot all the ion speciess

    :param quantities: list of strings to plot from the profiles_1d ods like zeff, temperature & rotation_frequency_tor_sonic

    :param kw: arguments passed to matplotlib plot statements

    :return: figure handler
    """

    from matplotlib import pyplot

    axs = kw.pop('ax', {})
    if axs is None:
        axs = {}
    if not len(axs) and fig is None:
        fig = pyplot.figure()

    # time animation
    time_index, time = handle_time(ods, 'core_profiles', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(
                core_profiles_summary, ods, time_index, time, fig=fig, ods_species=ods_species, quantities=quantities, ax=axs, **kw
            )

    prof1d = ods['core_profiles']['profiles_1d'][time_index]
    rho = prof1d['grid.rho_tor_norm']

    # Determine subplot rows x colls
    if ods_species is None:
        ncols = len(prof1d['ion']) + 1
        ods_species = [-1] + list(prof1d['ion'])
    else:
        ncols = len(ods_species)

    nplots = sum([ncols if 'density' in i or 'temperature' in i else 1 for i in quantities])
    nrows = int(numpy.ceil(nplots / ncols))

    # Generate species with corresponding name
    species_in_tree = [f"ion.{i}" if i >= 0 else 'electrons' for i in ods_species]
    names = [f"{prof1d[i]['label']} ion" if i != 'electrons' else "electron" for i in species_in_tree]

    plotting_list = []
    label_name = []
    label_name_z = []
    unit_list = []
    for q in quantities:
        if 'density' in q or 'temperature' in q:
            for index, specie in enumerate(species_in_tree):
                unit_list.append(omas_info_node(o2u(f"core_profiles.profiles_1d.0.{specie}.{q}"))['units'])
                if q in prof1d[specie]:
                    if 'density' in q and 'ion' in specie and prof1d[specie]['element[0].z_n'] != 1.0:
                        plotting_list.append(prof1d[specie][q] * prof1d[specie]['element[0].z_n'])
                        label_name_z.append(r'$\times$' + f" {int(prof1d[specie]['element[0].z_n'])}")
                    else:
                        plotting_list.append(prof1d[specie][q])
                        label_name_z.append("")
                    label_name.append(f'{names[index]} {q.capitalize()}')

                else:
                    plotting_list.append(numpy.zeros(len(rho)))

        else:
            unit_list.append(omas_info_node(o2u(f"core_profiles.profiles_1d.0.{q}"))['units'])
            plotting_list.append(prof1d[q])
            label_name.append(q.capitalize())

    for index, y in enumerate(plotting_list):
        plot = index + 1

        if index % ncols == 0:
            sharey = None
            sharex = None
        elif 'Density' in label_name[index] or 'Temperature' in label_name[index]:
            sharey = ax
            sharex = ax
        ax = cached_add_subplot(fig, axs, nrows, ncols, plot, sharex=sharex, sharey=sharey)

        uband(rho, y, ax=ax, **kw)
        if "Temp" in label_name[index]:
            ax.set_ylabel(r'$T_{}$'.format(label_name[index][0]) + imas_units_to_latex(unit_list[index]))
        elif "Density" in label_name[index]:
            ax.set_ylabel(r'$n_{}$'.format(label_name[index][0]) + imas_units_to_latex(unit_list[index]) + label_name_z[index])
        else:
            ax.set_ylabel(label_name[index][:10] + imas_units_to_latex(unit_list[index]))
        if (nplots - plot) < ncols:
            ax.set_xlabel('$\\rho$')
    if 'label' in kw:
        ax.legend(loc='lower center')
    ax.set_xlim([0, 1])

    return {'ax': axs, 'fig': fig}


@add_to__ODS__
def core_profiles_pressures(ods, time_index=None, time=None, ax=None, **kw):
    """
    Plot pressures in `ods['core_profiles']['profiles_1d'][time_index]`

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """

    # time animation
    time_index, time = handle_time(ods, 'core_profiles', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(core_profiles_pressures, ods, time_index, time, ax=ax)

    import matplotlib
    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    prof1d = ods['core_profiles']['profiles_1d'][time_index]
    x = prof1d['grid.rho_tor_norm']

    for item in prof1d.paths():
        item = l2o(item)
        if 'pressure' in item:
            if 'ion' in item:
                try:
                    i = int(item.split("ion.")[-1].split('.')[0])
                    label = prof1d['ion'][i]['label']
                except ValueError:
                    label = item
            elif 'electrons' in item:
                label = 'e$^-$'
            else:
                label = item
            if item != label:
                label += ' (thermal)' if 'thermal' in item else ''
                label += ' (fast)' if 'fast' in item else ''
            uband(x, prof1d[item], ax=ax, label=label)

    ax.set_xlim([0, 1])
    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlabel('$\\rho_N$')
    leg = ax.legend(loc=0)

    if compare_version(matplotlib.__version__, '3.1.0') >= 0:
        leg.set_draggable(True)
    else:
        leg.draggable(True)
    return {'ax': ax}


@add_to__ODS__
def core_transport_fluxes(ods, time_index=None, time=None, fig=None, show_total_density=True, plot_zeff=False, **kw):
    """
    Plot densities and temperature profiles for all species, rotation profile, TGYRO fluxes and fluxes from power_balance per STEP state.

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param show_total_density: bool
        Show total thermal+fast in addition to thermal/fast breakdown if available

    :param plot_zeff: if True, plot zeff below the plasma rotation

    :kw: matplotlib plot parameters

    :return: axes
    """
    from matplotlib import pyplot

    axs = kw.pop('ax', {})
    if axs is None:
        axs = {}
    if not len(axs) and fig is None:
        fig = pyplot.figure()

    # time animation
    time_index, time = handle_time(ods, 'core_profiles', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(
                core_transport_fluxes,
                ods,
                time_index,
                time,
                fig=fig,
                ax=axs,
                show_total_density=show_total_density,
                plot_zeff=plot_zeff,
                **kw,
            )

    def sum_density_types(specie_index):
        final_density = numpy.zeros(len(prof1d['grid.rho_tor_norm']))
        for therm_fast in ['_thermal', '_fast']:
            if not show_total_density and therm_fast != "_thermal":
                continue  # Skip total thermal+fast because the flag turned it off
            density = ods_species[specie_index] + '.density' + therm_fast
            if density not in prof1d:
                continue
            final_density += prof1d[density]
        return final_density

    def plot_function(x, y, plot_num, ylabel, sharex=None, sharey=None):
        ax = cached_add_subplot(fig, axs, nrows, ncols, plot_num, sharex=sharex, sharey=sharey)
        uband(x, y, ax=ax, **kw)
        ax.set_ylabel(ylabel)
        return ax

    if plot_zeff:
        nrows = 5
    else:
        nrows = 4
    ncols = 2

    if "core_profiles" in ods:
        ods.physics_core_profiles_densities()
        prof1d = ods['core_profiles']['profiles_1d'][time_index]
        equilibrium = ods['equilibrium']['time_slice'][time_index]
        rho_core_prof = prof1d['grid.rho_tor_norm']

        ods_species = ['electrons'] + ['ion[%d]' % k for k in range(len(prof1d['ion']))]
        species_name = ['Electrons'] + [prof1d['ion[%d].label' % k] + ' ion' for k in range(len(prof1d['ion']))]

        # Temperature electrons
        ax = plot_function(x=rho_core_prof, y=prof1d[ods_species[0]]['temperature'] / 1e3, plot_num=1, ylabel='$T_{e}\,[keV]$')
        pyplot.setp(ax.get_xticklabels(), visible=False)

        # Temperature main ion species
        ax = plot_function(
            x=rho_core_prof, y=prof1d[ods_species[1]]['temperature'] / 1e3, plot_num=3, ylabel='$T_{i}\,[keV]$', sharey=ax, sharex=ax
        )
        pyplot.setp(ax.get_xticklabels(), visible=False)

        # Density electrons
        ax = plot_function(x=rho_core_prof, y=sum_density_types(specie_index=0), plot_num=5, ylabel='$n_{e}\,[m^{-3}]$', sharex=ax)
        pyplot.setp(ax.get_xticklabels(), visible=False)

        # Rotation
        if 'rotation_frequency_tor_sonic' in prof1d and 'psi' in prof1d['grid']:
            from .omas_physics import omas_environment

            with omas_environment(
                ods,
                coordsio={
                    f'equilibrium.time_slice.{k}.profiles_1d.psi': prof1d['grid']['psi'] for k in range(len(ods['equilibrium.time_slice']))
                },
            ):
                rotation = (equilibrium['profiles_1d']['r_outboard'] - equilibrium['profiles_1d']['r_inboard']) / 2 + equilibrium[
                    'profiles_1d'
                ]['geometric_axis']['r'] * -prof1d['rotation_frequency_tor_sonic']
                ax = plot_function(x=rho_core_prof, y=rotation, plot_num=7, ylabel='R*$\Omega_0$ (m/s)', sharex=ax)
                if not plot_zeff:
                    ax.set_xlabel('$\\rho$')

        # Zeff
        if plot_zeff:
            pyplot.setp(ax.get_xticklabels(), visible=False)
            ax = plot_function(x=rho_core_prof, y=prof1d['zeff'], plot_num=9, ylabel='$Z_{eff}$', sharex=ax)
            ax.set_xlabel('$\\rho$')

        # Fluxes
        if "core_transport" in ods:
            core_transport = ods['core_transport']['model']
            rho_transport_model = core_transport[0]['profiles_1d'][time_index]['grid_d']['rho_tor']

            # Qe
            ax = plot_function(
                x=rho_transport_model,
                y=core_transport[2]['profiles_1d'][time_index]['electrons']['energy']['flux'],
                plot_num=2,
                ylabel='$Q_e$ [W/$m^2$]',
                sharex=ax,
            )
            color = ax.lines[-1].get_color()
            uband(
                x=rho_transport_model,
                y=core_transport[3]['profiles_1d'][time_index]['electrons']['energy']['flux'],
                ax=ax,
                marker='o',
                ls='None',
                color=color,
            )
            uband(
                x=rho_core_prof, y=core_transport[4]['profiles_1d'][time_index]['electrons']['energy']['flux'], ax=ax, ls='--', color=color
            )
            pyplot.setp(ax.get_xticklabels(), visible=False)

            # Add legend on top (black) as it applies to all lines
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color='k', ls='--', label='Power Balance'),
                Line2D([0], [0], color='k', label='Model total'),
                Line2D([0], [0], marker='o', ls='None', color='k', label='Model target', markersize=6),
            ]

            fig.legend(handles=legend_elements).set_draggable(True)

            # Qi
            ax = plot_function(
                x=rho_transport_model,
                y=core_transport[2]['profiles_1d'][time_index]['total_ion_energy']['flux'],
                plot_num=4,
                ylabel='$Q_i$ [W/$m^2$]',
                sharex=ax,
                sharey=ax,
            )
            uband(
                x=rho_transport_model,
                y=core_transport[3]['profiles_1d'][time_index]['total_ion_energy']['flux'],
                ax=ax,
                marker='o',
                ls='None',
                color=color,
            )
            uband(x=rho_core_prof, y=core_transport[4]['profiles_1d'][time_index]['total_ion_energy']['flux'], ax=ax, ls='--', color=color)
            pyplot.setp(ax.get_xticklabels(), visible=False)

            # Particle flux (electron particle source)
            ax = plot_function(
                x=rho_transport_model,
                y=3 / 2 * core_transport[2]['profiles_1d'][time_index]['electrons']['particles']['flux'],
                plot_num=6,
                ylabel=r'$ \frac{3}{2}T_{e}\Gamma_{e}$ [W/$m^2$]',
                sharex=ax,
            )
            uband(
                x=rho_transport_model,
                y=3 / 2 * core_transport[3]['profiles_1d'][time_index]['electrons']['particles']['flux'],
                ax=ax,
                marker='o',
                ls='None',
                color=color,
            )
            pyplot.setp(ax.get_xticklabels(), visible=False)

            # Pi (toroidal momentum flux)
            ax = plot_function(
                x=rho_transport_model,
                y=core_transport[2]['profiles_1d'][time_index]['momentum_tor']['flux'],
                plot_num=8,
                ylabel='$\Pi_{i}$ [N/$m$]',
                sharex=ax,
            )
            ax.set_xlabel('$\\rho$')

            uband(
                x=rho_transport_model,
                y=core_transport[3]['profiles_1d'][time_index]['momentum_tor']['flux'],
                ax=ax,
                marker='o',
                ls='None',
                color=color,
            )
            uband(x=rho_core_prof, y=core_transport[4]['profiles_1d'][time_index]['momentum_tor']['flux'], ax=ax, ls='--', color=color)

            ax.set_xlim(0, 1)

    return {'ax': axs, 'fig': fig}


@add_to__ODS__
def core_sources_summary(ods, time_index=None, time=None, fig=None, **kw):
    """
    Plot sources for electrons and all ion species

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes
    """
    import matplotlib
    from matplotlib import pyplot

    axs = kw.pop('ax', {})
    if axs is None:
        axs = {}
    if not len(axs) and fig is None:
        fig = pyplot.figure()

    # time animation
    time_index, time = handle_time(ods, 'core_sources', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(core_sources, ods, time_index, time, fig=fig, ax=axs ** kw)

    colors = [k['color'] for k in list(matplotlib.rcParams['axes.prop_cycle'])]
    lss = ['-', '--', 'dotted']
    colors, lss = numpy.meshgrid(colors, lss)
    if len(ods[f'core_sources.source']) > len(colors):
        colors = colors.T
        lss = lss.T
    colors = colors.flatten()
    lss = lss.flatten()

    # if list is too small use all colors
    if len(ods[f'core_sources.source']) > len(colors):
        import matplotlib.colors as mcolors

        colors = list(mcolors.CSS4_COLORS.keys())

    for k, s in enumerate(ods['core_sources.source']):
        rho = ods[f'core_sources.source.{s}.profiles_1d.{time_index}.grid.rho_tor_norm']
        label = ods[f'core_sources.source.{s}.identifier.name']

        tmp = {}
        tmp[f'core_sources.source.{s}.profiles_1d.{time_index}.electrons.energy'] = ('$q_e$', 'linear')
        tmp[f'core_sources.source.{s}.profiles_1d.{time_index}.total_ion_energy'] = ('$q_i$', 'linear')
        tmp[None] = None
        tmp[f'core_sources.source.{s}.profiles_1d.{time_index}.electrons.particles'] = ('$p_e$', 'linear')
        tmp[f'core_sources.source.{s}.profiles_1d.{time_index}.j_parallel'] = ('$J_\parallel$', 'linear')
        tmp[f'core_sources.source.{s}.profiles_1d.{time_index}.momentum_tor'] = (r'$\pi_i$', 'linear')

        ax = None
        for kp, item in enumerate(tmp):
            if item is None:
                continue
            ax = cached_add_subplot(fig, axs, 2, 3, kp + 1, sharex=ax)
            if item in ods:
                ax.plot(rho, ods[item], label=label, color=colors[k], ls=lss[k])
            else:
                ax.plot(numpy.nan, numpy.nan, label=label, color=colors[k], ls=lss[k])
            ax.set_title(tmp[item][0])
            ax.set_yscale(tmp[item][1])

        ax.legend(loc=0)
    return {'ax': axs, 'fig': fig}


@add_to__ODS__
def pf_active_data(ods, equilibrium_constraints=True, ax=None, **kw):
    '''
    plot pf_active time traces

    :param equilibrium_constraints: plot equilibrium constraints if present

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param \**kw: Additional keywords for plot

    :return: axes instance
    '''

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # time traces
    for channel in ods['pf_active.coil']:
        label = ods[f'pf_active.coil.{channel}.element[0].identifier']
        turns = ods[f'pf_active.coil.{channel}.element[0].turns_with_sign']
        data = ods[f'pf_active.coil.{channel}.current.data']
        time = ods[f'pf_active.coil.{channel}.current.time']
        ax.plot(time, data * turns, label=label, **kw)

    # equilibrium constraints
    if equilibrium_constraints:
        for channel in ods['pf_active.coil']:
            if f'equilibrium.time_slice.0.constraints.pf_current.{channel}.measured' in ods:
                ax.plot(
                    ods[f'equilibrium.time'],
                    ods[f'equilibrium.time_slice.:.constraints.pf_current.{channel}.measured'],
                    marker='o',
                    color='k',
                    mec='none',
                )

    return ax


@add_to__ODS__
def magnetics_bpol_probe_data(ods, equilibrium_constraints=True, ax=None, **kw):
    '''
    plot bpol_probe time traces and equilibrium constraints

    :param equilibrium_constraints: plot equilibrium constraints if present

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param \**kw: Additional keywords for plot

    :return: axes instance
    '''

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # time traces
    for channel in ods['magnetics.b_field_pol_probe']:
        valid = ods.get(f'magnetics.b_field_pol_probe.{channel}.field.validity', 0)
        if valid == 0:
            label = ods[f'magnetics.b_field_pol_probe.{channel}.identifier']
            data = ods[f'magnetics.b_field_pol_probe.{channel}.field.data']
            time = ods[f'magnetics.b_field_pol_probe.{channel}.field.time']
            ax.plot(time, data, label=label, **kw)

    # equilibrium constraints
    if equilibrium_constraints:
        for channel in ods['magnetics.b_field_pol_probe']:
            valid = ods.get(f'magnetics.b_field_pol_probe.{channel}.field.validity', 0)
            if valid == 0:
                if f'equilibrium.time_slice.0.constraints.bpol_probe.{channel}.measured' in ods:
                    ax.plot(
                        ods[f'equilibrium.time'],
                        ods[f'equilibrium.time_slice.:.constraints.bpol_probe.{channel}.measured'],
                        marker='o',
                        color='k',
                        mec='none',
                    )

    return ax


@add_to__ODS__
def magnetics_flux_loop_data(ods, equilibrium_constraints=True, ax=None, **kw):
    '''
    plot flux_loop time traces and equilibrium constraints

    :param equilibrium_constraints: plot equilibrium constraints if present

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param \**kw: Additional keywords for plot

    :return: axes instance
    '''

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # time traces
    for channel in ods['magnetics.flux_loop']:
        valid = ods.get(f'magnetics.flux_loop.{channel}.flux.validity', 0)
        if valid == 0:
            label = ods[f'magnetics.flux_loop.{channel}.identifier']
            data = ods[f'magnetics.flux_loop.{channel}.flux.data']
            time = ods[f'magnetics.flux_loop.{channel}.flux.time']
            ax.plot(time, data, label=label, **kw)

    # equilibrium constraints
    if equilibrium_constraints:
        for channel in ods['magnetics.flux_loop']:
            valid = ods.get(f'magnetics.flux_loop.{channel}.flux.validity', 0)
            if valid == 0:
                if f'equilibrium.time_slice.0.constraints.flux_loop.{channel}.measured' in ods:
                    ax.plot(
                        ods[f'equilibrium.time'],
                        ods[f'equilibrium.time_slice.:.constraints.flux_loop.{channel}.measured'],
                        marker='o',
                        color='k',
                        mec='none',
                    )

    return ax


@add_to__ODS__
def magnetics_ip_data(ods, equilibrium_constraints=True, ax=None, **kw):
    '''
    plot ip time trace and equilibrium constraint

    :param equilibrium_constraints: plot equilibrium constraints if present

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param \**kw: Additional keywords for plot

    :return: axes instance
    '''
    return _plot_signal_eq_constraint(
        ods,
        'magnetics.ip.0.time',
        'magnetics.ip.0.data',
        'equilibrium.time_slice.:.constraints.ip.measured',
        equilibrium_constraints,
        ax,
        label='ip',
        **kw,
    )


@add_to__ODS__
def magnetics_diamagnetic_flux_data(ods, equilibrium_constraints=True, ax=None, **kw):
    '''
    plot diamagnetic_flux time trace and equilibrium constraint

    :param equilibrium_constraints: plot equilibrium constraints if present

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param \**kw: Additional keywords for plot

    :return: axes instance
    '''
    return _plot_signal_eq_constraint(
        ods,
        'magnetics.diamagnetic_flux.0.time',
        'magnetics.diamagnetic_flux.0.data',
        'equilibrium.time_slice.:.constraints.diamagnetic_flux.measured',
        equilibrium_constraints,
        ax,
        label='dflux',
        **kw,
    )


@add_to__ODS__
def tf_b_field_tor_vacuum_r_data(ods, equilibrium_constraints=True, ax=None, **kw):
    '''
    plot b_field_tor_vacuum_r time trace and equilibrium constraint

    :param equilibrium_constraints: plot equilibrium constraints if present

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param \**kw: Additional keywords for plot

    :return: axes instance
    '''
    return _plot_signal_eq_constraint(
        ods,
        'tf.b_field_tor_vacuum_r.time',
        'tf.b_field_tor_vacuum_r.data',
        'equilibrium.time_slice.:.constraints.b_field_tor_vacuum_r.measured',
        equilibrium_constraints,
        ax,
        label='bt',
        **kw,
    )


def _plot_signal_eq_constraint(ods, time, data, constraint, equilibrium_constraints, ax, **kw):
    '''
    Utility function to plot individual signal and their constraint in equilibrium IDS

    :param time: ods location for time

    :param data: ods location for data

    :param constraint: ods location fro equilibrium constraint

    :param ax: axes where to plot

    :param kw: extra arguments passed to

    :return:
    '''
    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # time traces
    time = ods[time]
    data = ods[data]
    ax.plot(time, data, **kw)

    # equilibrium constraints
    if equilibrium_constraints and constraint in ods:
        ax.plot(ods['equilibrium.time'], ods[constraint], ls='', marker='o', color='k', mec='none')
    return ax


# ================================
# actuator aimings
# ================================


@add_to__ODS__
def pellets_trajectory_CX(ods, time_index=None, time=None, ax=None, **kw):
    """
    Plot pellets trajectory in poloidal cross-section

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """

    # time animation
    time_index, time = handle_time(ods, 'pellets', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(pellets_trajectory_CX, ods, time_index, time, ax=ax, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    pellets = ods['pellets']['time_slice'][time_index]['pellet']
    for pellet in pellets:
        R0 = pellets[pellet]['path_geometry.first_point.r']
        R1 = pellets[pellet]['path_geometry.second_point.r']
        Z0 = pellets[pellet]['path_geometry.first_point.z']
        Z1 = pellets[pellet]['path_geometry.second_point.z']
        ax.plot([R0, R1], [Z0, Z1], '--', **kw)

    return {'ax': ax}


@add_to__ODS__
def pellets_trajectory_CX_topview(ods, time_index=None, time=None, ax=None, **kw):
    """
    Plot  pellet trajectory in toroidal cross-section

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """
    # time animation
    time_index, time = handle_time(ods, 'pellets', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(pellets_trajectory_CX_topview, ods, time_index, time, ax=ax, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    pellets = ods['pellets']['time_slice'][time_index]['pellet']
    for pellet in pellets:
        R0 = pellets[pellet]['path_geometry.first_point.r']
        R1 = pellets[pellet]['path_geometry.second_point.r']
        phi0 = pellets[pellet]['path_geometry.first_point.phi']
        phi1 = pellets[pellet]['path_geometry.second_point.phi']

        x0 = R0 * numpy.cos(phi0)
        y0 = R0 * numpy.sin(phi0)

        x1 = R1 * numpy.cos(phi1)
        y1 = R1 * numpy.sin(phi1)
        ax.plot([x0, x1], [y0, y1], '--', **kw)

    return {'ax': ax}


@add_to__ODS__
def lh_antennas_CX(ods, time_index=None, time=None, ax=None, antenna_trajectory=None, **kw):
    """
    Plot LH antenna position in poloidal cross-section

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param antenna_trajectory: length of antenna on plot

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """
    # time animation
    time_index, time = handle_time(ods, 'lh_antennas', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(lh_antennas_CX, ods, time_index, time, ax=ax, antenna_trajectory=antenna_trajectory, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    equilibrium = ods['equilibrium']['time_slice'][time_index]
    antennas = ods['lh_antennas']['antenna']

    if antenna_trajectory is None:
        antenna_trajectory = 0.1 * ods['equilibrium']['vacuum_toroidal_field.r0']

    for antenna in antennas:
        R = antennas[antenna]['position.r.data']
        Z = antennas[antenna]['position.z.data']

        # just point to magnetic axis for now (is there a better way?)
        Raxis = equilibrium['global_quantities.magnetic_axis.r']
        Zaxis = equilibrium['global_quantities.magnetic_axis.z']

        Rvec = Raxis - R
        Zvec = Zaxis - Z

        R1 = R + Rvec * antenna_trajectory / numpy.sqrt(Rvec ** 2 + Zvec ** 2)
        Z1 = Z + Zvec * antenna_trajectory / numpy.sqrt(Rvec ** 2 + Zvec ** 2)

        ax.plot([R, R1], [Z, Z1], 's-', markevery=2, **kw)

    return {'ax': ax}


@add_to__ODS__
def lh_antennas_CX_topview(ods, time_index=None, time=None, ax=None, antenna_trajectory=None, **kw):
    """
    Plot LH antenna in toroidal cross-section

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :param antenna_trajectory: length of antenna on plot

    :return: axes handler
    """
    # time animation
    time_index, time = handle_time(ods, 'lh_antennas', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(lh_antennas_CX_topview, ods, time_index, time, ax=ax, antenna_trajectory=antenna_trajectory, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    equilibrium = ods['equilibrium']
    antennas = ods['lh_antennas']['antenna']
    if antenna_trajectory is None:
        antenna_trajectory = 0.1 * equilibrium['vacuum_toroidal_field.r0']

    for antenna in antennas:
        R = antennas[antenna]['position.r.data']
        phi = antennas[antenna]['position.phi.data']

        x0 = R * numpy.cos(phi)
        y0 = R * numpy.sin(phi)

        x1 = (R - antenna_trajectory) * numpy.cos(phi)
        y1 = (R - antenna_trajectory) * numpy.sin(phi)

        ax.plot([x0, x1], [y0, y1], 's-', markevery=2, **kw)

    return {'ax': ax}


@add_to__ODS__
def ec_launchers_CX(ods, time_index=None, time=None, ax=None, launcher_trajectory=None, **kw):
    """
    Plot EC launchers in poloidal cross-section

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :param launcher_trajectory: length of launcher on plot

    :return: axes handler
    """
    # time animation
    time_index, time = handle_time(ods, 'ec_launchers', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(ec_launchers_CX, ods, time_index, time, ax=ax, launcher_trajectory=launcher_trajectory, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    equilibrium = ods['equilibrium']
    launchers = ods['ec_launchers.launcher']
    if launcher_trajectory is None:
        launcher_trajectory = 0.1 * equilibrium['vacuum_toroidal_field.r0']

    for launcher in launchers:
        R0 = launchers[launcher]['launching_position.r']
        Z0 = launchers[launcher]['launching_position.z']
        ang_tor = launchers[launcher]['steering_angle_tor.data']
        ang_pol = launchers[launcher]['steering_angle_pol.data']
        ang_pol_proj = 0.5 * numpy.pi - numpy.arctan2(numpy.tan(ang_pol), numpy.cos(ang_tor))

        R1 = R0 - launcher_trajectory * numpy.cos(ang_pol_proj)
        Z1 = Z0 - launcher_trajectory * numpy.sin(ang_pol_proj)
        ax.plot([R0, R1], [Z0, Z1], 'o-', markevery=2, **kw)

        R1 = R0 - launcher_trajectory * numpy.cos(ang_pol)
        Z1 = Z0 - launcher_trajectory * numpy.sin(ang_pol)
        ax.plot([R0, R1], [Z0, Z1], 'o-', markevery=2, **kw)

    return {'ax': ax}


@add_to__ODS__
def ec_launchers_CX_topview(ods, time_index=None, time=None, ax=None, launcher_trajectory=None, **kw):
    """
    Plot EC launchers in toroidal cross-section

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :param launcher_trajectory: length of launcher on plot

    :return: axes handler
    """
    # time animation
    time_index, time = handle_time(ods, 'ec_launchers', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(ec_launchers_CX_topview, ods, time_index, time, ax=ax, launcher_trajectory=launcher_trajectory, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    equilibrium = ods['equilibrium']
    launchers = ods['ec_launchers.launcher']
    if launcher_trajectory is None:
        launcher_trajectory = 0.1 * equilibrium['vacuum_toroidal_field.r0']

    for launcher in launchers:
        R = launchers[launcher]['launching_position.r']
        phi = launchers[launcher]['launching_position.phi']
        ang_tor = launchers[launcher]['steering_angle_tor.data']

        x0 = R * numpy.cos(phi)
        y0 = R * numpy.sin(phi)
        x1 = x0 - launcher_trajectory * numpy.cos(ang_tor + phi)
        y1 = y0 - launcher_trajectory * numpy.sin(ang_tor + phi)
        ax.plot([x0, x1], [y0, y1], 'o-', markevery=2, **kw)

    return {'ax': ax}


# ================================
# Heating and current drive
# ================================
@add_to__ODS__
def waves_beam_CX(ods, time_index=None, time=None, ax=None, **kw):
    """
    Plot waves beams in poloidal cross-section

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """
    # time animation
    time_index, time = handle_time(ods, 'waves', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(waves_beam_CX, ods, time_index, time, ax=ax, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    coherent_wave = ods['waves.coherent_wave']

    for cw in coherent_wave:
        bt = coherent_wave[cw]['beam_tracing'][time_index]
        for b in bt['beam'].values():
            ax.plot(b['position.r'], b['position.z'], **kw)
            # plotc(b['position.r'], b['position.z'], b['electrons.power']/max(b['electrons.power']), ax=ax, **kw)

    return {'ax': ax}


@add_to__ODS__
def waves_beam_profile(ods, time_index=None, time=None, what=['power_density', 'current_parallel_density'][0], ax=None, **kw):
    """
    Plot 1d profiles of waves beams given quantity

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param quantity: quantity to plot

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """
    # time animation
    time_index, time = handle_time(ods, 'waves', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(waves_beam_profile, ods, time_index, time, what=what, ax=ax, **kw)

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    coherent_wave = ods['waves.coherent_wave']

    for cw in coherent_wave:
        b = coherent_wave[cw]['profiles_1d'][time_index]
        ax.plot(b['grid.rho_tor_norm'], b[what], **kw)
    ax.set_title(what.replace('_', ' ').capitalize())
    ax.set_ylabel('[%s]' % omas_info_node(b.ulocation + '.' + what)['units'])
    ax.set_xlabel('rho')

    return {'ax': ax}


@add_to__ODS__
def waves_beam_summary(ods, time_index=None, time=None, fig=None, **kw):
    """
    Plot waves beam summary: CX, power_density, and current_parallel_density

    :param ods: input ods

    :param time_index: int, list of ints, or None
        time slice to plot. If None all timeslices are plotted.

    :param time: float, list of floats, or None
        time to plot. If None all timeslicess are plotted.
        if not None, it takes precedence over time_index

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: figure handler
    """

    from matplotlib import pyplot

    axs = kw.pop('ax', {})
    if axs is None:
        axs = {}
    if not len(axs) and fig is None:
        fig = pyplot.figure()

    # time animation
    time_index, time = handle_time(ods, 'waves', time_index, time)
    if isinstance(time_index, (list, numpy.ndarray)):
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(waves_beam_summary, ods, time_index, time, fig=fig, ax={}, **kw)

    ax = cached_add_subplot(fig, axs, 1, 2, 1)
    waves_beam_CX(ods, time_index=time_index, ax=ax, **kw)

    ax = cached_add_subplot(fig, axs, 2, 2, 2)
    waves_beam_profile(ods, time_index=time_index, what='power_density', ax=ax, **kw)
    ax.set_xlabel('')

    ax = cached_add_subplot(fig, axs, 2, 2, 4, sharex=ax)
    waves_beam_profile(ods, time_index=time_index, what='current_parallel_density', ax=ax, **kw)

    ax.set_xlim([0, 1])

    return {'ax': axs}


@add_to__ODS__
def nbi_summary(ods, ax=None):
    '''
    Plot summary of NBI power time traces

    :param ods: input ods

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :return: axes handler
    '''
    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    time = ods['nbi.time']
    nbi = ods['nbi.unit']
    tmp = []
    for beam in nbi:
        tmp.append(nbi[beam]['power_launched.data'])
        ax.plot(time, tmp[-1], label=nbi[beam]['identifier'])

    ax.plot(time, numpy.sum(tmp, 0), 'k', lw=2, label='Total')

    ax.set_title('Neutral Beam Injectors power')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Power [W]')
    ax.legend()

    return {'ax': ax}


# ================================
# Hardware overlays
# ================================
@add_to__ODS__
def overlay(ods, ax=None, allow_autoscale=True, debug_all_plots=False, return_overlay_list=False, **kw):
    r"""
    Plots overlays of hardware/diagnostic locations on a tokamak cross section plot

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param allow_autoscale: bool
        Certain overlays will be allowed to unlock xlim and ylim, assuming that they have been locked by equilibrium_CX.
        If this option is disabled, then hardware systems like PF-coils will be off the plot and mostly invisible.

    :param debug_all_plots: bool
        Individual hardware systems are on by default instead of off by default.

    :param return_overlay_list:
        Return list of possible overlays that could be plotted

    :param \**kw: additional keywords for selecting plots.

        - Select plots by setting their names to True; e.g.: if you want the gas_injection plot, set gas_injection=True
          as a keyword.
          If debug_all_plots is True, then you can turn off individual plots by, for example, set_gas_injection=False.

        - Instead of True to simply turn on an overlay, you can pass a dict of keywords to pass to a particular overlay
          method, as in thomson={'labelevery': 5}. After an overlay pops off its keywords, remaining keywords are passed
          to plot, so you can set linestyle, color, etc.

        - Overlay functions accept these standard keywords:
            * mask: bool array
                Set of flags for switching plot elements on/off. Must be equal to the number of channels or items to be
                plotted.

            * labelevery: int
                Sets how often to add labels to the plot. A setting of 0 disables labels, 1 labels every element,
                2 labels every other element, 3 labels every third element, etc.

            * notesize: matplotlib font size specification
                Applies to annotations drawn on the plot. Examples: 'xx-small', 'medium', 16

            * label_ha: None or string or list of (None or string) instances
                Descriptions of how labels should be aligned horizontally. Either provide a single specification or a
                list of specs matching or exceeding the number of labels expected.
                Each spec should be: 'right', 'left', or 'center'. None (either as a scalar or an item in the list) will
                give default alignment for the affected item(s).

            * label_va: None or string or list of (None or string) instances
                Descriptions of how labels should be aligned vertically. Either provide a single specification or a
                list of specs matching or exceeding the number of labels expected.
                Each spec should be: 'top', 'bottom', 'center', 'baseline', or 'center_baseline'.
                None (either as a scalar or an item in the list) will give default alignment for the affected item(s).

            * label_r_shift: float or float array/list.
                Add an offset to the R coordinates of all text labels for the current hardware system.
                (in data units, which would normally be m)
                Scalar: add the same offset to all labels.
                Iterable: Each label can have its own offset.
                    If the list/array of offsets is too short, it will be padded with 0s.

            * label_z_shift: float or float array/list
                Add an offset to the Z coordinates of all text labels for the current hardware system
                (in data units, which would normally be m)
                Scalar: add the same offset to all labels.
                Iterable: Each label can have its own offset.
                    If the list/array of offsets is too short, it will be padded with 0s.

            * Additional keywords are passed to the function that does the drawing; usually matplotlib.axes.Axes.plot().

    :return: axes handler
    """

    if return_overlay_list:
        return [k.replace('_overlay', '') for k in __ods__ if k.endswith('_overlay') and k.replace('_overlay', '') in ods]

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    special_subs = ['position_control']
    for hw_sys in list_structures(ods.imas_version) + special_subs:
        if kw.get(hw_sys, debug_all_plots):
            try:
                overlay_function = eval('{}_overlay'.format(hw_sys))
            except NameError:
                continue
            overlay_kw = kw.get(hw_sys, {}) if isinstance(kw.get(hw_sys, {}), dict) else {}
            for k in ['mask', 'labelevery', 'notesize', 'label_ha', 'label_va', 'label_r_shift', 'label_z_shift']:
                if k in kw and k not in overlay_kw:
                    overlay_kw[k] = kw[k]
            if allow_autoscale and hw_sys in ['pf_active', 'gas_injection']:  # Not all systems need expanded range to fit everything
                ax.set_xlim(auto=True)
                ax.set_ylim(auto=True)
            overlay_function(ods, ax, **overlay_kw)

    return {'ax': ax}


@add_to__ODS__
def wall_overlay(ods, ax=None, component_index=None, types=['limiter', 'mobile', 'vessel'], unit_index=None, **kw):
    '''
    Plot walls on a tokamak cross section plot

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param component_index: list of index of components to plot

    :param types: list with one or more of ['limiter','mobile','vessel']

    :param unit_index: list of index of units of the component to plot

    :return: axes handler
    '''
    from matplotlib import pyplot

    for k in ['mask', 'labelevery', 'notesize', 'label_ha', 'label_va', 'label_r_shift', 'label_z_shift']:
        kw.pop(k, None)
    kw.setdefault('color', 'k')

    if ax is None:
        ax = pyplot.gca()

    if component_index is None:
        component_index = ods['wall.description_2d'].keys()
    elif isinstance(component_index, int):
        component_index = [component_index]
    elif isinstance(component_index, str):
        component_index = [ods['wall.description_2d[:].limiter.type.name'].index(component_index)]

    for component in component_index:
        for type in types:
            if type not in ods[f'wall.description_2d[{component}]']:
                continue
            if unit_index is None:
                unit_index = ods[f'wall.description_2d[{component}].{type}.unit'].keys()
            elif isinstance(unit_index, int):
                component_index = [unit_index]
            elif isinstance(unit_index, str):
                component_index = [ods[f'wall.description_2d[{component}].{type}.unit[{unit}].type.name'].index(component_index)]

            for unit in ods[f'wall.description_2d[{component}].{type}.unit']:
                ax.plot(
                    ods[f'wall.description_2d[{component}].{type}.unit[{unit}].outline.r'],
                    ods[f'wall.description_2d[{component}].{type}.unit[{unit}].outline.z'],
                    **kw,
                )

    ax.set_aspect('equal')

    return {'ax': ax}


@add_to__ODS__
def gas_injection_overlay(
    ods,
    ax=None,
    angle_not_in_pipe_name=False,
    which_gas='all',
    show_all_pipes_in_group=True,
    simple_labels=False,
    label_spacer=0,
    colors=None,
    draw_arrow=True,
    **kw,
):
    r"""
    Plots overlays of gas injectors

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param angle_not_in_pipe_name: bool
        Set this to include (Angle) at the end of injector labels. Useful if injector/pipe names don't already include
        angles in them.

    :param which_gas: string or list
        Filter for selecting which gas valves to display.

        - If string: get a preset group, like 'all'.

        - If list: only valves in the list will be shown. Abbreviations are tolerated; e.g. GASA is recognized as
          GASA_300. One abbreviation can turn on several valves. There are several valve names starting with
          RF_ on DIII-D, for example.

    :param show_all_pipes_in_group: bool
        Some pipes have the same R,Z coordinates of their exit positions (but different phi locations) and will
        appear at the same location on the plot. If this keyword is True, labels for all the pipes in such a group
        will be displayed together. If it is False, only the first one in the group will be labeled.

    :param simple_labels: bool
        Simplify labels by removing suffix after the last underscore.

    :param label_spacer: int
        Number of blank lines and spaces to insert between labels and symbol

    :param colors: list of matplotlib color specifications.
        These colors control the display of various gas ports. The list will be repeated to make sure it is long enough.
        Do not specify a single RGB tuple by itself. However, a single tuple inside list is okay [(0.9, 0, 0, 0.9)].
        If the color keyword is used (See \**kw), then color will be popped to set the default for colors in case colors
        is None.

    :param draw_arrow: bool or dict
        Draw an arrow toward the machine at the location of the gas valve. If dict, pass keywords to arrow drawing func.

    :param \**kw: Additional keywords for gas plot:

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Remaining keywords are passed to plot call for drawing markers at the gas locations.
    """

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # Make sure there is something to plot or else just give up and return
    npipes = get_channel_count(
        ods, 'gas_injection', check_loc='gas_injection.pipe.0.exit_position.r', channels_name='pipe', test_checker='~numpy.isnan(checker)'
    )

    if npipes == 0:
        return {'ax': ax}

    mask = kw.pop('mask', numpy.ones(npipes, bool))

    pipes = ods['gas_injection']['pipe']  # Shortcut

    # Identify gas injectors with the same poloidal location and group them so that their labels won't overlap.
    locations = {}
    for i in pipes:
        if mask[i]:
            pipe = pipes[i]
            label = pipe['name']
            if not gas_filter(label, which_gas):
                continue  # Skip this valve because it's not active

            r, z = pipe['exit_position']['r'], pipe['exit_position']['z']
            location_name = f'{r:0.3f}_{z:0.3f}'

            if simple_labels:
                label = '_'.join(label.split('_')[:-1])

            locations.setdefault(location_name, [])
            locations[location_name] += [label]

            if angle_not_in_pipe_name:
                try:
                    label += ' ({:0d})'.format(int(round(pipe['exit_position']['phi'] * 180 / numpy.pi)))
                except (TypeError, ValueError):
                    pass
            try:
                r2, z2 = pipe['second_point']['r'], pipe['second_point']['z']
            except (LookupError, ValueError):
                if len(locations[location_name]) > 3:
                    # If an item has already been added at this location, use its r2, z2 to fill in missing values
                    r2 = locations[location_name][-3]
                    z2 = locations[location_name][-2]
                else:
                    r2 = z2 = None
            locations[location_name] += [r2, z2]
    try:
        rsplit = ods['equilibrium.time_slice'][0]['global_quantities.magnetic_axis.r']
    except ValueError:
        draw_arrow = False  # This won't work without magnetic axis data, either.
        rsplit = numpy.mean([float(loc.split('_')[0]) for loc in locations])

    kw.setdefault('marker', 'd')
    kw.setdefault('linestyle', ' ')
    labelevery = kw.pop('labelevery', 1)
    notesize = kw.pop('notesize', 'xx-small')
    default_ha = [['left', 'right'][int(float(loc.split('_')[0]) < rsplit)] for loc in locations]
    default_va = [['top', 'bottom'][int(float(loc.split('_')[1]) > 0)] for loc in locations]
    label_ha, label_va, kw = text_alignment_setup(len(locations), default_ha=default_ha, default_va=default_va, **kw)
    label_dr, label_dz = label_shifter(len(locations), kw)

    # For each unique poloidal location, draw a marker and write a label describing all the injectors at this location.
    default_color = kw.pop('color', None)
    colors = numpy.atleast_1d(default_color if colors is None else colors).tolist()
    colors2 = colors * int(numpy.ceil(len(locations) / float(len(colors))))  # Make sure the list is long enough.
    for i, loc in enumerate(locations):
        r, z = numpy.array(loc.split('_')).astype(float)
        if show_all_pipes_in_group:
            show_locs = list(set(locations[loc][::3]))  # Each pipe has ['label', r2, z2], so ::3 selects just labels.
        else:
            show_locs = [locations[loc][0]]
        label = '{spacer:}\n{spacer:}'.format(spacer=' ' * label_spacer).join([''] + show_locs + [''])
        if draw_arrow:
            kw.update(draw_arrow if isinstance(draw_arrow, dict) else {})
            gas_mark = gas_arrow(ods, r, z, r2=locations[loc][-2], z2=locations[loc][-1], ax=ax, color=colors2[i], **kw)
        else:
            gas_mark = ax.plot(r, z, color=colors2[i], **kw)
        kw.pop('label', None)  # Prevent label from being applied every time through the loop to avoid spammy legend
        if (labelevery > 0) and ((i % labelevery) == 0):
            label = '\n' * label_spacer + label if label_va[i] == 'top' else label + '\n' * label_spacer
            ax.text(
                r + label_dr[i], z + label_dz[i], label, color=gas_mark[0].get_color(), va=label_va[i], ha=label_ha[i], fontsize=notesize
            )

    return {'ax': ax}


@add_to__ODS__
def pf_active_overlay(ods, ax=None, **kw):
    r"""
    Plots overlays of active PF coils.
    INCOMPLETE: only the oblique geometry definition is treated so far. More should be added later.

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param \**kw: Additional keywords
        scalex, scaley: passed to ax.autoscale_view() call at the end

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Remaining keywords are passed to matplotlib.patches.Polygon call
            Hint: you may want to set facecolor instead of just color
    """
    import matplotlib
    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    nc = get_channel_count(
        ods, 'pf_active', check_loc='pf_active.coil.0.element.0.geometry.geometry_type', channels_name='coil', test_checker='checker > -1'
    )
    if nc == 0:
        return {'ax': ax}

    kw.setdefault('label', 'Active PF coils')
    kw.setdefault('facecolor', 'gray')
    kw.setdefault('edgecolor', 'k')
    kw.setdefault('alpha', 0.7)
    labelevery = kw.pop('labelevery', 0)
    notesize = kw.pop('notesize', 'xx-small')
    mask = kw.pop('mask', numpy.ones(nc, bool))
    scalex, scaley = kw.pop('scalex', True), kw.pop('scaley', True)
    label_ha, label_va, kw = text_alignment_setup(nc, default_ha='center', default_va='center', **kw)
    label_dr, label_dz = label_shifter(nc, kw)

    def path_rectangle(rectangle):
        """
        :param rectangle: ODS sub-folder: element.*.geometry.rectangle

        :return: n x 2 array giving the path around the outline of the coil element, suitable for input to Polygon()
        """
        x = rectangle['r']
        y = rectangle['z']
        dx = rectangle['width']
        dy = rectangle['height']
        return numpy.array(
            [[x - dx / 2.0, x - dx / 2.0, x + dx / 2.0, x + dx / 2.0], [y - dy / 2.0, y + dy / 2.0, y + dy / 2.0, y - dy / 2.0]]
        ).T

    def path_outline(outline):
        """
        :param outline: ODS sub-folder: element.*.geometry.outline

        :return: n x 2 array giving the path around the outline of the coil element, suitable for input to Polygon()
        """
        return numpy.array([outline['r'], outline['z']]).T

    patches = []
    for i in range(nc):  # From  iris:/fusion/usc/src/idl/efitview/diagnoses/DIII-D/coils.pro ,  2018 June 08  D. Eldon
        if mask[i]:
            try:
                geometry_type = geo_type_lookup(ods['pf_active.coil'][i]['element.0.geometry.geometry_type'], 'pf_active', ods.imas_version)
            except (IndexError, ValueError):
                geometry_type = 'unrecognized'
            try:
                path = eval('path_{}'.format(geometry_type))(ods['pf_active.coil'][i]['element.0.geometry'][geometry_type])
            except NameError:
                print('Warning: unrecognized geometry type for pf_active coil {}: {}'.format(i, geometry_type))
                continue
            patches.append(matplotlib.patches.Polygon(path, closed=True, **kw))
            kw.pop('label', None)  # Prevent label from being placed on more than one patch
            try:
                pf_id = ods['pf_active.coil'][i]['element.0.identifier']
            except ValueError:
                pf_id = None
            if (labelevery > 0) and ((i % labelevery) == 0) and (pf_id is not None):
                ax.text(
                    numpy.mean(path[:, 0]) + label_dr[i],
                    numpy.mean(path[:, 1]) + label_dz[i],
                    pf_id,
                    ha=label_ha[i],
                    va=label_va[i],
                    fontsize=notesize,
                )

    for p in patches:
        ax.add_patch(p)  # Using patch collection breaks auto legend labeling, so add patches individually.

    ax.autoscale_view(scalex=scalex, scaley=scaley)  # add_patch doesn't include this
    ax.set_aspect('equal')

    return {'ax': ax}


@add_to__ODS__
def magnetics_overlay(
    ods,
    ax=None,
    show_flux_loop=True,
    show_bpol_probe=True,
    show_btor_probe=True,
    flux_loop_style={'marker': 's'},
    pol_probe_style={},
    tor_probe_style={'marker': '.'},
    **kw,
):
    '''
    Plot magnetics on a tokamak cross section plot

    :param ods: OMAS ODS instance

    :param flux_loop_style: dictionary with matplotlib options to render flux loops

    :param pol_probe_style: dictionary with matplotlib options to render poloidal magnetic probes

    :param tor_probe_style: dictionary with matplotlib options to render toroidal magnetic probes

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :return: axes handler
    '''
    from matplotlib import pyplot

    kw0 = copy.copy(kw)

    if ax is None:
        ax = pyplot.gca()

    # flux loops
    nfl = get_channel_count(
        ods, 'magnetics', check_loc='magnetics.flux_loop.0.position.0.r', channels_name='flux_loop', test_checker='~numpy.isnan(checker)'
    )
    if show_flux_loop and nfl:
        kw = copy.copy(kw0)
        labelevery = kw.pop('labelevery', 0)
        notesize = kw.pop('notesize', 'xx-small')
        label_ha, label_va, kw = text_alignment_setup(nfl, **kw)
        label_dr, label_dz = label_shifter(nfl, kw)

        for k, (r, z) in enumerate(zip(ods[f'magnetics.flux_loop.:.position[0].r'], ods[f'magnetics.flux_loop.:.position[0].z'])):
            ax.plot(r, z, **flux_loop_style)
            flux_loop_style.setdefault('color', ax.lines[-1].get_color())
            if (labelevery > 0) and ((k % labelevery) == 0):
                ax.text(
                    r + label_dr[k],
                    z + label_dz[k],
                    ods.get(f'magnetics.flux_loop.{k}.identifier', str(k)),
                    color=flux_loop_style['color'],
                    fontsize=notesize,
                    ha=label_ha[k],
                    va=label_va[k],
                )

    # poloidal magnetic probes
    nbp = get_channel_count(
        ods,
        'magnetics',
        check_loc='magnetics.b_field_pol_probe.0.position.r',
        channels_name='b_field_pol_probe',
        test_checker='~numpy.isnan(checker)',
    )
    if show_bpol_probe and nbp:
        kw = copy.copy(kw0)
        labelevery = kw.pop('labelevery', 0)
        notesize = kw.pop('notesize', 'xx-small')
        label_ha, label_va, kw = text_alignment_setup(nbp, **kw)
        label_dr, label_dz = label_shifter(nbp, kw)

        from .omas_physics import probe_endpoints

        PX, PY = probe_endpoints(
            ods['magnetics.b_field_pol_probe[:].position.r'],
            ods['magnetics.b_field_pol_probe[:].position.z'],
            ods['magnetics.b_field_pol_probe[:].poloidal_angle'],
            ods['magnetics.b_field_pol_probe[:].length'],
            ods.cocosio,
        )

        for k, (px, py) in enumerate(zip(PX, PY)):
            r = numpy.mean(px)
            z = numpy.mean(py)
            if show_bpol_probe:
                ax.plot(px, py, label='_' + ods.get(f'magnetics.b_field_pol_probe[{k}].identifier', str(k)), **pol_probe_style, **kw)
                pol_probe_style.setdefault('color', ax.lines[-1].get_color())
                if (labelevery > 0) and ((k % labelevery) == 0):
                    ax.text(
                        r + label_dr[k],
                        z + label_dz[k],
                        ods.get(f'magnetics.b_field_pol_probe[{k}].identifier', str(k)),
                        color=pol_probe_style['color'],
                        fontsize=notesize,
                        ha=label_ha[k],
                        va=label_va[k],
                    )

    # toroidal magnetic probes
    nbt = get_channel_count(
        ods,
        'magnetics',
        check_loc='magnetics.b_field_tor_probe.0.position.r',
        channels_name='b_field_tor_probe',
        test_checker='~numpy.isnan(checker)',
    )
    if show_btor_probe and nbt:
        kw = copy.copy(kw0)
        labelevery = kw.pop('labelevery', 0)
        notesize = kw.pop('notesize', 'xx-small')
        label_ha, label_va, kw = text_alignment_setup(nbt, **kw)
        label_dr, label_dz = label_shifter(nbt, kw)
        for k, (r, z) in enumerate(zip(ods['magnetics.b_field_tor_probe[:].position.r'], ods['magnetics.b_field_tor_probe[:].position.z'])):
            ax.plot(r, z, '.m', label='_' + ods.get(f'magnetics.b_field_tor_probe[{k}].identifier', str(k)), **tor_probe_style, **kw)
            tor_probe_style.setdefault('color', ax.lines[-1].get_color())
            if (labelevery > 0) and ((k % labelevery) == 0):
                ax.text(
                    r + label_dr[k],
                    z + label_dz[k],
                    ods.get(f'magnetics.b_field_tor_probe[{k}].identifier', str(k)),
                    color=tor_probe_style['color'],
                    fontsize=notesize,
                    ha=label_ha[k],
                    va=label_va[k],
                )

    ax.set_aspect('equal')
    return {'ax': ax}


@add_to__ODS__
def interferometer_overlay(ods, ax=None, **kw):
    r"""
    Plots overlays of interferometer chords.

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param \**kw: Additional keywords

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Remaining keywords are passed to plot call
    """

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'interferometer', check_loc='interferometer.channel.0.line_of_sight.first_point.r', test_checker='~numpy.isnan(checker)'
    )
    if nc == 0:
        return {'ax': ax}

    color = kw.pop('color', None)
    labelevery = kw.pop('labelevery', 1)
    mask = kw.pop('mask', numpy.ones(nc, bool))
    notesize = kw.pop('notesize', 'medium')
    label_ha, label_va, kw = text_alignment_setup(nc, default_ha='left', default_va='top', **kw)
    label_dr, label_dz = label_shifter(nc, kw)

    j = 0
    for i in range(nc):
        if mask[i]:
            ch = ods['interferometer.channel'][i]
            los = ch['line_of_sight']
            r1, z1, r2, z2 = los['first_point.r'], los['first_point.z'], los['second_point.r'], los['second_point.z']
            line = ax.plot([r1, r2], [z1, z2], color=color, label='interferometer' if i == 0 else '', **kw)
            color = line[0].get_color()  # If this was None before, the cycler will have given us something. Lock it in.
            if (labelevery > 0) and ((i % labelevery) == 0):
                ax.text(
                    max([r1, r2]) + label_dr[j],
                    min([z1, z2]) + label_dz[j],
                    ch['identifier'],
                    color=color,
                    va=label_va[i],
                    ha=label_ha[i],
                    fontsize=notesize,
                )
            j += 1
    return {'ax': ax}


@add_to__ODS__
def thomson_scattering_overlay(ods, ax=None, **kw):
    r"""
    Overlays Thomson channel locations

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param \**kw: Additional keywords for Thomson plot:

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Remaining keywords are passed to plot call
    """

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'thomson_scattering', check_loc='thomson_scattering.channel.0.position.r', test_checker='~numpy.isnan(checker)'
    )
    if nc == 0:
        return {'ax': ax}

    labelevery = kw.pop('labelevery', 5)
    notesize = kw.pop('notesize', 'xx-small')
    mask = kw.pop('mask', numpy.ones(nc, bool))
    kw.setdefault('marker', '+')
    kw.setdefault('label', 'Thomson scattering')
    kw.setdefault('linestyle', ' ')
    label_ha, label_va, kw = text_alignment_setup(nc, **kw)
    label_dr, label_dz = label_shifter(nc, kw)

    r = numpy.array([ods['thomson_scattering']['channel'][i]['position']['r'] for i in range(nc)])[mask]
    z = numpy.array([ods['thomson_scattering']['channel'][i]['position']['z'] for i in range(nc)])[mask]
    ts_id = numpy.array([ods['thomson_scattering']['channel'][i]['identifier'] for i in range(nc)])[mask]

    ts_mark = ax.plot(r, z, **kw)
    for i in range(sum(mask)):
        if (labelevery > 0) and ((i % labelevery) == 0):
            ax.text(
                r[i] + label_dr[i],
                z[i] + label_dz[i],
                ts_id[i],
                color=ts_mark[0].get_color(),
                fontsize=notesize,
                ha=label_ha[i],
                va=label_va[i],
            )

    return {'ax': ax}


@add_to__ODS__
def charge_exchange_overlay(ods, ax=None, which_pos='closest', **kw):
    r"""
    Overlays Charge Exchange Recombination (CER) spectroscopy channel locations

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param which_pos: string
        'all': plot all valid positions this channel uses. This can vary in time depending on which beams are on.

        'closest': for each channel, pick the time slice with valid data closest to the time used for the
            equilibrium contours and show position at this time. Falls back to all if equilibrium time cannot be
            read from time_slice 0 of equilibrium in the ODS.

    :param \**kw: Additional keywords for CER plot:

        color_tangential: color to use for tangentially-viewing channels

        color_vertical: color to use for vertically-viewing channels

        color_radial: color to use for radially-viewing channels

        marker_tangential, marker_vertical, marker_radial: plot symbols to use for T, V, R viewing channels

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Remaining keywords are passed to plot call
    """

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'charge_exchange', check_loc='charge_exchange.channel.0.position.r.data', test_checker='any(~numpy.isnan(checker))'
    )
    if nc == 0:
        return {'ax': ax}

    try:
        eq_time = ods['equilibrium.time_slice.0.time']
    except ValueError:
        eq_time = None

    # Resolve keywords
    mask = kw.pop('mask', numpy.ones(nc, bool))
    labelevery = kw.pop('labelevery', 5)
    if eq_time is None:
        which_pos = 'all'
    colors = {}
    for colorkw in ['color_tangential', 'color_vertical', 'color_radial']:
        ckw = kw.pop(colorkw, kw.get('color', None))
        if ckw is not None:
            colors[colorkw.split('_')[-1][0].upper()] = ckw
    kw.pop('color', None)
    marker = kw.pop('marker', None)
    markers = {
        'T': kw.pop('marker_tangential', 's' if marker is None else marker),
        'V': kw.pop('marker_vertical', 'd' if marker is None else marker),
        'R': kw.pop('marker_radial', '*' if marker is None else marker),
    }
    notesize = kw.pop('notesize', 'xx-small')
    ha, va, kw = text_alignment_setup(nc, **kw)
    label_dr, label_dz = label_shifter(nc, kw)

    # Get channel positions; each channel has a list of positions as it can vary with time as beams switch on/off.
    r = [[numpy.NaN]] * nc
    z = [[numpy.NaN]] * nc
    for i in range(nc):
        rs = ods['charge_exchange.channel'][i]['position.r.data']
        zs = ods['charge_exchange.channel'][i]['position.z.data']
        w = (rs > 0) & (~numpy.isnan(rs)) & (~numpy.isnan(zs))  # Validity mask: remove zero and NaN
        ts = ods['charge_exchange.channel'][i]['position.r.time'][w]
        rs = rs[w]
        zs = zs[w]
        if which_pos == 'all':  # Show the set of all valid positions measured by this channel.
            rz = list(set(zip(rs, zs)))
            r[i] = [rz[j][0] for j in range(len(rz))]
            z[i] = [rz[j][1] for j in range(len(rz))]
        else:  # 'closest': pick just the closest time. The list of positions will only have one element.
            w = closest_index(ts, eq_time)
            r[i] = [rs[w]]
            z[i] = [zs[w]]
    cer_id = numpy.array([ods['charge_exchange.channel'][i]['identifier'] for i in range(nc)])

    # Plot
    label_bank = {'T': 'Tang. CER', 'V': 'Vert. CER', 'R': 'Rad. CER'}  # These get popped so only one each in legend
    j = 0
    for i in range(nc):
        if mask[i]:
            ch_type = cer_id[i][0].upper()
            color = colors.get(ch_type, None)  # See if a color has been specified for this view direction
            cer_mark = ax.plot(
                r[i], z[i], marker=markers.get(ch_type, 'x'), linestyle=' ', color=color, label=label_bank.pop(ch_type, ''), **kw
            )
            colors[ch_type] = color = cer_mark[0].get_color()  # Save color for this view dir in case it was None
            if (labelevery > 0) and ((i % labelevery) == 0):
                ax.text(
                    numpy.mean(r[i]) + label_dr[j],
                    numpy.mean(z[i]) + label_dz[j],
                    cer_id[i],
                    color=color,
                    fontsize=notesize,
                    ha=ha[i],
                    va=va[i],
                )
        j += 1
    return {'ax': ax}


@add_to__ODS__
def bolometer_overlay(ods, ax=None, reset_fan_color=True, colors=None, **kw):
    r"""
    Overlays bolometer chords

    :param ods: ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param reset_fan_color: bool
        At the start of each bolometer fan (group of channels), set color to None to let a new one be picked by the
        cycler. This will override manually specified color.

    :param colors: list of matplotlib color specifications. Do not use a single RGBA style spec.

    :param \**kw: Additional keywords for bolometer plot

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Remaining keywords are passed to plot call for drawing lines for the bolometer sightlines
    """

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'bolometer', check_loc='bolometer.channel.0.line_of_sight.first_point.r', test_checker='~numpy.isnan(checker)'
    )
    if nc == 0:
        return {'ax': ax}

    mask = kw.pop('mask', numpy.ones(nc, bool))

    r1 = ods['bolometer.channel.:.line_of_sight.first_point.r'][mask]
    z1 = ods['bolometer.channel.:.line_of_sight.first_point.z'][mask]
    r2 = ods['bolometer.channel.:.line_of_sight.second_point.r'][mask]
    z2 = ods['bolometer.channel.:.line_of_sight.second_point.z'][mask]
    bolo_id = ods['bolometer.channel.:.identifier'][mask]

    ncm = len(r1)

    if colors is None:
        colors = [kw.pop('color', None)]
    ci = 0
    colors2 = colors * nc
    color = colors2[ci]  # Multiplying list by nc makes sure it's always long enough.
    kw.setdefault('alpha', 0.8)
    default_label = kw.pop('label', None)
    labelevery = kw.pop('labelevery', 2)
    notesize = kw.pop('notesize', 'xx-small')
    default_ha = [['right', 'left'][int(z1[i] > 0)] for i in range(ncm)]
    label_ha, label_va, kw = text_alignment_setup(ncm, default_ha=default_ha, default_va='top', **kw)
    label_dr, label_dz = label_shifter(ncm, kw)

    for i in range(ncm):
        if (i > 0) and (bolo_id[i][0] != bolo_id[i - 1][0]) and reset_fan_color:
            ci += 1
            color = colors2[ci]  # Allow color to reset when changing fans
            new_label = True
        else:
            new_label = False

        label = 'Bolometers {}'.format(bolo_id[i][0]) if default_label is None else default_label
        bolo_line = ax.plot([r1[i], r2[i]], [z1[i], z2[i]], color=color, label=label if new_label or (i == 0) else '', **kw)
        if color is None:
            color = bolo_line[0].get_color()  # Make subsequent lines the same color
        if (labelevery > 0) and ((i % labelevery) == 0):
            ax.text(
                r2[i] + label_dr[i],
                z2[i] + label_dz[i],
                '{}{}'.format(['\n', ''][int(z1[i] > 0)], bolo_id[i]),
                color=color,
                ha=label_ha[i],
                va=label_va[i],
                fontsize=notesize,
            )

    return {'ax': ax}


@add_to__ODS__
def langmuir_probes_overlay(ods, ax=None, embedded_probes=None, colors=None, show_embedded=True, show_reciprocating=False, **kw):
    r"""
    Overlays Langmuir probe locations

    :param ods: ODS instance
        Must contain langmuir_probes with embedded position data

    :param ax: Axes instance

    :param embedded_probes: list of strings
        Specify probe names to use. Only the embedded probes listed will be plotted. Set to None to plot all probes.
        Probe names are like 'F11' or 'P-6' (the same as appear on the overlay).

    :param colors: list of matplotlib color specifications. Do not use a single RGBA style spec.

    :param show_embedded: bool
        Recommended: don't enable both embedded and reciprocating plots at the same time; make two calls instead.
        It will be easier to handle mapping of masks, colors, etc.

    :param show_reciprocating: bool

    :param \**kw: Additional keywords.

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Others will be passed to the plot() call for drawing the probes.
    """
    from matplotlib import pyplot

    # Get a handle on the axes
    if ax is None:
        ax = pyplot.gca()

    # Make sure there is something to plot or else just give up and return
    if show_embedded:
        if embedded_probes is not None:
            embedded_probes = numpy.atleast_1d(embedded_probes)
            embedded_indices = []

            for probe in ods['langmuir_probes.embedded']:
                if ods['langmuir_probes.embedded'][probe]['name'] in embedded_probes:
                    embedded_indices += [probe]
            nce = len(embedded_indices)
        else:
            nce = get_channel_count(
                ods,
                'langmuir_probes',
                check_loc='langmuir_probes.embedded.0.position.r',
                test_checker='~numpy.isnan(checker)',
                channels_name='embedded',
            )
            embedded_indices = range(nce)
    else:
        nce = 0
        embedded_indices = []
    if show_reciprocating:
        ncr = get_channel_count(
            ods,
            'langmuir_probes',
            check_loc='langmuir_probes.reciprocating.0.plunge.0.position.r',
            test_checker='~numpy.isnan(checker)',
            channels_name='reciprocating',
        )
    else:
        ncr = 0
    if (nce == 0) and (ncr == 0):
        return {'ax': ax}

    # Set up masks
    mask = kw.pop('mask', numpy.ones(nce + ncr, bool))
    mask_e = mask[:nce]  # For wall-embedded probes
    # mask_r = mask[nce:]  # For reciprocating probes
    if ncr > 0:
        raise NotImplementedError('Reciprocating Langmuir probe overlay plots are not ready yet. Try embedded LPs.')

    # Get embedded data
    r_e = numpy.array([ods['langmuir_probes.embedded'][i]['position.r'] for i in embedded_indices])[mask_e]
    z_e = numpy.array([ods['langmuir_probes.embedded'][i]['position.z'] for i in embedded_indices])[mask_e]
    lp_id_e = numpy.array([ods['langmuir_probes.embedded'][i]['name'] for i in embedded_indices])[mask_e]
    ncem = len(r_e)  # Number of Channels, Embedded, Masked

    # Get reciprocating data
    ncrm = 0  # Coming soon

    nc = ncem + ncem

    # Handle plot keywords
    if colors is None:
        colors = [kw.pop('color', None)]
    ci = 0
    color = (colors * nc)[ci]  # Multiplying list by nc makes sure it's always long enough.
    kw.setdefault('alpha', 0.8)
    kw.setdefault('marker', '*')
    kw.setdefault('linestyle', ' ')
    default_label = kw.pop('label', None)
    labelevery = kw.pop('labelevery', 2)
    notesize = kw.pop('notesize', 'xx-small')
    label_dr, label_dz = label_shifter(ncem, kw)

    # Decide which side each probe is on, for aligning annotation labels
    ha = ['center'] * ncem
    va = ['center'] * ncem
    try:
        wall_r = ods['wall.description_2d[0].limiter.unit[0].outline.r']
        wall_z = ods['wall.description_2d[0].limiter.unit[0].outline.z']
    except (KeyError, ValueError):
        va = ['bottom' if z_e[i] > 0 else 'top' for i in range(ncem)]
    else:
        wr0 = numpy.min(wall_r)
        wr1 = numpy.max(wall_r)
        dr = wr1 - wr0
        wz0 = numpy.min(wall_z)
        wz1 = numpy.max(wall_z)
        dz = wz1 - wz0
        lr_margin = 0.2
        tb_margin = 0.1
        right = wr0 + dr * (1 - lr_margin)
        left = wr0 + dr * lr_margin
        top = wz0 + dz * (1 - tb_margin)
        bottom = wz0 + dz * tb_margin
        for i in range(ncem):
            if z_e[i] > top:
                va[i] = 'bottom'
            elif z_e[i] < bottom:
                va[i] = 'top'
            if r_e[i] > right:
                ha[i] = 'left'
            elif r_e[i] < left:
                ha[i] = 'right'

    ha, va, kw = text_alignment_setup(ncem, default_ha=ha, default_va=va, **kw)

    # Plot
    for i in range(ncem):
        label = 'Embedded Langmuir probes' if default_label is None else default_label
        lp_mark = ax.plot(r_e[i], z_e[i], color=color, label=label if i == 0 else '', **kw)
        if color is None:
            color = lp_mark[0].get_color()  # Make subsequent marks the same color
        if (labelevery > 0) and ((i % labelevery) == 0):
            ax.text(
                r_e[i] + label_dr[i],
                z_e[i] + label_dz[i],
                '\n {} \n'.format(lp_id_e[i]),
                color=color,
                ha=ha[i],
                va=va[i],
                fontsize=notesize,
            )

    return {'ax': ax}


@add_to__ODS__
def position_control_overlay(
    ods, ax=None, t=None, xpoint_marker='x', strike_marker='s', labels=None, measured_xpoint_marker='+', show_measured_xpoint=False, **kw
):
    r"""
    Overlays position_control data

    :param ods: ODS instance
        Must contain langmuir_probes with embedded position data

    :param ax: Axes instance

    :param t: float
        Time to display in seconds. If not specified, defaults to the average time of all boundary R coordinate samples.

    :param xpoint_marker: string
        Matplotlib marker spec for X-point target(s)

    :param strike_marker: string
        Matplotlib marker spec for strike point target(s)

    :param labels: list of strings [optional]
        Override default point labels. Length must be long enough to cover all points.

    :param show_measured_xpoint: bool
        In addition to the target X-point, mark the measured X-point coordinates.

    :param measured_xpoint_marker: string
        Matplotlib marker spec for X-point measurement(s)

    :param \**kw: Additional keywords.

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Others will be passed to the plot() call for drawing shape control targets
    """
    import numpy as np
    from matplotlib import pyplot
    from matplotlib import rcParams
    from scipy.interpolate import interp1d
    import time

    timing_ref = kw.pop('timing_ref', None)
    if timing_ref is not None:
        print(time.time() - timing_ref, 'position_control_overlay start')

    # Unpack basics
    device = ods['dataset_description.data_entry'].get('machine', '')
    shot = ods['dataset_description.data_entry'].get('pulse', 0)
    if t is None:
        try:
            t = np.nanmean(ods['pulse_schedule.position_control.boundary_outline[:].r.reference.data'])
        except (ValueError, IndexError):
            t = 0

    if ax is None:
        ax = pyplot.gca()

    # Handle multi-slice request
    if timing_ref is not None:
        print(time.time() - timing_ref, 'position_control_overlay setup 1')
    if len(np.atleast_1d(t)) > 1:
        for tt in t:
            position_control_overlay(
                ods,
                ax=ax,
                t=tt,
                xpoint_marker=xpoint_marker,
                strike_marker=strike_marker,
                show_measured_xpoint=show_measured_xpoint,
                **copy.deepcopy(kw),
            )
        return {'ax': ax}
    else:
        t = np.atleast_1d(t)[0]

    labelevery = kw.pop('labelevery', 1)
    label_ha = kw.pop('label_ha', None)
    label_va = kw.pop('label_va', None)
    notesize = kw.pop('notesize', 'xx-small')

    if timing_ref is not None:
        print(time.time() - timing_ref, 'position_control_overlay setup 2')

    # Select data
    b = ods['pulse_schedule.position_control.boundary_outline']
    x = ods['pulse_schedule.position_control.x_point']
    s = ods['pulse_schedule.position_control.strike_point']
    ikw = dict(bounds_error=False, fill_value=np.NaN)
    try:
        nbp = np.shape(b['[:].r.reference.data'])[0]
    except (IndexError, ValueError):
        nbp = 0
    try:
        nx = np.shape(x['[:].r.reference.data'])[0]
    except (IndexError, ValueError):
        nx = 0
    try:
        ns = np.shape(s['[:].r.reference.data'])[0]
    except (IndexError, ValueError):
        ns = 0
    if nbp + nx + ns == 0:
        printe('Trouble accessing position_control data in ODS. Aborting plot overlay.')
        return {'ax': ax}
    r = [interp1d(b[i]['r.reference.time'], b[i]['r.reference.data'], **ikw)(t) for i in range(nbp)]
    z = [interp1d(b[i]['z.reference.time'], b[i]['z.reference.data'], **ikw)(t) for i in range(nbp)]
    bname = b['[:].r.reference_name']
    rx = [interp1d(x[i]['r.reference.time'], x[i]['r.reference.data'], **ikw)(t) for i in range(nx)]
    zx = [interp1d(x[i]['z.reference.time'], x[i]['z.reference.data'], **ikw)(t) for i in range(nx)]
    xname = x['[:].r.reference_name']
    rs = [interp1d(s[i]['r.reference.time'], s[i]['r.reference.data'], **ikw)(t) for i in range(ns)]
    zs = [interp1d(s[i]['z.reference.time'], s[i]['z.reference.data'], **ikw)(t) for i in range(ns)]
    sname = s['[:].r.reference_name']
    # Measured X-point position from eq might not be present
    nxm = len(ods['equilibrium.time_slice.0.boundary.x_point'])
    if nxm > 0:
        eq = ods['equilibrium']
        if len(eq['time']) == 1:
            it = eq['time_slice'].keys()[0]
            rxm = [eq['time_slice'][it]['boundary.x_point'][i]['r'] for i in range(nxm)]
            zxm = [eq['time_slice'][it]['boundary.x_point'][i]['z'] for i in range(nxm)]
        else:
            rxm = [interp1d(eq['time'], eq['time_slice[:].boundary.x_point.{}.r'.format(i)], **ikw)(t) for i in range(nxm)]
            zxm = [interp1d(eq['time'], eq['time_slice[:].boundary.x_point.{}.z'.format(i)], **ikw)(t) for i in range(nxm)]
    else:
        rxm = zxm = np.NaN
    if timing_ref is not None:
        print(time.time() - timing_ref, 'position_control_overlay data unpacked')

    # Masking
    mask = np.array(kw.pop('mask', np.ones(nbp + nx + ns, bool)))
    # Extend mask to make correct length, if needed
    if len(mask) < (nbp + nx + ns):
        extra_mask = np.ones(nbp + nx + ns - len(mask), bool)
        mask = np.append(mask, extra_mask)
    maskb = mask[:nbp]
    maskx = mask[nbp : nbp + nx]
    masks = mask[nbp + nx : nbp + nx + ns]
    r = (np.array(r)[maskb]).tolist()
    z = (np.array(z)[maskb]).tolist()
    bname = (np.array(bname)[maskb]).tolist()
    rx = (np.array(rx)[maskx]).tolist()
    zx = (np.array(zx)[maskx]).tolist()
    xname = (np.array(xname)[maskx]).tolist()
    rs = (np.array(rs)[masks]).tolist()
    zs = (np.array(zs)[masks]).tolist()
    sname = (np.array(sname)[masks]).tolist()
    mnbp = len(r)
    mnx = len(rx)
    mns = len(rs)

    label_dr, label_dz = label_shifter(mnbp + mnx + mns, kw)

    # Handle main plot setup and customizations
    kw.setdefault('linestyle', ' ')
    kwx = copy.deepcopy(kw)
    kws = copy.deepcopy(kw)
    kw.setdefault('marker', 'o')
    plot_out = ax.plot(r, z, **kw)

    kwx.setdefault('markersize', rcParams['lines.markersize'] * 1.5)
    if show_measured_xpoint:
        kwxm = copy.deepcopy(kwx)
        kwxm.setdefault('marker', measured_xpoint_marker)
        xmplot_out = ax.plot(rxm, zxm, **kwxm)
    else:
        xmplot_out = None
    kwx['marker'] = xpoint_marker
    kwx.setdefault('mew', rcParams['lines.markeredgewidth'] * 1.25 + 1.25)
    kwx['color'] = plot_out[0].get_color()
    xplot_out = ax.plot(rx, zx, **kwx)

    kws['marker'] = strike_marker
    kws['color'] = plot_out[0].get_color()
    splot_out = ax.plot(rs, zs, **kws)

    if timing_ref is not None:
        print(time.time() - timing_ref, 'position_control_overlay main plots')

    # Handle plot annotations
    try:
        rsplit = ods['equilibrium.time_slice'][0]['global_quantities.magnetic_axis.r']
    except ValueError:
        # Guesses for a good place to split labels between left and right align
        r0 = {'DIII-D': 1.6955}
        rsplit = r0.get(device, 1.7)

    default_ha = [['left', 'right'][int((r + rx + rs)[i] < rsplit)] for i in range(mnbp + mnx + mns)]
    default_va = [['top', 'bottom'][int((z + zx + rs)[i] > 0)] for i in range(mnbp + mnx + mns)]
    label_ha, label_va, kw = text_alignment_setup(
        mnbp + mnx + mns, default_ha=default_ha, default_va=default_va, label_ha=label_ha, label_va=label_va
    )

    if labels is None:
        labels = bname + xname + sname

    for i in range(mnbp):
        if (labelevery > 0) and ((i % labelevery) == 0) and ~np.isnan(r[i]):
            ax.text(
                r[i] + label_dr[i],
                z[i] + label_dz[i],
                '\n {} \n'.format(labels[i]),
                color=plot_out[0].get_color(),
                va=label_va[i],
                ha=label_ha[i],
                fontsize=notesize,
            )
    for i in range(mnx):
        if (labelevery > 0) and ((i % labelevery) == 0) and ~np.isnan(rx[i]):
            ax.text(
                rx[i] + label_dr[i],
                zx[i] + label_dz[i],
                '\n {} \n'.format(labels[mnbp + i]),
                color=xplot_out[0].get_color(),
                va=label_va[mnbp + i],
                ha=label_ha[mnbp + i],
                fontsize=notesize,
            )

    for i in range(mns):
        if (labelevery > 0) and ((i % labelevery) == 0) and ~np.isnan(rs[i]):
            ax.text(
                rs[i] + label_dr[i],
                zs[i] + label_dz[i],
                '\n {} \n'.format(labels[mnbp + mnx + i]),
                color=splot_out[0].get_color(),
                va=label_va[mnbp + mnx + i],
                ha=label_ha[mnbp + mnx + i],
                fontsize=notesize,
            )

    if timing_ref is not None:
        print(time.time() - timing_ref, 'position_control_overlay done')

    return {'ax': ax}


@add_to__ODS__
def pulse_schedule_overlay(ods, ax=None, t=None, **kw):
    r"""
    Overlays relevant data from pulse_schedule, such as position control

    :param ods: ODS instance
        Must contain langmuir_probes with embedded position data

    :param ax: Axes instance

    :param t: float
        Time in s

    :param \**kw: Additional keywords.

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Others will be passed to the plot() calls.
    """

    from matplotlib import pyplot
    import time

    if kw.get('timing_ref', None) is not None:
        print(time.time() - kw['timing_ref'], 'pulse_schedule_overlay start')

    if ax is None:
        ax = pyplot.gca()

    position_control_overlay(ods, ax=ax, t=t, **kw)
    return {'ax': ax}


@add_to__ODS__
def summary(ods, fig=None, quantity=None, **kw):
    """
    Plot summary time traces. Internally makes use of plot_quantity method.

    :param ods: input ods

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param quantity: if None plot all time-dependent global_quantities. Else a list of strings with global quantities to plot

    :return: list of axes
    """

    from matplotlib import pyplot

    if quantity is None:
        quantity = ods['summary.global_quantities']

    axs = kw.pop('ax', {})
    if axs is None:
        axs = {}
    if not len(axs) and fig is None:
        fig = pyplot.figure()

    # two passes, one for counting number of plots the second for actual plotting
    n = 0
    for step in ['count', 'plot']:
        k = 0
        for q in quantity:
            if 'value' in ods['summary.global_quantities'][q] and isinstance(ods['summary.global_quantities'][q]['value'], numpy.ndarray):
                if step == 'count':
                    n += 1
                k += 1
                if step == 'plot':
                    r = int(numpy.sqrt(n + 1))
                    c = int(numpy.ceil(n / numpy.sqrt(n)))
                    if k == 1:
                        ax = ax0 = cached_add_subplot(fig, axs, r, c, k)
                    else:
                        ax = cached_add_subplot(fig, axs, r, c, k, sharex=ax0)
                    ax.set_title(q)
                    ods.plot_quantity('summary.global_quantities.%s.value' % q, label=q, ax=ax, xlabel=['', None][int(k > (n - c))])

    return {'ax': axs, 'fig': fig}


@add_to__ODS__
def quantity(
    ods, key, yname=None, xname=None, yunits=None, xunits=None, ylabel=None, xlabel=None, label=None, xnorm=1.0, ynorm=1.0, ax=None, **kw
):
    r"""
    Provides convenient way to plot 1D quantities in ODS

    For example:
        >>> ods.plot_quantity('@core.*elec.*dens', '$n_e$', lw=2)
        >>> ods.plot_quantity('@core.*ion.0.*dens.*th', '$n_D$', lw=2)
        >>> ods.plot_quantity('@core.*ion.1.*dens.*th', '$n_C$', lw=2)

    :param ods: ODS instance

    :param key: ODS location or search pattern

    :param yname: name of the y quantity

    :param xname: name of the x quantity

    :param yunits: units of the y quantity

    :param xunits: units of the x quantity

    :param ylabel: plot ylabel

    :param xlabel: plot xlabel

    :param ynorm: normalization factor for y

    :param xnorm: normalization factor for x

    :param label: label for the legend

    :param ax: axes instance into which to plot (default: gca())

    :param \**kw: extra arguments are passed to the plot function

    :return: axes instance

    """

    from matplotlib import pyplot

    # handle regular expressions
    key = ods.search_paths(key, 1, '@')[0]

    if ax is None:
        ax = pyplot.gca()

    ds = ods.xarray(key)
    x = ds[ds.attrs['x'][0]]
    y = ds[ds.attrs['y']]

    if yname is None:
        yname = latexit.get(ds.attrs['y'], ds.attrs['y'])

    if xname is None:
        xname = latexit.get(ds.attrs['x'][0], ds.attrs['x'][0])

    if yunits is None:
        yunits = y.attrs.get('units', '-')
    yunits = "[%s]" % latexit.get(yunits, yunits)
    yunits = yunits if yunits not in ['[-]', '[None]'] else ''

    if xunits is None:
        xunits = x.attrs.get('units', '-')
    xunits = "[%s]" % latexit.get(xunits, xunits)
    xunits = xunits if xunits not in ['[-]', '[None]'] else ''

    if label is None:
        label = yname
    kw['label'] = label

    if ylabel is None:
        ylabel = yunits

    if xlabel is None:
        xlabel = ' '.join(filter(None, [xname, xunits]))

    uband(x * xnorm, y * ynorm, ax=ax, **kw)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return {'ax': ax}


# this test is here to prevent importing matplotlib at the top of this file
if 'matplotlib' in locals() or 'pyplot' in locals() or 'plt' in locals():
    raise Exception('Do not import matplotlib at the top level of %s' % os.path.split(__file__)[1])
