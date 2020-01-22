'''plotting ODS methods and utilities

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_physics import cocos_transform

__all__ = []
__ods__ = []


def add_to__ODS__(f):
    '''
    anything wrapped here will be available as a ODS method with name 'plot_'+f.__name__
    '''
    __all__.append(f.__name__)
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
        tmp = ax.errorbar(nominal_values(xi),
                          nominal_values(yi), xerr=std_devs(xi),
                          yerr=std_devs(yi), **kwargs)
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

        l, = ax.plot(xnom, ynom, **kw)

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
    except (TypeError, AssertionError, ValueError):
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
    if isinstance(which_gas, basestring):
        if which_gas == 'all':
            include = True
    elif isinstance(which_gas, list):
        include = any([wg in label for wg in which_gas])
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

    shaft_len = 3.5 * (1 + pad) / 2.

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
    label_ha = numpy.atleast_1d(kw.pop('label_ha', None)).tolist()
    label_va = numpy.atleast_1d(kw.pop('label_va', None)).tolist()
    if len(label_ha) == 1:
        label_ha *= n
    if len(label_va) == 1:
        label_va *= n

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


# hold last 100 references of matplotlib.widgets.Slider
_stimes = []


def ods_time_plot(ods_plot_function, time, ods, time_index, **kw):
    r'''
    Utility function for generating time dependent plots

    :param ods_plot_function: ods plot function to be called
    this function must accept ax (either a single or a list of axes)
    and must return the axes (or list of axes) it used

    :param time: array of times

    :param ods: ods

    :param time_index: time indexes to be scanned

    :param \**kw: extra aruments to passed to ods_plot_function

    :return: slider instance and list of axes used
    '''
    from matplotlib import pyplot
    from matplotlib.widgets import Slider

    time_index = numpy.atleast_1d(time_index)
    time = time[time_index]
    axs = {}

    def do_clean(time0):
        for ax in axs:
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
        tmp = ods_plot_function(ods, time_index0, ax=ax, **kw)
        if isinstance(tmp, dict):
            axs.update(tmp)
        else:
            axs[1, 1, 1] = tmp

    stime, axtime = kw.pop('stime', (None, None))

    update(time[0])

    if stime is None:
        axtime = pyplot.axes([0.1, 0.96, 0.75, 0.03])
        stime = Slider(axtime, 'Time[s]', min(time), max(time), valinit=min(time), valstep=min(numpy.diff(time)))
        if stime not in _stimes:
            _stimes.append(stime)
            if len(_stimes) > 100:
                _stimes.pop(0)
        stime.on_changed(do_clean)
    stime.on_changed(update)
    for time0 in time:
        axtime.axvline(time0, color=['r', 'y', 'c', 'm'][stime.cnt - 2])
    return (stime, axtime), axs


def cached_add_subplot(fig, ax_cache, *args, **kw):
    r'''
    Utility function that works like matplotlib add_subplot
    but reuses axes if these were already used before

    :param fig: matplotlib figure

    :param ax_cache: caching dictionary

    :param \*args: arguments passed to matplotlib add_subplot

    :param \**kw: keywords arguments passed to matplotlib add_subplot

    :return: matplotlib axes
    '''
    if args in ax_cache:
        return ax_cache[args]
    else:
        ax = fig.add_subplot(*args, **kw)
        ax_cache[args] = ax
        return ax


# ================================
# ODSs' plotting methods
# ================================
@add_to__ODS__
def equilibrium_CX(ods, time_index=None, levels=numpy.r_[0.1:0.9 + 0.0001:0.1], contour_quantity='rho', allow_fallback=True, ax=None, sf=3, label_contours=None, **kw):
    r"""
    Plot equilibrium cross-section
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: ODS instance
        input ods containing equilibrium data

    :param time_index: int
        time slice to plot

    :param levels: sorted numeric iterable
        values to pass to 2D plot as contour levels

    :param contour_quantity: string
        quantity to contour; options: psi (poloidal magnetic flux), rho (sqrt of toroidal flux), phi (toroidal flux)

    :param allow_fallback: bool
        If rho/phi is requested but not available, plot on psi instead if allowed. Otherwise, raise ValueError.

    :param ax: Axes instance [optional]
        axes to plot in (active axes is generated if `ax is None`)

    :param sf: int
        Resample scaling factor. For example, set to 3 to resample to 3x higher resolution. Makes contours smoother.

    :param label_contours: bool or None
        True/False: do(n't) label contours
        None: only label if contours are of q

    :param \**kw: arguments passed to matplotlib plot statements

    :return: Axes instance
    """
    if time_index is None:
        time_index = numpy.arange(len(ods['equilibrium'].time()))
    if isinstance(time_index, (list, numpy.ndarray)):
        time = ods['equilibrium'].time()
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(equilibrium_CX, time, ods, time_index, levels=levels, contour_quantity=contour_quantity, allow_fallback=allow_fallback, ax=ax, sf=sf, label_contours=label_contours, **kw)

    import matplotlib
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
    kw1['linewidth'] = kw['linewidth'] + 1

    # Boundary
    ax.plot(eq['boundary']['outline']['r'], eq['boundary']['outline']['z'], label=label, **kw1)
    kw1.setdefault('color', ax.lines[-1].get_color())

    # Magnetic axis
    if 'global_quantities.magnetic_axis.r' in eq and 'global_quantities.magnetic_axis.z':
        ax.plot(eq['global_quantities']['magnetic_axis']['r'], eq['global_quantities']['magnetic_axis']['z'], '+', **kw1)

    # Choose quantity to plot
    phi_available = 'phi' in eq['profiles_2d'][0] and 'phi' in eq['profiles_1d']
    psi_available = 'psi' in eq['profiles_2d'][0] and 'psi' in eq['profiles_1d']
    q_available = 'q' in eq['profiles_1d'] and psi_available  # Use 1d and 2d psi to interpolate to get 2d q

    if psi_available and (not q_available) and (contour_quantity == 'q'):
        if allow_fallback:
            contour_quantity = 'psi'
            printd('q was requested but not found; falling back to psi contours')
        else:
            raise ValueError('q (safety factor) is not available')
    elif phi_available and (not q_available) and (contour_quantity == 'q'):
        if allow_fallback:
            contour_quantity = 'rho'
            printd('q was requested but not found; falling back to rho contours')
        else:
            raise ValueError('q (safety factor) is not available')
    elif psi_available and (not phi_available) and (contour_quantity in ['rho', 'phi']):
        if allow_fallback:
            contour_quantity = 'psi'
            printd('phi was requested but not found; falling back to psi contours')
        else:
            raise ValueError('phi (toroidal magnetic flux) is not available')
    elif phi_available and (not psi_available) and (contour_quantity in ['psi']):
        if allow_fallback:
            contour_quantity = 'rho'
            printd('psi was requested but not found; falling back to rho contours')
        else:
            raise ValueError('psi (poloidal magnetic flux) is not available')
    elif (not phi_available) and (not psi_available):
        if allow_fallback:
            print('No equilibrium data to plot. Aborting.')
            return
        else:
            raise ValueError('No equilibrium data to plot. Need either psi, phi, or q.')

    # Pull out contour value
    if contour_quantity == 'rho':
        value_2d = numpy.sqrt(abs(eq['profiles_2d'][0]['phi']))
        value_1d = numpy.sqrt(abs(eq['profiles_1d']['phi']))
    elif contour_quantity == 'phi':
        value_2d = abs(eq['profiles_2d'][0]['phi'])
        value_1d = abs(eq['profiles_1d']['phi'])
    elif contour_quantity == 'psi':
        value_2d = eq['profiles_2d'][0]['psi']
        value_1d = eq['profiles_1d']['psi']
    elif contour_quantity == 'q':
        import scipy.interpolate
        x_value_2d = eq['profiles_2d'][0]['psi']
        x_value_1d = eq['profiles_1d']['psi']
        value_1d = eq['profiles_1d']['q']
        value_2d = scipy.interpolate.interp1d(x_value_1d, value_1d, bounds_error=False, fill_value='extrapolate')(x_value_2d)
    else:
        raise ValueError('Unrecognized contour_quantity: {}. Please choose psi, rho, phi, or q'.format(contour_quantity))
    if contour_quantity != 'q':
        value_2d = (value_2d - min(value_1d)) / (max(value_1d) - min(value_1d))

    # Wall clipping
    if wall is not None:
        path = matplotlib.path.Path(numpy.transpose(numpy.array([wall[0]['outline']['r'], wall[0]['outline']['z']])))
        wall_path = matplotlib.patches.PathPatch(path, facecolor='none')
        ax.add_patch(wall_path)

    # Contours
    if 'r' in eq['profiles_2d'][0] and 'z' in eq['profiles_2d'][0]:
        r = eq['profiles_2d'][0]['r']
        z = eq['profiles_2d'][0]['z']
    else:
        z, r = numpy.meshgrid(eq['profiles_2d'][0]['grid']['dim2'], eq['profiles_2d'][0]['grid']['dim1'])

    # Resample
    if sf > 1:
        import scipy.ndimage
        r = scipy.ndimage.zoom(r, sf)
        z = scipy.ndimage.zoom(z, sf)
        value_2d = scipy.ndimage.zoom(value_2d, sf)

    kw.setdefault('colors', kw1['color'])
    kw['linewidths'] = kw.pop('linewidth')
    value_2d = value_2d.copy()
    value_2d[:, -1] = value_2d[:, -2]
    value_2d[-1, :] = value_2d[-2, :]
    value_2d[-1, -1] = value_2d[-2, -2]
    levels = numpy.r_[0.1:0.9 + 0.0001:0.1]
    cs = ax.contour(r, z, value_2d, levels, **kw)

    if label_contours or ((label_contours is None) and (contour_quantity == 'q')):
        ax.clabel(cs)

    # Internal flux surfaces w/ or w/o masking
    if wall is not None:
        for collection in cs.collections:
            collection.set_clip_path(wall_path)

    # Wall
    if wall is not None:
        ax.plot(wall[0]['outline']['r'], wall[0]['outline']['z'], 'k', linewidth=2)

        ax.axis([min(wall[0]['outline']['r']), max(wall[0]['outline']['r']), min(wall[0]['outline']['z']),
                 max(wall[0]['outline']['z'])])

    # Axes
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    return ax


@add_to__ODS__
def equilibrium_summary(ods, time_index=None, fig=None, **kw):
    """
    Plot equilibrium cross-section and P, q, P', FF' profiles
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: figure handler
    """

    from matplotlib import pyplot

    if fig is None:
        fig = pyplot.figure()

    if time_index is None:
        time_index = numpy.arange(len(ods['equilibrium'].time()))
    if isinstance(time_index, (list, numpy.ndarray)):
        time = ods['equilibrium'].time()
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(equilibrium_summary, time, ods, time_index, fig=fig, ax={}, **kw)

    axs = kw.pop('ax', {})

    ax = cached_add_subplot(fig, axs, 1, 3, 1)
    equilibrium_CX(ods, time_index=time_index, ax=ax, **kw)
    eq = ods['equilibrium']['time_slice'][time_index]

    # x
    if 'phi' in eq['profiles_2d'][0] and 'phi' in eq['profiles_1d']:
        x = numpy.sqrt(abs(eq['profiles_1d']['phi']))
        xName = '$\\rho$'
    else:
        x = eq['profiles_1d']['psi']
        xName = '$\\psi$'
    x = (x - min(x)) / (max(x) - min(x))

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

    ax.set_xlim([0, 1])

    return axs


@add_to__ODS__
def core_profiles_summary(ods, time_index=None, fig=None, combine_dens_temps=True, show_thermal_fast_breakdown=True, show_total_density=True, **kw):
    """
    Plot densities and temperature profiles for electrons and all ion species
    as per `ods['core_profiles']['profiles_1d'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param combine_dens_temps: combine species plot of density and temperatures

    :param show_thermal_fast_breakdown: bool
        Show thermal and fast components of density in addition to total if available

    :param show_total_density: bool
        Show total thermal+fast in addition to thermal/fast breakdown if available

    :param kw: arguments passed to matplotlib plot statements

    :return: figure handler
    """

    from matplotlib import pyplot

    if fig is None:
        fig = pyplot.figure()

    if time_index is None:
        time_index = numpy.arange(len(ods['core_profiles'].time()))
    if isinstance(time_index, (list, numpy.ndarray)):
        time = ods['core_profiles'].time()
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(core_profiles_summary, time, ods, time_index, fig=fig, ax={}, combine_dens_temps=combine_dens_temps, show_thermal_fast_breakdown=show_thermal_fast_breakdown, show_total_density=show_total_density, **kw)

    axs = kw.pop('ax', {})

    prof1d = ods['core_profiles']['profiles_1d'][time_index]
    x = prof1d['grid.rho_tor_norm']

    what = ['electrons'] + ['ion[%d]' % k for k in range(len(prof1d['ion']))]
    names = ['Electrons'] + [prof1d['ion[%d].label' % k] + ' ion' for k in range(len(prof1d['ion']))]

    r = len(prof1d['ion']) + 1

    ax = ax0 = ax1 = None
    for k, item in enumerate(what):

        # densities (thermal and fast)
        for therm_fast in ['', '_thermal', '_fast']:
            if (not show_thermal_fast_breakdown) and len(therm_fast):
                continue  # Skip _thermal and _fast because the flag turned these details off
            if (not show_total_density) and (len(therm_fast) == 0):
                continue  # Skip total thermal+fast because the flag turned it off
            therm_fast_name = {
                '': ' (thermal+fast)',
                '_thermal': ' (thermal)' if show_total_density else '',
                '_fast': ' (fast)',
            }[therm_fast]
            density = item + '.density' + therm_fast
            # generate axes
            if combine_dens_temps:
                if ax0 is None:
                    ax = ax0 = cached_add_subplot(fig, axs, 1, 2, 1)
            else:
                ax = ax0 = cached_add_subplot(fig, axs, r, 2, (2 * k) + 1, sharex=ax, sharey=ax0)
            # plot if data is present
            if item + '.density' + therm_fast in prof1d:
                uband(x, prof1d[density], label=names[k] + therm_fast_name, ax=ax0, **kw)
                if k == len(prof1d['ion']):
                    ax0.set_xlabel('$\\rho$')
                if k == 0:
                    ax0.set_title('Density [m$^{-3}$]')
                if not combine_dens_temps:
                    ax0.set_ylabel(names[k])
            # add plot of measurements
            if density + '_fit.measured' in prof1d and density + '_fit.rho_tor_norm' in prof1d:
                uerrorbar(prof1d[density + '_fit.rho_tor_norm'], prof1d[density + '_fit.measured'], ax=ax)

        # temperatures
        if combine_dens_temps:
            if ax1 is None:
                ax = ax1 = cached_add_subplot(fig, axs, 1, 2, 2, sharex=ax)
        else:
            ax = ax1 = cached_add_subplot(fig, axs, r, 2, (2 * k) + 2, sharex=ax, sharey=ax1)
        # plot if data is present
        if item + '.temperature' in prof1d:
            uband(x, prof1d[item + '.temperature'], label=names[k], ax=ax1, **kw)
            if k == len(prof1d['ion']):
                ax1.set_xlabel('$\\rho$')
            if k == 0:
                ax1.set_title('Temperature [eV]')
            # add plot of measurements
            if item + '.temperature_fit.measured' in prof1d and item + '.temperature_fit.rho_tor_norm' in prof1d:
                uerrorbar(prof1d[item + '.temperature_fit.rho_tor_norm'], prof1d[item + '.temperature_fit.measured'], ax=ax)

    ax.set_xlim([0, 1])
    if ax0 is not None:
        ax0.set_ylim([0, ax0.get_ylim()[1]])
    if ax1 is not None:
        ax1.set_ylim([0, ax1.get_ylim()[1]])
    return axs


@add_to__ODS__
def core_profiles_pressures(ods, time_index=None, ax=None, **kw):
    """
    Plot pressures in `ods['core_profiles']['profiles_1d'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """

    if time_index is None:
        time_index = numpy.arange(len(ods['core_profiles'].time()))
    if isinstance(time_index, (list, numpy.ndarray)):
        time = ods['core_profiles'].time()
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(core_profiles_pressures, time, ods, time_index=time_index, ax=ax)

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
    import matplotlib
    if compare_version(matplotlib.__version__, '3.1.0') >= 0:
        leg.set_draggable(True)
    else:
        leg.draggable(True)
    return ax


# ================================
# Heating and current drive
# ================================
@add_to__ODS__
def waves_beam_CX(ods, time_index=None, ax=None, **kw):
    """
    Plot waves beams in poloidal cross-section

    :param ods: input ods

    :param time_index: time slice to plot

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """
    if time_index is None:
        time_index = numpy.arange(len(ods['waves'].time()))
    if isinstance(time_index, (list, numpy.ndarray)):
        time = ods['waves'].time()
        if len(time) == 1:
            time_index = time_index[0]
        else:
            return ods_time_plot(waves_beam_CX, time, ods, time_index, ax=ax, **kw)

    import matplotlib
    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    coherent_wave = ods['waves.coherent_wave']

    for cw in coherent_wave:
        bt = coherent_wave[cw]['beam_tracing'][time_index]
        for b in bt['beam'].values():
            ax.plot(b['position.r'], b['position.z'], **kw)

    return ax


@add_to__ODS__
def nbi_summary(ods, ax=None):
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
    return


# ================================
# Hardware overlays
# ================================
@add_to__ODS__
def overlay(ods, ax=None, allow_autoscale=True, debug_all_plots=False, **kw):
    r"""
    Plots overlays of hardware/diagnostic locations on a tokamak cross section plot

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param allow_autoscale: bool
        Certain overlays will be allowed to unlock xlim and ylim, assuming that they have been locked by equilibrium_CX.
        If this option is disabled, then hardware systems like PF-coils will be off the plot and mostly invisible.

    :param debug_all_plots: bool
        Individual hardware systems are on by default instead of off by default.

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

            * label_r_shift: numeric
                Add a constant offset to the R coordinates of all text labels for the current hardware system
                (in data units, which would normally be m)

            * label_z_shift: numeric
                Add a constant offset to the Z coordinates of all text labels for the current hardware system
                (in data units, which would normally be m)

            * Additional keywords are passed to the function that does the drawing; usually matplotlib.axes.Axes.plot().
    """

    from matplotlib import pyplot

    if ax is None:
        ax = pyplot.gca()

    overlay_on_by_default = ['thomson_scattering']  # List of strings describing default hardware to be shown
    for hw_sys in list_structures(ods.imas_version):
        if kw.get(hw_sys, ((hw_sys in overlay_on_by_default) or debug_all_plots)):
            overlay_kw = kw.get(hw_sys, {}) if isinstance(kw.get(hw_sys, {}), dict) else {}
            try:
                overlay_function = eval('{}_overlay'.format(hw_sys))
            except NameError:
                pass
            else:
                if allow_autoscale and hw_sys in ['pf_active', 'gas_injection']:  # Not all systems need expanded range to fit everything
                    ax.set_xlim(auto=True)
                    ax.set_ylim(auto=True)
                overlay_function(ods, ax, **overlay_kw)

    return


@add_to__ODS__
def gas_injection_overlay(ods, ax=None, angle_not_in_pipe_name=False, which_gas='all',
                          simple_labels=False, label_spacer=0, colors=None, draw_arrow=True, **kw):
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

    # Make sure there is something to plot or else just give up and return
    npipes = get_channel_count(
        ods, 'gas_injection', check_loc='gas_injection.pipe.0.exit_position.r', test_checker='checker > 0',
        channels_name='pipe')
    if npipes == 0:
        return

    mask = kw.pop('mask', numpy.ones(npipes, bool))

    if ax is None:
        ax = pyplot.gca()

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
            location_name = '{:0.3f}_{:0.3f}'.format(r, z)

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
            except ValueError:
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
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

    # For each unique poloidal location, draw a marker and write a label describing all the injectors at this location.
    default_color = kw.pop('color', None)
    colors = numpy.atleast_1d(default_color if colors is None else colors).tolist()
    colors *= int(numpy.ceil(len(locations) / float(len(colors))))  # Make sure the list is long enough.
    for i, loc in enumerate(locations):
        r, z = numpy.array(loc.split('_')).astype(float)
        label = '{spacer:}\n{spacer:}'.format(spacer=' ' * label_spacer).join([''] + [locations[loc][0]] + [''])
        if draw_arrow:
            kw.update(draw_arrow if isinstance(draw_arrow, dict) else {})
            gas_mark = gas_arrow(ods, r, z, r2=locations[loc][-2], z2=locations[loc][-1], ax=ax, color=colors[i], **kw)
        else:
            gas_mark = ax.plot(r, z, color=colors[i], **kw)
        kw.pop('label', None)  # Prevent label from being applied every time through the loop to avoid spammy legend
        if (labelevery > 0) and ((i % labelevery) == 0):
            label = '\n' * label_spacer + label if label_va[i] == 'top' else label + '\n' * label_spacer
            ax.text(
                r + label_dr, z + label_dz, label,
                color=gas_mark[0].get_color(), va=label_va[i], ha=label_ha[i], fontsize=notesize,
            )
    return


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
    # Make sure there is something to plot or else just give up and return

    import matplotlib
    from matplotlib import pyplot

    nc = get_channel_count(
        ods, 'pf_active', check_loc='pf_active.coil.0.element.0.geometry.geometry_type', channels_name='coil',
        test_checker='checker > -1')
    if nc == 0:
        return

    if ax is None:
        ax = pyplot.gca()

    kw.setdefault('label', 'Active PF coils')
    kw.setdefault('facecolor', 'gray')
    kw.setdefault('edgecolor', 'k')
    kw.setdefault('alpha', 0.7)
    labelevery = kw.pop('labelevery', 0)
    notesize = kw.pop('notesize', 'xx-small')
    mask = kw.pop('mask', numpy.ones(nc, bool))
    scalex, scaley = kw.pop('scalex', True), kw.pop('scaley', True)
    label_ha, label_va, kw = text_alignment_setup(nc, default_ha='center', default_va='center', **kw)
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

    def path_rectangle(rectangle):
        """
        :param rectangle: ODS sub-folder: element.*.geometry.rectangle
        :return: n x 2 array giving the path around the outline of the coil element, suitable for input to Polygon()
        """
        x = rectangle['r']
        y = rectangle['z']
        dx = rectangle['width']
        dy = rectangle['height']
        return numpy.array([
            [x - dx / 2., x - dx / 2., x + dx / 2., x + dx / 2.],
            [y - dy / 2., y + dy / 2., y + dy / 2., y - dy / 2.]]).T

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
                geometry_type = geo_type_lookup(ods['pf_active.coil'][i]['element.0.geometry.geometry_type'],
                                                'pf_active', ods.imas_version)
            except (IndexError, ValueError):
                geometry_type = 'unrecognized'
            try:
                path = eval('path_{}'.format(geometry_type))(
                    ods['pf_active.coil'][i]['element.0.geometry'][geometry_type])
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
                    numpy.mean(path[:, 0]) + label_dr,
                    numpy.mean(path[:, 1]) + label_dz,
                    pf_id,
                    ha=label_ha[i], va=label_va[i], fontsize=notesize,
                )

    for p in patches:
        ax.add_patch(p)  # Using patch collection breaks auto legend labeling, so add patches individually.

    ax.autoscale_view(scalex=scalex, scaley=scaley)  # add_patch doesn't include this

    return


@add_to__ODS__
def magnetics_overlay(
        ods, ax=None, show_bpol_probe=True, show_flux_loop=True, bpol_probe_color=None, flux_loop_color=None,
        bpol_probe_marker='s', flux_loop_marker='o', **kw):
    r"""
    Plots overlays of magnetics: B_pol probes and flux loops

    :param ods: OMAS ODS instance

    :param ax: axes instance into which to plot (default: gca())

    :param show_bpol_probe: bool
        Turn display of B_pol probes on/off

    :param show_flux_loop: bool
        Turn display of flux loops on/off

    :param bpol_probe_color: matplotlib color specification for B_pol probes

    :param flux_loop_color: matplotlib color specification for flux loops

    :param bpol_probe_marker: matplotlib marker specification for B_pol probes

    :param flux_loop_marker: matplotlib marker specification for flux loops

    :param \**kw: Additional keywords

        * Accepts standard omas_plot overlay keywords listed in overlay() documentation: mask, labelevery, ...

        * Remaining keywords are passed to plot call
    """

    from matplotlib import pyplot

    # Make sure there is something to plot or else just give up and return
    nbp = get_channel_count(
        ods, 'magnetics', check_loc='magnetics.b_field_pol_probe.0.position.r', channels_name='b_field_pol_probe',
        test_checker='checker > 0')
    nfl = get_channel_count(
        ods, 'magnetics', check_loc='magnetics.flux_loop.0.position.0.r', channels_name='flux_loop',
        test_checker='checker > 0')
    if max([nbp, nfl]) == 0:
        return

    if ax is None:
        ax = pyplot.gca()

    color = kw.pop('color', None)
    bpol_probe_color = color if bpol_probe_color is None else bpol_probe_color
    flux_loop_color = color if flux_loop_color is None else flux_loop_color
    kw.pop('marker', None)
    kw.setdefault('linestyle', ' ')
    labelevery = kw.pop('labelevery', 0)
    mask = kw.pop('mask', numpy.ones(nbp + nfl, bool))
    notesize = kw.pop('notesize', 'xx-small')
    label_ha, label_va, kw = text_alignment_setup(nbp + nfl, **kw)
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

    def show_mag(n, topname, posroot, label, color_, marker, mask_):
        r = numpy.array([ods[topname][i][posroot]['r'] for i in range(n)])
        z = numpy.array([ods[topname][i][posroot]['z'] for i in range(n)])
        mark = ax.plot(r[mask_], z[mask_], color=color_, label=label, marker=marker, **kw)
        color_ = mark[0].get_color()  # If this was None before, the cycler will have given us something. Lock it in.
        for i in range(sum(mask_)):
            if (labelevery > 0) and ((i % labelevery) == 0):
                ax.text(
                    r[mask_][i] + label_dr, z[mask_][i] + label_dz, ods[topname][i]['identifier'],
                    color=color_, fontsize=notesize, ha=label_ha[i], va=label_va[i],
                )

    if show_bpol_probe:
        show_mag(
            nbp, 'magnetics.b_field_pol_probe', 'position', '$B_{pol}$ probes', bpol_probe_color, bpol_probe_marker,
            mask[:nbp])
    if show_flux_loop:
        show_mag(nfl, 'magnetics.flux_loop', 'position.0', 'Flux loops', flux_loop_color, flux_loop_marker, mask[nbp:])

    return


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

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'interferometer', check_loc='interferometer.channel.0.line_of_sight.first_point.r',
        test_checker='checker > 0')
    if nc == 0:
        return

    if ax is None:
        ax = pyplot.gca()

    color = kw.pop('color', None)
    labelevery = kw.pop('labelevery', 1)
    mask = kw.pop('mask', numpy.ones(nc, bool))
    notesize = kw.pop('notesize', 'medium')
    label_ha, label_va, kw = text_alignment_setup(nc, default_ha='left', default_va='top', **kw)
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

    for i in range(nc):
        if mask[i]:
            ch = ods['interferometer.channel'][i]
            los = ch['line_of_sight']
            r1, z1, r2, z2 = los['first_point.r'], los['first_point.z'], los['second_point.r'], los['second_point.z']
            line = ax.plot([r1, r2], [z1, z2], color=color, label='interferometer' if i == 0 else '', **kw)
            color = line[0].get_color()  # If this was None before, the cycler will have given us something. Lock it in.
            if (labelevery > 0) and ((i % labelevery) == 0):
                ax.text(
                    max([r1, r2]) + label_dr,
                    min([z1, z2]) + label_dz,
                    ch['identifier'],
                    color=color, va=label_va[i], ha=label_ha[i], fontsize=notesize,
                )
    return


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

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'thomson_scattering', check_loc='thomson_scattering.channel.0.position.r', test_checker='checker > 0')
    if nc == 0:
        return

    if ax is None:
        ax = pyplot.gca()

    labelevery = kw.pop('labelevery', 5)
    notesize = kw.pop('notesize', 'xx-small')
    mask = kw.pop('mask', numpy.ones(nc, bool))
    kw.setdefault('marker', '+')
    kw.setdefault('label', 'Thomson scattering')
    kw.setdefault('linestyle', ' ')
    label_ha, label_va, kw = text_alignment_setup(nc, **kw)
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

    r = numpy.array([ods['thomson_scattering']['channel'][i]['position']['r'] for i in range(nc)])[mask]
    z = numpy.array([ods['thomson_scattering']['channel'][i]['position']['z'] for i in range(nc)])[mask]
    ts_id = numpy.array([ods['thomson_scattering']['channel'][i]['identifier'] for i in range(nc)])[mask]

    ts_mark = ax.plot(r, z, **kw)
    for i in range(sum(mask)):
        if (labelevery > 0) and ((i % labelevery) == 0):
            ax.text(
                r[i] + label_dr,
                z[i] + label_dz,
                ts_id[i],
                color=ts_mark[0].get_color(), fontsize=notesize, ha=label_ha[i], va=label_va[i]
            )
    return


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

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'charge_exchange', check_loc='charge_exchange.channel.0.position.r.data', test_checker='any(checker > 0)')
    if nc == 0:
        return

    if ax is None:
        ax = pyplot.gca()

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
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

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
    for i in range(nc):
        if mask[i]:
            ch_type = cer_id[i][0].upper()
            color = colors.get(ch_type, None)  # See if a color has been specified for this view direction
            cer_mark = ax.plot(r[i], z[i], marker=markers.get(ch_type, 'x'), linestyle=' ', color=color,
                               label=label_bank.pop(ch_type, ''), **kw)
            colors[ch_type] = color = cer_mark[0].get_color()  # Save color for this view dir in case it was None
            if (labelevery > 0) and ((i % labelevery) == 0):
                ax.text(
                    numpy.mean(r[i]) + label_dr,
                    numpy.mean(z[i]) + label_dz,
                    cer_id[i],
                    color=color, fontsize=notesize, ha=ha[i], va=va[i]
                )
    return


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

    # Make sure there is something to plot or else just give up and return
    nc = get_channel_count(
        ods, 'bolometer', check_loc='bolometer.channel.0.line_of_sight.first_point.r', test_checker='checker > 0')
    if nc == 0:
        return

    if ax is None:
        ax = pyplot.gca()
    mask = kw.pop('mask', numpy.ones(nc, bool))

    r1 = numpy.array([ods['bolometer']['channel'][i]['line_of_sight.first_point.r'] for i in range(nc)])[mask]
    z1 = numpy.array([ods['bolometer']['channel'][i]['line_of_sight.first_point.z'] for i in range(nc)])[mask]
    r2 = numpy.array([ods['bolometer']['channel'][i]['line_of_sight.second_point.r'] for i in range(nc)])[mask]
    z2 = numpy.array([ods['bolometer']['channel'][i]['line_of_sight.second_point.z'] for i in range(nc)])[mask]
    bolo_id = numpy.array([ods['bolometer']['channel'][i]['identifier'] for i in range(nc)])[mask]

    ncm = len(r1)

    if colors is None:
        colors = [kw.pop('color', None)] * nc
    else:
        colors *= nc  # Just make sure that this will always be long enough.
    ci = 0
    color = colors[ci]
    kw.setdefault('alpha', 0.8)
    default_label = kw.pop('label', None)
    labelevery = kw.pop('labelevery', 2)
    notesize = kw.pop('notesize', 'xx-small')
    default_ha = [['right', 'left'][int(z1[i] > 0)] for i in range(ncm)]
    label_ha, label_va, kw = text_alignment_setup(ncm, default_ha=default_ha, default_va='top', **kw)
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

    for i in range(ncm):
        if (i > 0) and (bolo_id[i][0] != bolo_id[i - 1][0]) and reset_fan_color:
            ci += 1
            color = colors[ci]  # Allow color to reset when changing fans
            new_label = True
        else:
            new_label = False

        label = 'Bolometers {}'.format(bolo_id[i][0]) if default_label is None else default_label
        bolo_line = ax.plot([r1[i], r2[i]], [z1[i], z2[i]], color=color,
                            label=label if new_label or (i == 0) else '', **kw)
        if color is None:
            color = bolo_line[0].get_color()  # Make subsequent lines the same color
        if (labelevery > 0) and ((i % labelevery) == 0):
            ax.text(
                r2[i] + label_dr,
                z2[i] + label_dz,
                '{}{}'.format(['\n', ''][int(z1[i] > 0)], bolo_id[i]),
                color=color,
                ha=label_ha[i],
                va=label_va[i],
                fontsize=notesize,
            )
    return


@add_to__ODS__
def langmuir_probes_overlay(
        ods, ax=None, embedded_probes=None, colors=None, show_embedded=True, show_reciprocating=False, **kw
):
    r"""
    Overlays Langmuir probes
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
                test_checker='checker > 0',
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
            test_checker='checker > 0',
            channels_name='reciprocating',
        )
    else:
        ncr = 0
    if (nce == 0) and (ncr == 0):
        return

    # Get a handle on the axes
    if ax is None:
        ax = pyplot.gca()
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
        colors = [kw.pop('color', None)] * nc
    else:
        colors *= nc  # Just make sure that this will always be long enough.
    ci = 0
    color = colors[ci]
    kw.setdefault('alpha', 0.8)
    kw.setdefault('marker', '*')
    kw.setdefault('linestyle', ' ')
    default_label = kw.pop('label', None)
    labelevery = kw.pop('labelevery', 2)
    notesize = kw.pop('notesize', 'xx-small')
    label_dr = kw.pop('label_r_shift', 0)
    label_dz = kw.pop('label_z_shift', 0)

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
                r_e[i] + label_dr,
                z_e[i] + label_dz,
                '\n {} \n'.format(lp_id_e[i]),
                color=color, ha=ha[i], va=va[i], fontsize=notesize,
            )
    return


@add_to__ODS__
def summary(ods, fig=None, quantity=None):
    '''
    Plot summary time traces. Internally makes use of plot_quantity method.

    :param ods: input ods

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param quantity: if None plot all time-dependent global_quantities. Else a list of strings with global quantities to plot

    :return: figure handler
    '''

    from matplotlib import pyplot

    if fig is None:
        fig = pyplot.figure()
    if quantity is None:
        quantity = ods['summary.global_quantities']

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
                        ax = ax0 = fig.add_subplot(r, c, k)
                    else:
                        ax = fig.add_subplot(r, c, k, sharex=ax0)
                    ax.set_title(q)
                    ods.plot_quantity('summary.global_quantities.%s.value' % q, label=q, ax=ax, xlabel=['', None][int(k > (n - c))])
    return fig


latexit = {}
latexit['rho_tor_norm'] = '$\\rho$'
latexit['zeff'] = '$Z_{\\rm eff}$'
latexit['m^-3'] = '$m^{-3}$'
latexit['psi'] = '$\\psi$'


@add_to__ODS__
def quantity(ods, key,
             yname=None, xname=None,
             yunits=None, xunits=None,
             ylabel=None, xlabel=None, label=None,
             xnorm=1.0, ynorm=1.0,
             ax=None, **kw):
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
    return ax


# this test is here to prevent
if 'matplotlib' in locals() or 'pyplot' in locals() or 'plt' in locals():
    raise Exception('Do not import matplotlib at the top level of %s' % os.path.split(__file__)[1])
