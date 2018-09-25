from __future__ import print_function, division, unicode_literals

import matplotlib
from matplotlib import pyplot

import inspect
from .omas_utils import *
from .omas_physics import cocos_transform

__all__ = []


def add_to__ODS__(f):
    __all__.append(f.__name__)
    return f


# ================================
# plotting helper functions
# ================================
def sanitize_version_number(version):
    """Removes common non-numerical characters from version numbers obtained from git tags, such as '_rc', etc."""
    if version.startswith('.'):
        version = '-1' + version
    version = version.replace('_rc', '.')
    return version


def compare_version(version1, version2):
    """Returns 1 if version1 > version2, -1 if version1 < version2, or 0 if version1 == version2."""
    version1 = sanitize_version_number(version1)
    version2 = sanitize_version_number(version2)

    def normalize(v):
        if 'r' in v:
            v = v.split('r')[0]
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]

    return (normalize(version1) > normalize(version2)) - (normalize(version1) < normalize(version2))


def contourPaths(x, y, Z, levels, remove_boundary_points=False, smooth_factor=1):
    """
    :param x: 1D x coordinate

    :param y: 1D y coordinate

    :param Z: 2D data

    :param levels: levels to trace

    :param remove_boundary_points: remove traces at the boundary

    :param smooth_factor: smooth contours by cranking up grid resolution. Requires scipy.

    :return: list of segments
    """
    import matplotlib
    try:
        from scipy import ndimage
    except ImportError:
        printd('Warning: failed to import scipy in contourPaths. smoothing disabled.')
        ndimage = None
    if compare_version(matplotlib.__version__, '2.1') >= 0:
        import matplotlib._contour as _contour
    else:
        from matplotlib import _cntr

    sf = int(round(smooth_factor))
    if sf > 1 and ndimage is not None:
        x = ndimage.zoom(x, sf)
        y = ndimage.zoom(y, sf)
        Z = ndimage.zoom(Z, sf)

    [X, Y] = numpy.meshgrid(x, y)
    if compare_version(matplotlib.__version__, '2.1') >= 0:
        contour_generator = _contour.QuadContourGenerator(X, Y, Z, None, True, 0)
    else:
        Cntr = matplotlib._cntr.Cntr(X, Y, Z)

    allsegs = []
    for level in levels:
        if compare_version(matplotlib.__version__, '2.1') >= 0:
            segs = contour_generator.create_contour(level)
        else:
            nlist = Cntr.trace(level)
            nseg = len(nlist) // 2
            segs = nlist[:nseg]
        if not remove_boundary_points:
            segs_ = segs
        else:
            segs_ = []
            for segarray in segs:
                x_ = segarray[:, 0]
                y_ = segarray[:, 1]
                valid = []
                for i in range(len(x_) - 1):
                    if numpy.isclose(x_[i], x_[i + 1]) and (
                            numpy.isclose(x_[i], max(x)) or numpy.isclose(x_[i], min(x))):
                        continue
                    if numpy.isclose(y_[i], y_[i + 1]) and (
                            numpy.isclose(y_[i], max(y)) or numpy.isclose(y_[i], min(y))):
                        continue
                    valid.append((x_[i], y_[i]))
                    if i == len(x_):
                        valid.append(x_[i + 1], y_[i + 1])
                if len(valid):
                    segs_.append(numpy.array(valid))

        segs = map(matplotlib.path.Path, segs_)
        allsegs.append(segs)
    return allsegs


class Uband(object):
    """
    This class wraps the line and PollyCollection(s) associated with a banded
    errorbar plot for use in the uband function.

    It's methods are Line2D methods distributed to both the line and bands if
    applicable, or just to the line alone otherwise.

    """

    def __init__(self, line, bands):
        """
        :param line: Line2D
            A line of the x,y nominal values
        :param bands: list of PolyCollections
            The fill_between and/or fill_betweenx PollyCollections spanning the
            std_devs of the x,y data.

        """
        self.line = line  # matplotlib.lines.Line2D
        self.bands = list(matplotlib.cbook.flatten([bands]))  # matplotlib.collections.PolyCollection(s)


def _method_factory(self, key, bands=True):
    """Add a method that calls the same method for line and band
    or just for the line"""
    if bands:
        def method(self, *args, **kw):
            """
            Call the same method for line and band.
            Returns Line2D method call result.
            """
            for band in self.bands:
                getattr(band, key)(*args, **kw)
            return getattr(self.line, key)(*args, **kw)
    else:
        def method(self, *args, **kw):
            """Call the line method"""
            return getattr(self.line, key)(*args, **kw)
    return method


for _name, _method in inspect.getmembers(matplotlib.lines.Line2D, predicate=inspect.ismethod):
    if _name.startswith('_'):
        continue
    setattr(Uband, _name, _method_factory(Uband, _name,
                                          bands=_name in ['set_color', 'set_lw', 'set_linewidth', 'set_dashes',
                                                          'set_linestyle']))


def uband(x, y, ax=None, fill_kw={'alpha': 0.25}, **kw):
    '''
    Given arguments x,y where either or both have uncertainties, plot x,y using pyplt.plot
    of the nominal values and surround it with with a shaded error band using matplotlib's
    fill_between and/or fill_betweenx.

    If y or x is more than 1D, it is flattened along every dimension but the last.

    :param x: array of independent axis values

    :param y: array of values with uncertainties, for which shaded error band is plotted

    :param ax: The axes instance into which to plot (default: gca())

    :param fill_kw: dict. Passed to pyplot.fill_between

    :param \**kw: Passed to pyplot.plot

    :return: list. A list of Uband objects containing the line and bands of each (x,y) along the last dimension.

    '''

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
        xnom = numpy.atleast_1d(numpy.squeeze(uncertainties.unumpy.nominal_values(xi)))
        xerr = numpy.atleast_1d(numpy.squeeze(uncertainties.unumpy.std_devs(xi)))
        ynom = numpy.atleast_1d(numpy.squeeze(uncertainties.unumpy.nominal_values(yi)))
        yerr = numpy.atleast_1d(numpy.squeeze(uncertainties.unumpy.std_devs(yi)))

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


@add_to__ODS__
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


def gas_arrow(ods, r, z, direction=None, snap_to=numpy.pi/4.0, ax=None, color=None, pad=1.0, **kw):
    """
    Draws an arrow pointing in from the gas valve
    :param ods: ODS instance

    :param r: float
        R position of gas injector (m)

    :param z: float
        Z position of gas injector (m)

    :param direction: float
        Direction of injection (radians, COCOS should match ods.cocos). None = try to guess.

    :param snap_to: float
        Snap direction angle to nearest value. Set snap to pi/4 to snap to 0, pi/4, pi/2, 3pi/4, etc. No in-between.

    :param ax: Axes instance to plot on

    :param color: matplotlib color specification

    :param pad: float
        Padding between arrow tip and specified (r,z)
    """

    def pick_direction():
        """Guesses the direction for the arrow (from injector toward machine) in case you don't know"""
        dr = ods['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['r'] - r
        dz = ods['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['z'] - z
        theta = numpy.arctan2(dz, -dr)
        if snap_to > 0:
            theta = snap_to * round(theta/snap_to)
        return theta

    if direction is None:
        direction = pick_direction()
    else:
        direction = cocos_transform(ods.cocos, 11)['BP'] * direction

    if ax is None:
        ax = pyplot.gca()

    shaft_len = 3.5 * (1+pad)/2.

    da = numpy.pi/10  # Angular half width of the arrow head
    x0 = numpy.cos(-direction) * pad
    y0 = numpy.sin(-direction) * pad
    head_mark = [
        (x0, y0),
        (x0+numpy.cos(-direction+da), y0+numpy.sin(-direction+da)),
        (x0+numpy.cos(-direction), y0+numpy.sin(-direction)),
        (x0+shaft_len*numpy.cos(-direction), y0+shaft_len*numpy.sin(-direction)),
        (x0+numpy.cos(-direction), y0+numpy.sin(-direction)),
        (x0+numpy.cos(-direction-da), y0+numpy.sin(-direction-da)),
    ]

    kw.pop('marker', None)  # Ignore this
    return ax.plot(r, z, marker=head_mark, color=color, markersize=100*(pad+shaft_len)/5, **kw)


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


# ================================
# ODSs' plotting methods
# ================================
@add_to__ODS__
def equilibrium_CX(ods, time_index=0, contour_smooth=3, levels=numpy.r_[0.1:10:0.1], ax=None, **kw):
    """
    Plot equilibrium cross-section
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param contour_smooth: Provides smoother contours by up-sampling first if >= 1 after rounding to nearest int.

    :param levels: list of sorted numeric values to pass to 2D plot as contour levels

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes
    """
    if ax is None:
        ax = pyplot.gca()

    label = kw.pop('label', '')  # Withhold this from all plots except the boundary to avoid spamming legend
    kw.setdefault('linewidth', 1)

    wall = None
    eq = ods['equilibrium']['time_slice'][time_index]
    if 'wall' in ods:
        if time_index in ods['wall']['description_2d']:
            wall = ods['wall']['description_2d'][time_index]['limiter']['unit']
        elif 0 in ods['wall']['description_2d']:
            wall = ods['wall']['description_2d'][0]['limiter']['unit']

    # first try to plot as function of `rho` and fallback on `psi`
    if 'phi' in eq['profiles_2d'][0] and 'phi' in eq['profiles_1d']:
        value2D = numpy.sqrt(abs(eq['profiles_2d'][0]['phi']))
        value1D = numpy.sqrt(abs(eq['profiles_1d']['phi']))
    else:
        value2D = eq['profiles_2d'][0]['psi']
        value1D = eq['profiles_1d']['psi']
    value2D = (value2D - min(value1D)) / (max(value1D) - min(value1D))

    # contours
    line = numpy.array([numpy.nan, numpy.nan])
    for item1 in contourPaths(eq['profiles_2d'][0]['grid']['dim1'], eq['profiles_2d'][0]['grid']['dim2'], value2D,
                              levels, smooth_factor=contour_smooth):
        for item in item1:
            line = numpy.vstack((line, item.vertices, numpy.array([numpy.nan, numpy.nan])))

    # internal flux surfaces w/ or w/o masking
    if wall is not None:
        path = matplotlib.path.Path(numpy.transpose(numpy.array([wall[0]['outline']['r'], wall[0]['outline']['z']])))
        patch = matplotlib.patches.PathPatch(path, facecolor='none')
        ax.add_patch(patch)
        pyplot.plot(line[:, 0], line[:, 1], **kw)
        ax.lines[-1].set_clip_path(patch)
    else:
        pyplot.plot(line[:, 0], line[:, 1], **kw)

    # plotting style
    kw1 = copy.deepcopy(kw)
    kw1['linewidth'] = kw['linewidth'] + 1
    kw1.setdefault('color', ax.lines[-1].get_color())

    # boundary
    ax.plot(eq['boundary']['outline']['r'], eq['boundary']['outline']['z'], label=label, **kw1)

    # axis
    ax.plot(eq['global_quantities']['magnetic_axis']['r'], eq['global_quantities']['magnetic_axis']['z'], '+', **kw1)

    # wall
    if wall is not None:
        ax.plot(wall[0]['outline']['r'], wall[0]['outline']['z'], 'k', linewidth=2)

        ax.axis([min(wall[0]['outline']['r']), max(wall[0]['outline']['r']), min(wall[0]['outline']['z']),
                 max(wall[0]['outline']['z'])])

    # axes
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    return ax


@add_to__ODS__
def equilibrium_summary(ods, time_index=0, fig=None, **kw):
    """
    Plot equilibrium cross-section and P, q, P', FF' profiles
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: figure handler
    """
    if fig is None:
        fig = pyplot.figure()

    ax = pyplot.subplot(1, 3, 1)
    ax = equilibrium_CX(ods, time_index=time_index, ax=ax, **kw)
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
    ax = pyplot.subplot(2, 3, 2)
    ax.plot(x, eq['profiles_1d']['pressure'], **kw)
    kw.setdefault('color', ax.lines[-1].get_color())
    ax.set_title('$\,$ Pressure')
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    pyplot.setp(ax.get_xticklabels(), visible=False)

    # q
    ax = fig.add_subplot(2, 3, 3, sharex=ax)
    ax.plot(x, eq['profiles_1d']['q'], **kw)
    ax.set_title('$q$ Safety factor')
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    if 'label' in kw:
        ax.legend(loc=0).draggable(True)
    pyplot.setp(ax.get_xticklabels(), visible=False)

    # dP_dpsi
    ax = fig.add_subplot(2, 3, 5, sharex=ax)
    ax.plot(x, eq['profiles_1d']['dpressure_dpsi'], **kw)
    ax.set_title("$P\,^\\prime$ source function")
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    pyplot.xlabel(xName)

    # FdF_dpsi
    ax = fig.add_subplot(236, sharex=ax)
    ax.plot(x, eq['profiles_1d']['f_df_dpsi'], **kw)
    ax.set_title("$FF\,^\\prime$ source function")
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    pyplot.xlabel(xName)

    ax.set_xlim([0, 1])

    return fig


@add_to__ODS__
def core_profiles_summary(ods, time_index=0, fig=None, combine_dens_temps=True, show_thermal_fast_breakdown=True,
                          show_total_density=False, **kw):
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
    if fig is None:
        fig = pyplot.figure()

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
            if item + '.density' + therm_fast in prof1d:
                if combine_dens_temps:
                    if k == 0:
                        ax = ax0 = pyplot.subplot(1, 2, 1)
                else:
                    ax = ax0 = pyplot.subplot(r, 2, (2 * k) + 1, sharex=ax)
                uband(x, prof1d[density], label=names[k] + therm_fast_name, ax=ax0, **kw)
                if k == len(prof1d['ion']):
                    ax0.set_xlabel('$\\rho$')
                    if combine_dens_temps:
                        ax0.legend(loc=0).draggable(True)
                if k == 0:
                    ax0.set_title('Density [m$^{-3}$]')
                if not combine_dens_temps:
                    ax0.set_ylabel(names[k])

        # temperatures
        if item + '.temperature' in prof1d:
            if combine_dens_temps:
                if k == 0:
                    ax = ax1 = pyplot.subplot(1, 2, 2, sharex=ax)
            else:
                ax = ax1 = pyplot.subplot(r, 2, (2 * k) + 2, sharex=ax)
            uband(x, prof1d[item + '.temperature'], label=names[k], ax=ax1, **kw)
            if k == len(prof1d['ion']):
                ax1.set_xlabel('$\\rho$')
            if k == 0:
                ax1.set_title('Temperature [eV]')

    ax.set_xlim([0, 1])
    if ax0 is not None:
        ax0.set_ylim([0, ax0.get_ylim()[1]])
    if ax1 is not None:
        ax1.set_ylim([0, ax1.get_ylim()[1]])
    return fig


@add_to__ODS__
def core_profiles_pressures(ods, time_index=0, ax=None, **kw):
    """
    Plot pressures in `ods['core_profiles']['profiles_1d'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes handler
    """
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
    ax.legend(loc=0).draggable(True)
    return ax


# ================================
# Hardware overlays
# ================================
@add_to__ODS__
def overlay(ods, ax=None, allow_autoscale=True, debug_all_plots=False, **kw):
    """
    Plots overlays of hardware/diagnostic locations on a tokamak cross section plot

    :param ods: OMAS ODS instance

    :param ax: Axes instance or None

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

            * Additional keywords are passed to the function that does the drawing; usually matplotlib.axes.Axes.plot().
    """
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
def gas_injection_overlay(
        ods, ax=None, angle_not_in_pipe_name=False, which_gas='all', simple_labels=False, label_spacer=0, colors=None,
        draw_arrow=True, **kw):
    """
    Plots overlays of gas injectors

    :param ods: OMAS ODS instance

    :param ax: Axes instance

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

        * Accepts standard omas_plot overlay keywords: mask, labelevery, notesize

        * Remaining keywords are passed to plot call for drawing markers at the gas locations.

    """
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
                    label += ' ({:0d})'.format(int(round(pipe['exit_position']['phi']*180/numpy.pi)))
                except (TypeError, ValueError):
                    pass
    try:
        rsplit = ods['equilibrium.time_slice'][0]['global_quantities.magnetic_axis.r']
    except ValueError:
        draw_arrow = False  # This won't work without magnetic axis data, either.
        rsplit = numpy.mean([float(loc.split('_')[0]) for loc in locations])

    kw.setdefault('marker', 'd')
    kw.setdefault('linestyle', ' ')
    labelevery = kw.pop('labelevery', 1)
    notesize = kw.pop('notesize', 'xx-small')

    # For each unique poloidal location, draw a marker and write a label describing all the injectors at this location.
    default_color = kw.pop('color', None)
    colors = numpy.atleast_1d(default_color if colors is None else colors).tolist()
    colors *= int(numpy.ceil(len(locations) / float(len(colors))))  # Make sure the list is long enough.
    for i, loc in enumerate(locations):
        r, z = numpy.array(loc.split('_')).astype(float)
        label = '{spacer:}\n{spacer:}'.format(spacer=' '*label_spacer).join([''] + locations[loc] + [''])
        if draw_arrow:
            kw.update(draw_arrow if isinstance(draw_arrow, dict) else {})
            gas_mark = gas_arrow(ods, r, z, ax=ax, color=colors[i], **kw)
        else:
            gas_mark = ax.plot(r, z, color=colors[i], **kw)
        kw.pop('label', None)  # Prevent label from being applied every time through the loop to avoid spammy legend
        if (labelevery > 0) and ((i % labelevery) == 0):
            va = ['top', 'bottom'][int(z > 0)]
            label = '\n'*label_spacer + label if va == 'top' else label + '\n'*label_spacer
            ax.text(r, z, label, color=gas_mark[0].get_color(),
                    va=va, ha=['left', 'right'][int(r < rsplit)], fontsize=notesize)
    return


@add_to__ODS__
def pf_active_overlay(ods, ax=None, **kw):
    """
    Plots overlays of active PF coils.
    INCOMPLETE: only the oblique geometry definition is treated so far. More should be added later.

    :param ods: OMAS ODS instance

    :param ax: Axes instance

    :param \**kw: Additional keywords
        scalex, scaley: passed to ax.autoscale_view() call at the end

        * Accepts standard omas_plot overlay keywords: mask, labelevery, notesize

        * Remaining keywords are passed to matplotlib.patches.Polygon call
            Hint: you may want to set facecolor instead of just color
    """
    # Make sure there is something to plot or else just give up and return
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
                ax.text(numpy.mean(xarr), numpy.mean(yarr), pf_id, ha='center', va='center', fontsize=notesize)

    for p in patches:
        ax.add_patch(p)  # Using patch collection breaks auto legend labeling, so add patches individually.

    ax.autoscale_view(scalex=scalex, scaley=scaley)  # add_patch doesn't include this

    return


@add_to__ODS__
def magnetics_overlay(
        ods, ax=None, show_bpol_probe=True, show_flux_loop=True, bpol_probe_color=None, flux_loop_color=None,
        bpol_probe_marker='s', flux_loop_marker='o', **kw):
    """
    Plots overlays of magnetics: B_pol probes and flux loops

    :param ods: OMAS ODS instance

    :param ax: Axes instance

    :param show_bpol_probe: bool
        Turn display of B_pol probes on/off

    :param show_flux_loop: bool
        Turn display of flux loops on/off

    :param bpol_probe_color: matplotlib color specification for B_pol probes

    :param flux_loop_color: matplotlib color specification for flux loops

    :param bpol_probe_marker: matplotlib marker specification for B_pol probes

    :param flux_loop_marker: matplotlib marker specification for flux loops

    :param \**kw: Additional keywords

        * Accepts standard omas_plot overlay keywords: mask, labelevery, notesize

        * Remaining keywords are passed to plot call
    """
    # Make sure there is something to plot or else just give up and return
    nbp = get_channel_count(
        ods, 'magnetics', check_loc='magnetics.bpol_probe.0.position.r', channels_name='bpol_probe',
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
    mask = kw.pop('mask', numpy.ones(nbp+nfl, bool))
    notesize = kw.pop('notesize', 'xx-small')

    def show_mag(n, topname, posroot, label, color_, marker, mask_):
        r = numpy.array([ods[topname][i][posroot]['r'] for i in range(n)])
        z = numpy.array([ods[topname][i][posroot]['z'] for i in range(n)])
        mark = ax.plot(r[mask_], z[mask_], color=color_, label=label, marker=marker, **kw)
        color_ = mark[0].get_color()  # If this was None before, the cycler will have given us something. Lock it in.
        for i in range(sum(mask_)):
            if (labelevery > 0) and ((i % labelevery) == 0):
                ax.text(r[mask_][i], z[mask_][i], ods[topname][i]['identifier'], color=color_, fontsize=notesize)

    if show_bpol_probe:
        show_mag(
            nbp, 'magnetics.bpol_probe', 'position', '$B_{pol}$ probes', bpol_probe_color, bpol_probe_marker,
            mask[:nbp])
    if show_flux_loop:
        show_mag(nfl, 'magnetics.flux_loop', 'position.0', 'Flux loops', flux_loop_color, flux_loop_marker, mask[nbp:])

    return


@add_to__ODS__
def interferometer_overlay(ods, ax=None, **kw):
    """
    Plots overlays of interferometer chords.

    :param ods: OMAS ODS instance

    :param ax: Axes instance

    :param \**kw: Additional keywords

        * Accepts standard omas_plot overlay keywords: mask, labelevery, notesize

        * Remaining keywords are passed to plot call
    """
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

    for i in range(nc):
        if mask[i]:
            ch = ods['interferometer.channel'][i]
            los = ch['line_of_sight']
            r1, z1, r2, z2 = los['first_point.r'], los['first_point.z'], los['second_point.r'], los['second_point.z']
            line = ax.plot([r1, r2], [z1, z2], color=color, label='interferometer' if i == 0 else '', **kw)
            color = line[0].get_color()  # If this was None before, the cycler will have given us something. Lock it in.
            if (labelevery > 0) and ((i % labelevery) == 0):
                ax.text(
                    max([r1, r2]), min([z1, z2]), ch['identifier'], color=color, va='top', ha='left', fontsize=notesize)

    return


@add_to__ODS__
def thomson_scattering_overlay(ods, ax=None, **kw):
    """
    Overlays Thomson channel locations

    :param ods: OMAS ODS instance

    :param ax: Axes instance

    :param \**kw: Additional keywords for Thomson plot:

        * Accepts standard omas_plot overlay keywords: mask, labelevery, notesize

        * Remaining keywords are passed to plot call
    """
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

    r = numpy.array([ods['thomson_scattering']['channel'][i]['position']['r'] for i in range(nc)])[mask]
    z = numpy.array([ods['thomson_scattering']['channel'][i]['position']['z'] for i in range(nc)])[mask]
    ts_id = numpy.array([ods['thomson_scattering']['channel'][i]['identifier'] for i in range(nc)])[mask]

    ts_mark = ax.plot(r, z, **kw)
    for i in range(sum(mask)):
        if (labelevery > 0) and ((i % labelevery) == 0):
            ax.text(r[i], z[i], ts_id[i], color=ts_mark[0].get_color(), fontsize=notesize)
    return


@add_to__ODS__
def charge_exchange_overlay(ods, ax=None, which_pos='closest', **kw):
    """
    Overlays Charge Exchange Recombination (CER) spectroscopy channel locations

    :param ods: OMAS ODS instance

    :param ax: Axes instance

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

        * Accepts standard omas_plot overlay keywords: mask, labelevery, notesize

        * Remaining keywords are passed to plot call
    """
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
                ax.text(numpy.mean(r[i]), numpy.mean(z[i]), cer_id[i], color=color, fontsize=notesize)
    return


@add_to__ODS__
def bolometer_overlay(ods, ax=None, reset_fan_color=True, colors=None, **kw):
    """
    Overlays bolometer chords

    :param ods: ODS instance

    :param ax: Axes instance

    :param reset_fan_color: bool
        At the start of each bolometer fan (group of channels), set color to None to let a new one be picked by the
        cycler. This will override manually specified color.

    :param colors: list of matplotlib color specifications. Do not use a single RGBA style spec.

    :param \**kw: Additional keywords for bolometer plot

        * Accepts standard omas_plot overlay keywords: mask, labelevery, notesize

        * Remaining keywords are passed to plot call for drawing markers at the gas locations.
    """
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
        colors = [kw.pop('color', None)]*nc
    else:
        colors *= nc  # Just make sure that this will always be long enough.
    ci = 0
    color = colors[ci]
    kw.setdefault('alpha', 0.8)
    default_label = kw.pop('label', None)
    labelevery = kw.pop('labelevery', 2)
    notesize = kw.pop('notesize', 'xx-small')
    for i in range(ncm):
        if (i > 0) and (bolo_id[i][0] != bolo_id[i-1][0]) and reset_fan_color:
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
            ax.text(r2[i], z2[i], '{}{}'.format(['\n', ''][int(z1[i] > 0)], bolo_id[i]), color=color,
                    ha=['right', 'left'][int(z1[i] > 0)], va='top', fontsize=notesize)
