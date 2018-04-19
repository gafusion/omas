import matplotlib
from matplotlib import pyplot
import numpy as np
import copy

__all__=[]

def add_to__all__(f):
    __all__.append(f.__name__)
    return f

def contourPaths(x, y, Z, levels, remove_boundary_points=False, smooth_factor=1):
    '''
    returns contour paths

    :param x: x grid

    :param y: y grid

    :param Z: z values

    :param levels: levels of the contours

    :param remove_boundary_points: how to treat the last point of contour surfaces that are close

    :param smooth_factor: smoothing before contouring

    :return: list of matplotlib contour paths objects
    '''
    import matplotlib
    from matplotlib import _cntr

    sf = int(round(smooth_factor))
    if sf > 1:
        x = scipy.ndimage.zoom(x, sf)
        y = scipy.ndimage.zoom(y, sf)
        Z = scipy.ndimage.zoom(Z, sf)

    [X,Y]=np.meshgrid(x,y)
    Cntr = matplotlib._cntr.Cntr(X,Y,Z)

    allsegs = []
    for level in levels:
        nlist = Cntr.trace(level)
        nseg = len(nlist)//2
        segs = nlist[:nseg]
        if not remove_boundary_points:
            segs_ = segs
        else:
            segs_ = []
            for segarray in segs:
                x_ = segarray[:,0]
                y_ = segarray[:,1]
                valid = []
                for i in range(len(x_)-1):
                    if np.isclose(x_[i],x_[i+1]) and (np.isclose(x_[i],max(x)) or np.isclose(x_[i],min(x))):
                        continue
                    if np.isclose(y_[i],y_[i+1]) and (np.isclose(y_[i],max(y)) or np.isclose(y_[i],min(y))):
                        continue
                    valid.append((x_[i],y_[i]))
                    if i==len(x_):
                        valid.append(x_[i+1],y_[i+1])
                if len(valid):
                    segs_.append(np.array(valid))

        segs=map(matplotlib.path.Path,segs_)
        allsegs.append(segs)
    return allsegs

@add_to__all__
def equilibrium_CX(ods, time_index=0, ax=None, **kw):
    '''
    Plot equilibrium cross-section
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param ax: axes to plot in (active axes is generated if `ax is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: axes
    '''
    if ax is None:
        ax=pyplot.gca()

    wall=None
    eq=ods['equilibrium']['time_slice'][time_index]
    if 'wall' in ods:
        wall=ods['wall']['description_2d'][time_index]['limiter']['unit']

    # first try to plot as function of `rho` and fallback on `psi`
    if 'phi' in eq['profiles_2d'][0] and 'phi' in eq['profiles_1d']:
        value2D=np.sqrt(abs(eq['profiles_2d'][0]['phi']))
        value1D=np.sqrt(abs(eq['profiles_1d']['phi']))
    else:
        value2D=eq['profiles_2d'][0]['psi']
        value1D=eq['profiles_1d']['psi']
    value2D=(value2D-min(value1D))/(max(value1D)-min(value1D))
    levels=np.r_[0.1:10:0.1]

    # contours
    line=np.array([np.nan,np.nan])
    for item1 in contourPaths(eq['profiles_2d'][0]['grid']['dim1'],
                              eq['profiles_2d'][0]['grid']['dim2'],
                              value2D, levels, smooth_factor=1):
        for item in item1:
            line=np.vstack(( line, item.vertices,np.array([np.nan,np.nan]) ))

    # internal flux surfaces w/ or w/o masking
    if wall is not None:
        path  = matplotlib.path.Path(np.transpose(np.array([wall[0]['outline']['r'],wall[0]['outline']['z']])))
        patch = matplotlib.patches.PathPatch(path, facecolor='none')
        ax.add_patch(patch)
        pyplot.plot(line[:,0],line[:,1],**kw)
        ax.lines[-1].set_clip_path(patch)
    else:
        pyplot.plot(line[:,0],line[:,1],**kw)

    # plotting style
    kw1=copy.deepcopy(kw)
    kw1['linewidth']=kw.setdefault('linewidth',1)+1
    kw1.setdefault('color',ax.lines[-1].get_color())

    # boundary
    ax.plot(eq['boundary']['outline']['r'],eq['boundary']['outline']['z'],**kw1)

    # axis
    ax.plot(eq['global_quantities']['magnetic_axis']['r'],eq['global_quantities']['magnetic_axis']['z'],'+',**kw1)

    # wall
    if wall is not None:
        ax.plot(wall[0]['outline']['r'],wall[0]['outline']['z'],'k',linewidth=2)

        ax.axis([min(wall[0]['outline']['r']), max(wall[0]['outline']['r']),
                        min(wall[0]['outline']['z']), max(wall[0]['outline']['z'])])

    # axes
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    return ax

@add_to__all__
def equilibrium_summary(ods, time_index=0, fig=None, **kw):
    '''
    Plot equilibrium cross-section and P, q, P', FF' profiles
    as per `ods['equilibrium']['time_slice'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param kw: arguments passed to matplotlib plot statements

    :return: list of axes
    '''
    if fig is None:
        fig=pyplot.figure()

    axs=[]

    ax=pyplot.subplot(1,3,1)
    ax=equilibrium_CX(ods, time_index=time_index, ax=ax, **kw)
    axs.append(ax)

    eq=ods['equilibrium']['time_slice'][time_index]

    # x
    if 'phi' in eq['profiles_2d'][0] and 'phi' in eq['profiles_1d']:
        x=np.sqrt(abs(eq['profiles_1d']['phi']))
        xName='$\\rho$'
    else:
        x=eq['profiles_1d']['psi']
        xName='$\\psi$'
    x=(x-min(x))/(max(x)-min(x))

    # pressure
    ax=pyplot.subplot(2,3,2)
    axs.append(ax)
    ax.plot(x,eq['profiles_1d']['pressure'], **kw)
    kw.setdefault('color',ax.lines[-1].get_color())
    ax.set_title('$\,$ Pressure')
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    pyplot.setp(ax.get_xticklabels(), visible=False)

    # q
    ax=fig.add_subplot(2,3,3,sharex=ax)
    axs.append(ax)
    ax.plot(x,eq['profiles_1d']['q'], **kw )
    ax.set_title('$q$ Safety factor')
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    if 'label' in kw:
        ax.legend(loc=0).draggable(True)
    pyplot.setp(ax.get_xticklabels(), visible=False)

    # dP_dpsi
    ax=fig.add_subplot(2,3,5,sharex=ax)
    axs.append(ax)
    ax.plot(x,eq['profiles_1d']['dpressure_dpsi'], **kw )
    ax.set_title("$P\,^\\prime$ source function")
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    pyplot.xlabel(xName)

    # FdF_dpsi
    ax=fig.add_subplot(236,sharex=ax)
    axs.append(ax)
    ax.plot(x,eq['profiles_1d']['f_df_dpsi'], **kw)
    ax.set_title("$FF\,^\\prime$ source function")
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    pyplot.xlabel(xName)

    ax.set_xlim([0,1])

@add_to__all__
def core_profiles_summary(ods, time_index=0, fig=None, combine_dens_temps=True, **kw):
    '''
    Plot densities and temperature profiles for electrons and all ion species
    as per `ods['core_profiles']['profiles_1d'][time_index]`

    :param ods: input ods

    :param time_index: time slice to plot

    :param fig: figure to plot in (a new figure is generated if `fig is None`)

    :param combine_dens_temps: combine species plot of density and temperatures

    :param kw: arguments passed to matplotlib plot statements

    :return: list of axes
    '''
    prof1d=ods['core_profiles']['profiles_1d'][time_index]
    x=prof1d['grid.rho_tor_norm']

    what=['electrons']+['ion[%d]'%k for k in range(len(prof1d['ion']))]
    names=['Electrons']+[prof1d['ion[%d].label'%k]+' ion' for k in range(len(prof1d['ion']))]

    r=len(prof1d['ion'])+1

    axs=[]
    ax=None
    for k,item in enumerate(what):

        #densities
        if combine_dens_temps:
            if k==0:
                ax=ax0=pyplot.subplot(1,2,1)
                axs.append(ax0)
        else:
            ax=ax0=pyplot.subplot(r,2,(2*k)+1,sharex=ax)
            axs.append(ax)
        ax0.plot(x,prof1d[item+'.density'],label=names[k],**kw)
        if k==len(prof1d['ion']):
            ax0.set_xlabel('$\\rho$')
            if combine_dens_temps:
                ax0.legend(loc=0).draggable(True)
        if k==0:
            ax0.set_title('Density [m$^{-3}$]')
        if not combine_dens_temps:
            ax0.set_ylabel(names[k])

        #temperatures
        if combine_dens_temps:
            if k==0:
                ax=ax1=pyplot.subplot(1,2,2,sharex=ax0)
                axs.append(ax1)
        else:
            ax=ax1=pyplot.subplot(r,2,(2*k)+2,sharex=ax)
            axs.append(ax)
        ax1.plot(x,prof1d[item+'.temperature'],label=names[k],**kw)
        if k==len(prof1d['ion']):
            ax1.set_xlabel('$\\rho$')
        if k==0:
            ax1.set_title('Temperature [eV]')

    ax.set_xlim([0,1])
    return axs

