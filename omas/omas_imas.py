from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS


# --------------------------------------------
# IMAS convenience functions
# --------------------------------------------
def imas_open(user, machine, shot, run, new=False, imas_version=omas_rcparams['default_imas_version']):
    """
    function to open an IMAS

    :param user: IMAS username

    :param machine: IMAS machine

    :param shot: IMAS shot

    :param run: IMAS run id

    :param new: whether the open should create a new IMAS tree

    :param imas_version: IMAS version

    :return: IMAS ids
    """
    import imas
    printd("ids = imas.ids(%d,%d)"%(shot,run),topic='imas_code')
    ids = imas.ids(shot,run)

    if user is None and machine is None:
        pass
    elif user is None or machine is None:
        raise (Exception('user={user}, machine={machine}, imas_version={imas_version}\n'
                         'Either specify all or none of `user`, `machine`, `imas_version`\n'
                         'If none of them are specified then use `imasdb` command to set '
                         'MDSPLUS_TREE_BASE_? environmental variables'.format(user=user, machine=machine, shot=shot,
                                                                              run=run, imas_version=imas_version)))

    if user is None and machine is None:
        if new:
            printd("ids.create()",topic='imas_code')
            ids.create()
        else:
            printd("ids.open()",topic='imas_code')
            try:
                ids.open()
            except Exception as _excp:
                if 'Error opening imas shot' in str(_excp):
                    raise(IOError('Error opening imas shot %d run %d'%(shot,run)))
        if not ids.isConnected():
            raise (Exception('Failed to establish connection to IMAS database '
                             '(shot:{shot} run:{run}, DB:{db})'.format(
                             shot=shot, run=run, db=os.environ.get('MDSPLUS_TREE_BASE_0', '???')[:-2])))

    else:
        if new:
            printd("ids.create_env(%s, %s, %s)"%(repr(user),repr(machine),repr(imas_version)),topic='imas_code')
            ids.create_env(user, machine, imas_version)
        else:
            printd("ids.open_env(%s, %s, %s)"%(repr(user),repr(machine),repr(imas_version)),topic='imas_code')
            try:
                ids.open_env(user, machine, imas_version)
            except Exception as _excp:
                if 'Error opening imas shot' in str(_excp):
                    raise(IOError('Error opening imas shot %d run %d'%(shot,run)))
        if not ids.isConnected():
            raise (Exception('Failed to establish connection to IMAS database '
                             '(user:%s machine:%s shot:%s run:%s, imas_version:%s)' %
                             (user, machine, shot, run, imas_version)))
    return ids


def imas_set(ids, path, value, skip_missing_nodes=False, allocate=False):
    """
    assign a value to a path of an open IMAS ids

    :param ids: open IMAS ids to write to

    :param path: ODS path

    :param value: value to assign

    :param skip_missing_nodes:  if the IMAS path does not exists:
                             `False` raise an error
                             `True` does not raise error
                             `None` prints a warning message

    :param allocate: whether to perform only IMAS memory allocation (ids.resize)

    :return: path if set was done, otherwise None
    """
    if numpy.atleast_1d(is_uncertain(value)).any():
        path=copy.deepcopy(path)
        tmp=imas_set(ids, path, nominal_values(value), skip_missing_nodes=skip_missing_nodes, allocate=allocate)
        path[-1]=path[-1]+'_error_upper'
        imas_set(ids, path, std_devs(value), skip_missing_nodes=skip_missing_nodes, allocate=allocate)
        return tmp

    ds = path[0]
    path = path[1:]

    # `info` IDS is used by OMAS to hold user, machine, shot, run, imas_version
    # for saving methods that do not carry that information. IMAS does not store
    # this information as part of the data dictionary.
    if ds == 'info':
        return

    # for ITM we have to append Array to the name of the data structure
    DS=ds
    if 'imas'=='itm':
        ds=ds+'Array'

    # identify data dictionary to use, from this point on `m` points to the IDS
    if hasattr(ids, ds):
        printd("",topic='imas_code')
        printd("m = getattr(ids, %r)"%ds,topic='imas_code')
        m = getattr(ids, ds)
        if hasattr(m,'time') and not isinstance(m.time,float) and not m.time.size:
            m.time.resize(1)
            m.time[0]=-1.0
    elif skip_missing_nodes is not False:
        if skip_missing_nodes is None:
            printe('WARNING: %s is not part of IMAS structure' % l2i([ds] + path))
        return
    else:
        raise (AttributeError('%s is not part of IMAS structure' % l2i([ds] + path)))

    # traverse IMAS structure until reaching the leaf
    printd("out = m",topic='imas_code')
    out = m
    for kp, p in enumerate(path):
        location=l2i([ds] + path[:kp+1])
        if isinstance(p, basestring):
            if hasattr(out, p):
                if kp < (len(path) - 1):
                    printd("out = getattr(out, %r)"%p,topic='imas_code')
                    out = getattr(out, p)
            elif skip_missing_nodes is not False:
                if skip_missing_nodes is None:
                    printe('WARNING: %s is not part of IMAS structure' % location)
                return
            else:
                raise (AttributeError('%s is not part of IMAS structure' % location))
        else:
            try:
                out = out[p]
                printd("out = out[%s]"%p,topic='imas_code')
            except (AttributeError,IndexError): # AttributeError is for ITM
                if not allocate:
                    raise (IndexError('%s structure array exceed allocation' % location))
                printd('resizing  : %s'%location, topic='imas')
                printd("out.resize(%d)"%(p+1),topic='imas_code')
                out.resize(p + 1)
                printd("out = out[%s]"%p,topic='imas_code')
                out = out[p]

    # if we are allocating data, simply stop here
    if allocate:
        return [DS] + path

    # assign data to leaf node
    printd('setting  : %s'%location, topic='imas')
    if not isinstance(value, (basestring, numpy.ndarray)):
        value=numpy.array(value)
    setattr(out, path[-1], value)
    printd("setattr(out, %r, %s)"%(path[-1],repr(value).replace('\\n','\n')),topic='imas_code')

    # return path
    return [DS] + path


def imas_get(ids, path, skip_missing_nodes=False):
    """
    read the value of a path in an open IMAS ids

    :param ids: open IMAS ids to read from

    :param path: ODS path

    :param skip_missing_nodes:  if the IMAS path does not exists:
                             `False` raise an error
                             `True` does not raise error
                             `None` prints a warning message

    :return: the value that was read if successful or None otherwise
    """
    printd('fetching: %s' % l2i(path), topic='imas')
    ds = path[0]
    path = path[1:]

    # for ITM we have to append Array to the name of the data structure
    if 'imas'=='itm':
        ds=ds+'Array'

    if hasattr(ids, ds):
        printd("m = getattr(ids, %s)"%repr(ds),topic='imas_code')
        m = getattr(ids, ds)
    elif skip_missing_nodes is not False:
        if skip_missing_nodes is None:
            printe('WARNING: %s is not part of IMAS structure' % l2i([ds] + path))
        return None
    else:
        raise (AttributeError('%s is not part of IMAS structure' % l2i([ds] + path)))

    # traverse the IDS to get the data
    out = m
    for kp, p in enumerate(path):
        if isinstance(p, basestring):
            if hasattr(out, p):
                printd("out = getattr(out, %s)"%repr(p),topic='imas_code')
                out = getattr(out, p)
            elif skip_missing_nodes is not False:
                if skip_missing_nodes is None:
                    printe('WARNING: %s is not part of IMAS structure' % l2i([ds] + path[:kp + 1]))
                    printe(out.__dict__.keys())
                return None
            else:
                raise (AttributeError('%s is not part of IMAS structure' % l2i([ds] + path[:kp + 1])))
        else:
            printd("out = out[%s]"%p,topic='imas_code')
            out = out[p]

    return out


# --------------------------------------------
# save and load OMAS to IMAS
# --------------------------------------------
def save_omas_imas(ods, user=None, machine=None, shot=None, run=None, new=False, imas_version=omas_rcparams['default_imas_version']):
    """
    save OMAS data set to IMAS

    :param ods: OMAS data set

    :param user: IMAS username (reads ods['info.user'] if user is None and finally fallsback on os.environ['USER'])

    :param machine: IMAS machine (reads ods['info.machine'] if machine is None)

    :param shot: IMAS shot (reads ods['info.shot'] if shot is None)

    :param run: IMAS run (reads ods['info.run'] if run is None and finally fallsback on 0)

    :param new: whether the open should create a new IMAS tree

    :param imas_version: IMAS version
        (reads ods['info.imas_version'] if imas_version is None and finally fallsback on imas version of current system)

    :return: paths that have been written to IMAS
    """

    # handle default values for user, machine, shot, run, imas_version
    # it tries to re-use existing information
    if user is None:
        user = ods.get('info.user', os.environ['USER'])
    if machine is None:
        machine = ods.get('info.machine', None)
    if shot is None:
        shot = ods.get('info.shot', None)
    if run is None:
        run = ods.get('info.run', 0)

    if user is not None and machine is not None:
        printd('Saving to IMAS (user:%s machine:%s shot:%d run:%d, imas_version:%s)' % (
            user, machine, shot, run, imas_version), topic='imas')
    elif user is None and machine is None:
        printd('Saving to IMAS (shot:%d run:%d, DB:%s)' % (
            shot, run, os.environ.get('MDSPLUS_TREE_BASE_0', '???')[:-2]), topic='imas')

    # get the list of paths from ODS
    paths = set_paths = ods.paths()

    try:
        # open IMAS tree
        ids = imas_open(user=user, machine=machine, shot=shot, run=run, new=new, imas_version=imas_version)

    except IOError as _excp:
        raise(IOError(str(_excp)+'\nIf this is a new shot/run then set `new=True`'))

    except ImportError:
        # fallback on saving IMAS as NC file if IMAS is not installed
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join(
            [omas_rcparams['fake_imas_dir'],
             '%s_%s_%d_%d_v%s.pkl' % (user, machine, shot, run, imas_versions.get(imas_version,imas_version))])
        printe('Overloaded save_omas_imas: %s' % filename)
        from . import save_omas_pkl
        if not os.path.exists(omas_rcparams['fake_imas_dir']):
            os.makedirs(omas_rcparams['fake_imas_dir'])
        ods['info.user'] = unicode(user)
        ods['info.machine'] = unicode(machine)
        ods['info.shot'] = int(shot)
        ods['info.run'] = int(run)
        ods['info.imas_version'] = unicode(imas_version)
        save_omas_pkl(ods, filename)

    else:

        try:
            # allocate memory
            # NOTE: for how memory allocation works it is important to traverse the tree in reverse
            set_paths = []
            for path in reversed(paths):
                set_paths.append(imas_set(ids, path, ods[path], None, allocate=True))
            set_paths = filter(None, set_paths)

            # first assign time information
            for path in set_paths:
                if path[-1] == 'time':
                    printd('writing %s' % l2i(path))
                    imas_set(ids, path, ods[path], True)

            # then assign the rest
            for path in set_paths:
                if path[-1] != 'time':
                    printd('writing %s' % l2i(path))
                    imas_set(ids, path, ods[path], True)

            # actual write of IDS data to IMAS database
            for ds in ods.keys():
                if ds == 'info':
                    continue
                if 'imas'=='itm':
                    ds=ds+'Array'
                printd("ids.%s.put(0)"%ds,topic='imas_code')
                getattr(ids,ds).put(0)

        finally:
            # close connection to IMAS database
            printd("ids.close()",topic='imas_code')
            ids.close()

    return set_paths

def load_omas_imas(user=os.environ['USER'], machine=None, shot=None, run=0, paths=None,
                   imas_version=omas_rcparams['default_imas_version'], verbose=None):
    """
    load OMAS data set from IMAS

    :param user: IMAS username (default is os.environ['USER'])

    :param machine: IMAS machine (reads ods['info.machine'] if machine is None)

    :param shot: IMAS shot (reads ods['info.shot'] if shot is None)

    :param run: IMAS run (reads ods['info.run'] if run is None and finally fallsback on 0)

    :param paths: paths that have been written to IMAS

    :param imas_version: IMAS version
        (reads ods['info.imas_version'] if imas_version is None and finally fallsback on imas version of current system)

    :return: OMAS data set
    """

    if shot is None or run is None:
        raise (Exception('`shot` and `run` must be specified'))

    printd('Loading from IMAS (user:%s machine:%s shot:%d run:%d, imas_version:%s)' % (
        user, machine, shot, run, imas_version), topic='imas')

    try:
        ids = imas_open(user=user, machine=machine, shot=shot, run=run, new=False, imas_version=imas_version)

    except ImportError:
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join(
            [omas_rcparams['fake_imas_dir'],
             '%s_%s_%d_%d_v%s.pkl' % (user, machine, shot, run, imas_versions.get(imas_version,imas_version))])
        printe('Overloaded load_omas_imas: %s' % filename)
        from . import load_omas_pkl
        ods = load_omas_pkl(filename)

    else:

        try:
            # if paths is None then figure out what IDS are available and get ready to retrieve everything
            if paths is None:
                paths = [[structure] for structure in list_structures(imas_version=imas_version)]
                if verbose is None:
                    verbose=True
#            joined_paths = map(o2i, paths)
            joined_paths = map(l2i, paths)

            # fetch relevant IDSs and find available signals
            fetch_paths = []
            for path in paths:
                ds = path[0]
                path = path[1:]
                if ds=='info':
                    continue
                if not hasattr(ids,ds):
                    if verbose: print('| ', ds)
                    continue
                # ids fetching
                if not len(getattr(ids, ds).time):
                    printd("ids.%s.get()"%ds,topic='imas_code')
                    getattr(ids, ds).get()
                # ids discovery
                if len(getattr(ids, ds).time):
                    if verbose: print('* ', ds)
                    available_paths = filled_paths_in_ids(ids, load_structure(ds, imas_version=imas_version)[1], [], [])
#                    joined_available_paths = map(o2i, available_paths)
                    joined_available_paths = map(l2i, available_paths)
                    for jpath, path in zip(joined_paths, paths):
                        if path[0] != ds:
                            continue
                        jpath = jpath.replace('.','\.')
                        jpath = '^'+jpath.replace('.:', '.[0-9]+') + '.*'
                        for japath, apath in zip(joined_available_paths, available_paths):
                            if re.match(jpath, japath):
                                fetch_paths.append(apath)
                else:
                    if verbose: print('- ', ds)
#            joined_fetch_paths=map(o2i, fetch_paths)
            joined_fetch_paths=map(l2i, fetch_paths)

            # build omas data structure
            ods = ODS()
            for path in fetch_paths:
                if len(path)==2 and path[-1]=='time':
                    data = imas_get(ids, path, None)
                    if data[0]==-1:
                        continue
                if path[-1].endswith('_error_upper') or path[-1].endswith('_error_lower'):
                    continue
                # get data from ids
                data = imas_get(ids, path, None)
                # skip empty arrays
                if isinstance(data,numpy.ndarray) and not data.size:
                    continue
                # skip missing floats and integers
                elif (isinstance(data,float) and data==-9E40) or (isinstance(data,int) and data==-999999999):
                    continue
                # skip empty strings
                elif isinstance(data,unicode) and not len(data):
                    continue
                # add uncertainty
                if l2i(path[:-1]+[path[-1]+'_error_upper']) in joined_fetch_paths:
                    stdata=imas_get(ids, path[:-1]+[path[-1]+'_error_upper'], None)
                    if isinstance(stdata,numpy.ndarray) and not stdata.size:
                        pass
                    elif (isinstance(stdata,float) and stdata==-9E40) or (isinstance(stdata,int) and stdata==-999999999):
                        pass
                    elif isinstance(stdata,unicode) and not len(stdata):
                        continue
                    else:
                        data = uarray(data,stdata)
                #print(path,data)
                h = ods
                for step in path[:-1]:
                    h = h[step]
                h[path[-1]] = data

        finally:
            # close connection to IMAS database
            printd("ids.close()",topic='imas_code')
            ids.close()

    ods['info.user'] = unicode(user)
    ods['info.machine'] = unicode(machine)
    ods['info.shot'] = int(shot)
    ods['info.run'] = int(run)
    ods['info.imas_version'] = unicode(imas_version)

    return ods


def filled_paths_in_ids(ids, ds, path=None, paths=None):
    """
    list paths in an IDS that are filled

    :param ids: input ids

    :param ds: hierarchical data schema as returned for example by load_structure('equilibrium')[1]

    :return: returns list of paths in an IDS that are filled
    """
    if path is None:
        path = []
    if paths is None:
        paths = []
    if not len(ds):
        paths.append(path)
        #print(paths[-1])
        return paths
    keys = ds.keys()
    if keys[0] == ':':
        keys = range(len(ids))
    for kid in keys:
        propagate_path = copy.copy(path)
        propagate_path.append(kid)
        if isinstance(kid, basestring):
            paths = filled_paths_in_ids(getattr(ids, kid), ds[kid], propagate_path, paths)
        else:
            paths = filled_paths_in_ids(ids[kid], ds[':'], propagate_path, paths)
    return paths


def through_omas_imas(ods):
    """
    test save and load OMAS IMAS

    :param ods: ods

    :return: ods
    """
    user = os.environ['USER']
    machine = 'ITER'
    shot = 1
    run = 0

    paths = save_omas_imas(ods, user=user, machine=machine, shot=shot, run=run, new=True)
    ods1 = load_omas_imas(user=user, machine=machine, shot=shot, run=run, paths=paths)
    return ods1
