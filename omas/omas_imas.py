from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import omas


# --------------------------------------------
# IMAS convenience functions
# --------------------------------------------
def imas_open(user, tokamak, shot, run, new=False, imas_version=default_imas_version):
    """
    function to open an IMAS

    :param user: IMAS username

    :param tokamak: IMAS tokamak

    :param shot: IMAS shot

    :param run: IMAS run id

    :param new: whether the open should create a new IMAS tree

    :param imas_version: IMAS version

    :return: IMAS ids
    """
    import imas
    printd("ids = imas.ids()",topic='imas_code')
    ids = imas.ids()
    printd("ids.setShot(%s)"%shot,topic='imas_code')
    ids.setShot(shot)
    printd("ids.setRun(%s)"%run,topic='imas_code')
    ids.setRun(run)

    if user is None and tokamak is None:
        pass
    elif user is None or tokamak is None:
        raise (Exception('user={user}, tokamak={tokamak}, imas_version={imas_version}\n'
                         'Either specify all or none of `user`, `tokamak`, `imas_version`\n'
                         'If none of them are specified then use `imasdb` command to set '
                         'MDSPLUS_TREE_BASE_? environmental variables'.format(user=user, tokamak=tokamak, shot=shot,
                                                                              run=run, imas_version=imas_version)))

    if user is None and tokamak is None:
        if new:
            printd("ids.create()",topic='imas_code')
            ids.create()
        else:
            printd("ids.open()",topic='imas_code')
            ids.open()
        if not ids.isConnected():
            raise (Exception(
                'Failed to establish connection to IMAS database'
                '(shot:{shot} run:{run}, DB:{db})'.format(
                    shot=shot, run=run, db=os.environ.get('MDSPLUS_TREE_BASE_0', '???')[:-2])))

    else:
        if new:
            printd("ids.create_env(%s, %s, %s)"%(repr(user),repr(tokamak),repr(imas_version)),topic='imas_code')
            ids.create_env(user, tokamak, imas_version)
        else:
            printd("ids.open_env(%s, %s, %s)"%(repr(user),repr(tokamak),repr(imas_version)),topic='imas_code')
            ids.open_env(user, tokamak, imas_version)

    if not ids.isConnected():
        raise (Exception(
            'Failed to establish connection to IMAS database (user:%s tokamak:%s shot:%s run:%s, imas_version:%s)' % (
                user, tokamak, shot, run, imas_version)))
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
    ds = path[0]
    path = path[1:]

    # `info` IDS is used by OMAS to hold user, tokamak, shot, run, imas_version
    # for saving methods that do not carry that information. IMAS does not store
    # this information as part of the data dictionary.
    if ds == 'info':
        return

    # identify data dictionary to use, from this point on `m` points to the IDS
    if hasattr(ids, ds):
        printd("",topic='imas_code')
        printd("m = getattr(ids, %s)"%repr(ds),topic='imas_code')
        m = getattr(ids, ds)
        if not m.time.size:
            m.time.resize(1)
            m.time[0]=-1.0
    elif skip_missing_nodes is not False:
        if skip_missing_nodes is None:
            printe('WARNING: %s is not part of IMAS structure' % o2i([ds] + path))
        return
    else:
        raise (AttributeError('%s is not part of IMAS structure' % o2i([ds] + path)))

    # traverse IMAS structure until reaching the leaf
    printd("out = m",topic='imas_code')
    out = m
    for kp, p in enumerate(path):
        location=o2i([ds] + path[:kp+1])
        if isinstance(p, basestring):
            if hasattr(out, p):
                if kp < (len(path) - 1):
                    printd("out = getattr(out, %s)"%repr(p),topic='imas_code')
                    out = getattr(out, p)
            elif skip_missing_nodes is not False:
                if skip_missing_nodes is None:
                    printe('WARNING: %s is not part of IMAS structure' % location)
                return None
            else:
                raise (AttributeError('%s is not part of IMAS structure' % location))
        else:
            try:
                out = out[p]
                printd("out = out[%s]"%p,topic='imas_code')
            except IndexError:
                if not allocate:
                    raise (IndexError('%s structure array exceed allocation' % location))
                printd('resizing  : %s'%location, topic='imas')
                printd("out.resize(%s + 1)"%p,topic='imas_code')
                out.resize(p + 1)
                printd("out = out[%s]"%p,topic='imas_code')
                out = out[p]

    # if we are allocating data, simply stop here
    if allocate:
        return [ds] + path

    # assign data to leaf node
    printd('setting  : %s'%location, topic='imas')
    if isinstance(value, (basestring, numpy.ndarray)):
        printd("setattr(out, %s, %s)"%(repr(path[-1]),value),topic='imas_code')
        setattr(out, path[-1], value)
    else:
        printd("setattr(out, %s, %s)"%(repr(path[-1]),repr(numpy.array(value))),topic='imas_code')
        setattr(out, path[-1], numpy.array(value))

    # write the data to IMAS
    try:
        printd("m.put(0)",topic='imas_code')
        m.put(0)
    except Exception:
        printe('Error %s: %s' %(['setting   ','allocating'][allocate],repr(path)))
        raise

    # return path
    return [ds] + path


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
    printd('fetching: %s' % o2i(path), topic='imas')
    ds = path[0]
    path = path[1:]

    if hasattr(ids, ds):
        printd("m = getattr(ids, %s)"%repr(ds),topic='imas_code')
        m = getattr(ids, ds)
    elif skip_missing_nodes is not False:
        if skip_missing_nodes is None:
            printe('WARNING: %s is not part of IMAS structure' % o2i([ds] + path))
        return None
    else:
        raise (AttributeError('%s is not part of IMAS structure' % o2i([ds] + path)))

    # use time to figure out if this IDS has data
    if not len(m.time):
        printd("m.get()",topic='imas_code')
        m.get()

    # traverse the IDS to get the data
    out = m
    for kp, p in enumerate(path):
        if isinstance(p, basestring):
            if hasattr(out, p):
                printd("out = getattr(out, %s)"%repr(p),topic='imas_code')
                out = getattr(out, p)
            elif skip_missing_nodes is not False:
                if skip_missing_nodes is None:
                    printe('WARNING: %s is not part of IMAS structure' % o2i([ds] + path[:kp + 1]))
                    printe(out.__dict__.keys())
                return None
            else:
                raise (AttributeError('%s is not part of IMAS structure' % o2i([ds] + path[:kp + 1])))
        else:
            printd("out = out[%s]"%p,topic='imas_code')
            out = out[p]

    return out


# --------------------------------------------
# save and load OMAS to IMAS
# --------------------------------------------
def save_omas_imas(ods, user=None, tokamak=None, shot=None, run=None, new=False, imas_version=default_imas_version):
    """
    save OMAS data set to IMAS

    :param ods: OMAS data set

    :param user: IMAS username (reads ods['info.user'] if user is None and finally fallsback on os.environ['USER'])

    :param tokamak: IMAS tokamak (reads ods['info.tokamak'] if tokamak is None)

    :param shot: IMAS shot (reads ods['info.shot'] if shot is None)

    :param run: IMAS run (reads ods['info.run'] if run is None and finally fallsback on 0)

    :param new: whether the open should create a new IMAS tree

    :param imas_version: IMAS version
                         (reads ods['info.imas_version'] if imas_version is None
                          and finally fallsback on imas version of current system)

    :return: paths that have been written to IMAS
    """

    # handle default values for user, tokamak, shot, run, imas_version
    # it tries to re-use existing information
    if user is None:
        user = ods.get('info.user', os.environ['USER'])
    if tokamak is None:
        tokamak = ods.get('info.tokamak', None)
    if shot is None:
        shot = ods.get('info.shot', None)
    if run is None:
        run = ods.get('info.run', 0)

    if user is not None and tokamak is not None:
        printd('Saving to IMAS (user:%s tokamak:%s shot:%d run:%d, imas_version:%s)' % (
            user, tokamak, shot, run, imas_version), topic='imas')
    elif user is None and tokamak is None:
        printd('Saving to IMAS (shot:%d run:%d, DB:%s)' % (
            shot, run, os.environ.get('MDSPLUS_TREE_BASE_0', '???')[:-2]), topic='imas')

    # get the list of paths from ODS
    paths = set_paths = ods.paths()

    try:
        # open IMAS tree
        ids = imas_open(user=user, tokamak=tokamak, shot=shot, run=run, new=new, imas_version=imas_version)

    except ImportError:
        # fallback on saving IMAS as NC file if IMAS is not installed
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join(
            [omas_rcparams['fake_imas_dir'],
             '%s_%s_%d_%d_v%s.pkl' % (user, tokamak, shot, run, re.sub('\.', '_', imas_version))])
        printe('Overloaded save_omas_imas: %s' % filename)
        from . import save_omas_pkl
        if not os.path.exists(omas_rcparams['fake_imas_dir']):
            os.makedirs(omas_rcparams['fake_imas_dir'])
        ods['info.user'] = unicode(user)
        ods['info.tokamak'] = unicode(tokamak)
        ods['info.shot'] = int(shot)
        ods['info.run'] = int(run)
        ods['info.imas_version'] = unicode(imas_version)
        save_omas_pkl(ods, filename)

    else:
        # allocate memory
        # NOTE: for how memory allocation works it is important to traverse the tree in reverse
        set_paths = []
        for path in reversed(paths):
            set_paths.append(imas_set(ids, path, ods[path], None, allocate=True))
        set_paths = filter(None, set_paths)

        # first assign time information
        for path in set_paths:
            if path[-1] == 'time':
                printd('writing %s' % o2i(path))
                imas_set(ids, path, ods[path], True)

        # then assign the rest
        for path in set_paths:
            if path[-1] != 'time':
                printd('writing %s' % o2i(path))
                imas_set(ids, path, ods[path], True)

    return set_paths


def load_omas_imas(user=os.environ['USER'], tokamak=None, shot=None, run=0, paths=None,
                   imas_version=default_imas_version):
    """
    load OMAS data set from IMAS

    :param user: IMAS username (reads ods['info.user'] if user is None and finally fallsback on os.environ['USER'])

    :param tokamak: IMAS tokamak (reads ods['info.tokamak'] if tokamak is None)

    :param shot: IMAS shot (reads ods['info.shot'] if shot is None)

    :param run: IMAS run (reads ods['info.run'] if run is None and finally fallsback on 0)

    :param paths: paths that have been written to IMAS

    :param imas_version: IMAS version
                         (reads ods['info.imas_version'] if imas_version is None
                          and finally fallsback on imas version of current system)

    :return: OMAS data set
    """

    if shot is None or run is None:
        raise (Exception('`shot` and `run` must be specified'))

    printd('Loading from IMAS (user:%s tokamak:%s shot:%d run:%d, imas_version:%s)' % (
        user, tokamak, shot, run, imas_version), topic='imas')

    try:
        ids = imas_open(user=user, tokamak=tokamak, shot=shot, run=run, new=False, imas_version=imas_version)

    except ImportError:
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join(
            [omas_rcparams['fake_imas_dir'],
             '%s_%s_%d_%d_v%s.pkl' % (user, tokamak, shot, run, re.sub('\.', '_', imas_version))])
        printe('Overloaded load_omas_imas: %s' % filename)
        from . import load_omas_pkl
        ods = load_omas_pkl(filename)

    else:
        # if paths is None then figure out what IDS are available and get ready to retrieve everything
        verbose=False
        if paths is None:
            paths = sorted([[structure] for structure in list_structures(imas_version=imas_version)])
            verbose=True
        joined_paths = map(lambda x: separator.join(map(str, x)), paths)

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
            if not len(getattr(ids, ds).time):
                getattr(ids, ds).get()
            if len(getattr(ids, ds).time):
                if verbose: print('* ', ds)
                available_paths = filled_paths_in_ids(ids, load_structure(ds, imas_version=imas_version)[1], [], [])
                joined_available_paths = map(lambda x: separator.join(map(str, x)), available_paths)
                for jpath, path in zip(joined_paths, paths):
                    if path[0] != ds:
                        continue
                    jpath = re.sub('\.', '\\.', jpath)
                    jpath = '^' + re.sub('.:', '.[0-9]+', jpath) + '.*'
                    for japath, apath in zip(joined_available_paths, available_paths):
                        if re.match(jpath, japath):
                            fetch_paths.append(apath)
            else:
                if verbose: print('- ', ds)

        # build omas data structure
        ods = omas()
        for path in fetch_paths:
            if len(path)==2 and path[-1]=='time':
                data = imas_get(ids, path, None)
                if data[0]==-1:
                    continue
            # skip _error_upper and _error_lower if _error_index=-999999999
            if path[-1].endswith('_error_upper') or path[-1].endswith('_error_lower'):
                data = imas_get(ids, path[:-1]+['_error_'.join(path[-1].split('_error_')[:-1])+'_error_index'], None)
                if data==-999999999:
                    continue
            # get data from ids
            data = imas_get(ids, path, None)
            # skip empty arrays
            if isinstance(data,numpy.ndarray) and not data.size:
                continue
            # skip missing floats and integers
            if (isinstance(data,float) and data==-9E40) or (isinstance(data,int) and data==-999999999):
                continue
            # skip empty strings
            if isinstance(data,unicode) and not len(data):
                continue
            #print(path,data)
            h = ods
            for step in path[:-1]:
                h = h[step]
            h[path[-1]] = data

    ods['info.user'] = unicode(user)
    ods['info.tokamak'] = unicode(tokamak)
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


def test_omas_imas(ods):
    """
    test save and load OMAS IMAS

    :param ods: ods

    :return: ods
    """
    user = os.environ['USER']
    tokamak = 'ITER'
    shot = 1
    run = 0

    paths = save_omas_imas(ods, user=user, tokamak=tokamak, shot=shot, run=run, new=True)
    ods1 = load_omas_imas(user=user, tokamak=tokamak, shot=shot, run=run, paths=paths)
    return ods1
