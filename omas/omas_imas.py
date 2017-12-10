from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import omas, save_omas_pkl, load_omas_pkl


# --------------------------------------------
# IMAS convenience functions
# --------------------------------------------
def imas_open(user, tokamak, shot, run, new=False, imas_version=default_imas_version):
    '''
    function to open an IMAS

    :param user: IMAS username

    :param tokamak: IMAS tokamak

    :param shot: IMAS shot

    :param run: IMAS run id

    :param new: whether the open should create a new IMAS tree

    :param imas_version: IMAS version

    :return: IMAS ids
    '''
    import imas
    ids = imas.ids()
    ids.setShot(shot)
    ids.setRun(run)
    if new:
        ids.create_env(user, tokamak, imas_version)
    else:
        ids.open_env(user, tokamak, imas_version)
    if not ids.isConnected():
        raise (Exception(
            'Failed to establish connection to IMAS database (user:%s tokamak:%s shot:%s run:%s, imas_version:%s)' % (
                user, tokamak, shot, run, imas_version)))
    return ids


def imas_set(ids, path, value, skipMissingNodes=False, allocate=False):
    '''
    assign a value to a path of an open IMAS ids

    :param ids: open IMAS ids to write to

    :param path: ODS path

    :param value: value to assign

    :param skipMissingNodes:  if the IMAS path does not exists:
                             `False` raise an error
                             `True` does not raise error
                             `None` prints a warning message

    :param allocate: whether to perform only IMAS memory allocation (ids.resize)

    :return: path if set was done, otherwise None
    '''
    printd('setting: %s' % repr(path), topic='imas')
    ds = path[0]
    path = path[1:]

    if ds == 'info':
        return
    if hasattr(ids, ds):
        m = getattr(ids, ds)
    elif skipMissingNodes is not False:
        if skipMissingNodes is None:
            printe('WARNING: %s is not part of IMAS structure' % o2i([ds] + path))
        return
    else:
        raise (AttributeError('%s is not part of IMAS structure' % o2i([ds] + path)))
    m.setExpIdx(0)

    out = m
    for kp, p in enumerate(path):
        if isinstance(p, basestring):
            if hasattr(out, p):
                if kp < (len(path) - 1):
                    out = getattr(out, p)
            elif skipMissingNodes is not False:
                if skipMissingNodes is None:
                    printe('WARNING: %s is not part of IMAS structure' % o2i([ds] + path))
                return None
            else:
                raise (AttributeError('%s is not part of IMAS structure' % o2i([ds] + path)))
        else:
            try:
                out = out[p]
            except IndexError:
                if not allocate:
                    raise (IndexError('%s structure array exceed allocation' % o2i([ds] + path)))
                printd('resizing: %d' % (p + 1), topic='imas')
                out.resize(p + 1)
                out = out[p]

    if allocate:
        return [ds] + path

    if isinstance(value, (basestring, numpy.ndarray)):
        setattr(out, path[-1], value)
    else:
        setattr(out, path[-1], numpy.array(value))
    try:
        m.put(0)
    except Exception:
        printe('Error setting: %s' % repr(path))
        raise
    return [ds] + path


def imas_get(ids, path, skipMissingNodes=False):
    '''
    read the value of a path in an open IMAS ids

    :param ids: open IMAS ids to read from

    :param path: ODS path

    :param skipMissingNodes:  if the IMAS path does not exists:
                             `False` raise an error
                             `True` does not raise error
                             `None` prints a warning message

    :return: the value that was read if successful or None otherwise
    '''
    printd('fetching: %s' % repr(path), topic='imas')
    ds = path[0]
    path = path[1:]

    if hasattr(ids, ds):
        m = getattr(ids, ds)
    elif skipMissingNodes is not False:
        if skipMissingNodes is None:
            printe('WARNING: %s is not part of IMAS structure' % o2i([ds] + path))
        return None
    else:
        raise (AttributeError('%s is not part of IMAS structure' % o2i([ds] + path)))

    m.get()

    out = m
    for kp, p in enumerate(path):
        if isinstance(p, basestring):
            if hasattr(out, p):
                out = getattr(out, p)
            elif skipMissingNodes is not False:
                if skipMissingNodes is None:
                    printe('WARNING: %s is not part of IMAS structure' % o2i([ds] + path[:kp + 1]))
                    printe(out.__dict__.keys())
                return None
            else:
                raise (AttributeError('%s is not part of IMAS structure' % o2i([ds] + path[:kp + 1])))
        else:
            out = out[p]

    printd('data: ' + repr(out), topic='imas')
    return out


# --------------------------------------------
# save and load OMAS to IMAS
# --------------------------------------------
def save_omas_imas(ods, user=None, tokamak=None, shot=None, run=None, new=False, imas_version=default_imas_version):
    '''
    save OMAS data set to IMAS

    :param ods: OMAS data set

    :param user: IMAS username (reads ods['info.user'] if user is None and finally fallsback on os.environ['USER'])

    :param tokamak: IMAS tokamak (reads ods['info.tokamak'] if tokamak is None)

    :param shot: IMAS shot (reads ods['info.shot'] if shot is None)

    :param run: IMAS run (reads ods['info.run'] if run is None and finally fallsback on 0)

    :param new: whether the open should create a new IMAS tree

    :param imas_version: IMAS version (reads ods['info.imas_version'] if imas_version is None and finally fallsback on imas version of current system)

    :return: paths that have been written to IMAS
    '''

    if user is None:
        user = ods.get('info.user', os.environ['USER'])
    if tokamak is None:
        tokamak = ods.get('info.tokamak', None)
    if shot is None:
        shot = ods.get('info.shot', None)
    if run is None:
        run = ods.get('info.run', 0)

    printd('Saving to IMAS (user:%s tokamak:%s shot:%d run:%d, imas_version:%s)' % (
        user, tokamak, shot, run, imas_version), topic='imas')

    paths = set_paths = ods.paths()

    try:
        ids = imas_open(user=user, tokamak=tokamak, shot=shot, run=run, new=new, imas_version=imas_version)

    except ImportError:
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join(
            [omas_rcparams['fake_imas_dir'],
             '%s_%s_%d_%d_v%s.nc' % (user, tokamak, shot, run, re.sub('\.', '_', imas_version))])
        printe('Overloaded save_omas_imas: %s' % filename)
        from . import save_omas_nc
        if not os.path.exists(omas_rcparams['fake_imas_dir']):
            os.makedirs(omas_rcparams['fake_imas_dir'])
        ods['info.user'] = unicode(user)
        ods['info.tokamak'] = unicode(tokamak)
        ods['info.shot'] = int(shot)
        ods['info.run'] = int(run)
        ods['info.imas_version'] = unicode(imas_version)
        save_omas_nc(ods, filename)

    else:
        set_paths = []
        for path in paths:
            set_paths.append(imas_set(ids, path, ods[path], None, allocate=True))
        set_paths = filter(None, set_paths)

        for path in set_paths:
            if 'time' in path[:1] or path[-1] != 'time':
                continue
            printd('writing %s' % o2i(path))
            imas_set(ids, path, ods[path], True)
        for path in set_paths:
            if 'time' in path[:1] or path[-1] == 'time':
                continue
            printd('writing %s' % o2i(path))
            imas_set(ids, path, ods[path], True)

    return set_paths


def load_omas_imas(user=os.environ['USER'], tokamak=None, shot=None, run=0, paths=None,
                   imas_version=default_imas_version):
    '''
    load OMAS data set from IMAS

    :param user: IMAS username (reads ods['info.user'] if user is None and finally fallsback on os.environ['USER'])

    :param tokamak: IMAS tokamak (reads ods['info.tokamak'] if tokamak is None)

    :param shot: IMAS shot (reads ods['info.shot'] if shot is None)

    :param run: IMAS run (reads ods['info.run'] if run is None and finally fallsback on 0)

    :param paths: paths that have been written to IMAS

    :param imas_version: IMAS version (reads ods['info.imas_version'] if imas_version is None and finally fallsback on imas version of current system)

    :return: OMAS data set
    '''
    if paths is None:
        raise (Exception('Must specify paths to load'))

    printd('Loading from IMAS (user:%s tokamak:%s shot:%d run:%d, imas_version:%s)' % (
        user, tokamak, shot, run, imas_version), topic='imas')

    try:
        ids = imas_open(user=user, tokamak=tokamak, shot=shot, run=run, new=False, imas_version=imas_version)

    except ImportError:
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join(
            [omas_rcparams['fake_imas_dir'],
             '%s_%s_%d_%d_v%s.nc' % (user, tokamak, shot, run, re.sub('\.', '_', imas_version))])
        printe('Overloaded load_omas_imas: %s' % filename)
        from . import load_omas_nc
        ods = load_omas_nc(filename)

    else:
        ods = omas()
        for path in paths:
            data = imas_get(ids, path, None)
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


def test_omas_imas(ods):
    '''
    test save and load OMAS IMAS

    :param ods: ods

    :return: ods
    '''
    user = os.environ['USER']
    tokamak = 'ITER'
    shot = 1
    run = 0

    paths = ods.paths()
    paths = save_omas_imas(ods, user=user, tokamak=tokamak, shot=shot, run=run, new=True)
    ods1 = load_omas_imas(user=user, tokamak=tokamak, shot=shot, run=run, paths=paths)
    return ods1


# --------------------------------------------
if __name__ == '__main__':
    print('=' * 20)

    from omas_core import ods_sample

    os.environ['OMAS_DEBUG_TOPIC'] = 'imas'
    ods = ods_sample()

    ods = test_omas_imas(ods)
