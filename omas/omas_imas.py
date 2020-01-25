'''save/load from IMAS routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS, codeparams_xml_save, codeparams_xml_load


# --------------------------------------------
# IMAS convenience functions
# --------------------------------------------
def imas_open(user, machine, pulse, run, new=False, imas_major_version='3', verbose=True):
    """
    function to open an IMAS

    :param user: IMAS username

    :param machine: IMAS machine

    :param pulse: IMAS pulse

    :param run: IMAS run id

    :param new: whether the open should create a new IMAS tree

    :param imas_major_version: IMAS major version

    :param verbose: print open parameters

    :return: IMAS ids
    """
    if verbose:
        print('Opening {new} IMAS data for user={user} machine={machine} pulse={pulse} run={run}'.format(new=['existing', 'new'][int(new)], user=repr(user), machine=repr(machine), pulse=pulse, run=run))

    import imas
    printd("ids = imas.ids(%d,%d)" % (pulse, run), topic='imas_code')
    ids = imas.ids(pulse, run)

    if user is None and machine is None:
        pass
    elif user is None or machine is None:
        raise Exception('user={user}, machine={machine}, imas_major_version={imas_major_version}\n'
                        'Either specify all or none of `user`, `machine`, `imas_version`\n'
                        'If none of them are specified then use `imasdb` command to set '
                        'MDSPLUS_TREE_BASE_? environmental variables'.format(user=repr(user), machine=repr(machine), pulse=pulse,
                                                                             run=run, imas_major_version=imas_major_version))

    # This approach of opening IDSs has been deprecated
    if user is None and machine is None:
        if new:
            printd("ids.create()", topic='imas_code')
            ids.create()
        else:
            printd("ids.open()", topic='imas_code')
            try:
                ids.open()
            except Exception as _excp:
                if 'Error opening imas pulse' in str(_excp):
                    raise IOError('Error opening imas pulse %d run %d' % (pulse, run))
        if not ids.isConnected():
            raise Exception('Failed to establish connection to IMAS database '
                            '(pulse:{pulse} run:{run}, DB:{db})'.format(pulse=pulse, run=run, db=os.environ.get('MDSPLUS_TREE_BASE_0', '???')[:-2]))

    # The new approach always requires specifying user and machine
    else:
        if new:
            printd("ids.create_env(%s, %s, %s)" % (repr(user), repr(machine), repr(imas_major_version)), topic='imas_code')
            ids.create_env(user, machine, imas_major_version)
        else:
            printd("ids.open_env(%s, %s, %s)" % (repr(user), repr(machine), repr(imas_major_version)), topic='imas_code')
            try:
                ids.open_env(user, machine, imas_major_version)
            except Exception as _excp:
                if 'Error opening imas pulse' in str(_excp):
                    raise IOError('Error opening imas pulse (user:%s machine:%s pulse:%s run:%s, imas_major_version:%s)' % (user, machine, pulse, run, imas_major_version))
        if not ids.isConnected():
            raise Exception('Failed to establish connection to IMAS database (user:%s machine:%s pulse:%s run:%s, imas_major_version:%s)' % (user, machine, pulse, run, imas_major_version))
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
        path = copy.deepcopy(path)
        tmp = imas_set(ids, path, nominal_values(value), skip_missing_nodes=skip_missing_nodes, allocate=allocate)
        path[-1] = path[-1] + '_error_upper'
        imas_set(ids, path, std_devs(value), skip_missing_nodes=skip_missing_nodes, allocate=allocate)
        return tmp

    ds = path[0]
    path = path[1:]

    # `info` IDS is used by OMAS to hold user, machine, pulse, run, imas_version
    # for saving methods that do not carry that information. IMAS does not store
    # this information as part of the data dictionary.
    if ds in add_datastructures.keys():
        return

    # identify data dictionary to use, from this point on `m` points to the IDS
    debug_path = ''
    if hasattr(ids, ds):
        debug_path += 'ids.%s' % ds
        m = getattr(ids, ds)
        if hasattr(m, 'time') and not isinstance(m.time, float) and not m.time.size:
            m.time.resize(1)
            m.time[0] = -1.0
    elif skip_missing_nodes is not False:
        if skip_missing_nodes is None:
            printe('WARNING: %s is not part of IMAS' % l2i([ds] + path))
        return
    else:
        printd(debug_path, topic='imas_code')
        raise AttributeError('%s is not part of IMAS' % l2i([ds] + path))

    # traverse IMAS structure until reaching the leaf
    out = m
    for kp, p in enumerate(path):
        location = l2i([ds] + path[:kp + 1])
        if isinstance(p, basestring):
            if hasattr(out, p):
                if kp < (len(path) - 1):
                    debug_path += '.' + p
                    out = getattr(out, p)
            elif skip_missing_nodes is not False:
                if skip_missing_nodes is None:
                    printe('WARNING: %s is not part of IMAS' % location)
                return
            else:
                printd(debug_path, topic='imas_code')
                raise AttributeError('%s is not part of IMAS' % location)
        else:
            try:
                out = out[p]
                debug_path += '[%d]' % p
            except IndexError:
                if not allocate:
                    raise IndexError('%s structure array exceed allocation' % location)
                printd(debug_path + ".resize(%d)" % (p + 1), topic='imas_code')
                out.resize(p + 1)
                debug_path += '[%d]' % p
                out = out[p]

    # if we are allocating data, simply stop here
    if allocate:
        return [ds] + path

    # assign data to leaf node
    printd('setting  : %s' % location, topic='imas')
    setattr(out, path[-1], value)
    if 'imas_code' in os.environ.get('OMAS_DEBUG_TOPIC', ''):  # use if statement here to avoid unecessary repr(value) when not debugging
        printd(debug_path + '.%s=%s' % (path[-1], repr(value).replace('\\n', '\n')), topic='imas_code')

    # return path
    return [ds] + path


def imas_empty(value):
    '''
    Check if value is an IMAS empty
        * array with no size
        * float of value -9E40
        * integer of value -999999999
        * empty string

    :param value: value to check

    :return: None if value is an IMAS empty
    '''
    if isinstance(value, numpy.ndarray) and not value.size:
        value = None
    # missing floats and integers
    elif (isinstance(value, float) and value == -9E40) or (isinstance(value, int) and value == -999999999):
        value = None
    # empty strings
    elif isinstance(value, basestring) and not len(value):
        value = None
    return value


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

    debug_path = ''
    if hasattr(ids, ds):
        debug_path += 'ids.%s' % ds
        m = getattr(ids, ds)
    elif skip_missing_nodes is not False:
        if skip_missing_nodes is None:
            printe('WARNING: %s is not part of IMAS' % l2i([ds] + path))
        return None
    else:
        printd(debug_path, topic='imas_code')
        raise AttributeError('%s is not part of IMAS' % l2i([ds] + path))

    # traverse the IDS to get the data
    out = m
    for kp, p in enumerate(path):
        if isinstance(p, basestring):
            if hasattr(out, p):
                debug_path += '.%s' % p
                out = getattr(out, p)
            elif skip_missing_nodes is not False:
                if skip_missing_nodes is None:
                    printe('WARNING: %s is not part of IMAS' % l2i([ds] + path[:kp + 1]))
                    printe(out.__dict__.keys())
                return None
            else:
                printd(debug_path, topic='imas_code')
                raise AttributeError('%s is not part of IMAS' % l2i([ds] + path[:kp + 1]))
        else:
            debug_path += '[%s]' % p
            out = out[p]

    # handle missing data
    data = imas_empty(out)

    printd(debug_path, topic='imas_code')
    return data


# --------------------------------------------
# save and load OMAS to IMAS
# --------------------------------------------
@codeparams_xml_save
def save_omas_imas(ods, user=None, machine=None, pulse=None, run=None, new=False, imas_version=None, verbose=True):
    """
    Save OMAS data to IMAS

    :param ods: OMAS data set

    :param user: IMAS username (reads ods['dataset_description.data_entry.user'] if user is None and finally fallsback on os.environ['USER'])

    :param machine: IMAS machine (reads ods['dataset_description.data_entry.machine'] if machine is None)

    :param pulse: IMAS pulse (reads ods['dataset_description.data_entry.pulse'] if pulse is None)

    :param run: IMAS run (reads ods['dataset_description.data_entry.run'] if run is None and finally fallsback on 0)

    :param new: whether the open should create a new IMAS tree

    :param imas_version: IMAS version

    :param verbose: whether the process should be verbose

    :return: paths that have been written to IMAS
    """

    # handle default values for user, machine, pulse, run, imas_version
    # it tries to re-use existing information
    if user is None:
        user = ods.get('dataset_description.data_entry.user', os.environ['USER'])
    if machine is None:
        machine = ods.get('dataset_description.data_entry.machine', None)
    if pulse is None:
        pulse = ods.get('dataset_description.data_entry.pulse', None)
    if run is None:
        run = ods.get('dataset_description.data_entry.run', 0)
    if imas_version is None:
        imas_version = ods.imas_version

    # set dataset_description entries that were empty
    if user is not None and 'dataset_description.data_entry.user' not in ods:
        ods['dataset_description.data_entry.user'] = user
    if machine is not None and 'dataset_description.data_entry.machine' not in ods:
        ods['dataset_description.data_entry.machine'] = machine
    if pulse is not None and 'dataset_description.data_entry.pulse' not in ods:
        ods['dataset_description.data_entry.pulse'] = pulse
    if run is not None and 'dataset_description.data_entry.run' not in ods:
        ods['dataset_description.data_entry.run'] = run
    if imas_version is not None and 'dataset_description.imas_version' not in ods:
        ods['dataset_description.imas_version'] = ods.imas_version

    if user is not None and machine is not None:
        printd('Saving to IMAS (user:%s machine:%s pulse:%d run:%d, imas_version:%s)' % (user, machine, pulse, run, imas_version), topic='imas')
    elif user is None and machine is None:
        printd('Saving to IMAS (pulse:%d run:%d, DB:%s)' % (pulse, run, os.environ.get('MDSPLUS_TREE_BASE_0', '???')[:-2]), topic='imas')

    # ensure requirements for writing data to IMAS are satisfied
    ods.satisfy_imas_requirements()

    # get the list of paths from ODS
    paths = set_paths = ods.paths()

    try:
        # open IMAS tree
        ids = imas_open(user=user, machine=machine, pulse=pulse, run=run, new=new, verbose=verbose)

    except IOError as _excp:
        raise IOError(str(_excp) + '\nIf this is a new pulse/run then set `new=True`')

    except ImportError:
        # fallback on saving IMAS as NC file if IMAS is not installed
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join([omas_rcparams['fake_imas_dir'], '%s_%s_%d_%d_v%s.pkl' % (user, machine, pulse, run, imas_versions.get(imas_version, imas_version))])
        printe('Overloaded save_omas_imas: %s' % filename)
        from . import save_omas_pkl
        if not os.path.exists(omas_rcparams['fake_imas_dir']):
            os.makedirs(omas_rcparams['fake_imas_dir'])
        ods['dataset_description.data_entry.user'] = unicode(user)
        ods['dataset_description.data_entry.machine'] = unicode(machine)
        ods['dataset_description.data_entry.pulse'] = int(pulse)
        ods['dataset_description.data_entry.run'] = int(run)
        ods['dataset_description.imas_version'] = unicode(imas_version)
        save_omas_pkl(ods, filename)

    else:

        try:
            # allocate memory
            # NOTE: for how memory allocation works it is important to traverse the tree in reverse
            set_paths = []
            for path in reversed(paths):
                set_paths.append(imas_set(ids, path, ods[path], None, allocate=True))
            set_paths = filter(None, set_paths)

            # assign the data
            for path in set_paths:
                printd('writing %s' % l2i(path))
                imas_set(ids, path, ods[path], True)

            # actual write of IDS data to IMAS database
            for ds in ods.keys():
                if ds in add_datastructures.keys():
                    continue
                printd("ids.%s.put(0)" % ds, topic='imas_code')
                getattr(ids, ds).put(0)

        finally:
            # close connection to IMAS database
            printd("ids.close()", topic='imas_code')
            ids.close()

    return set_paths


def infer_fetch_paths(ids, paths, imas_version, skip_ggd=True, skip_ion_state=True, verbose=True):
    """
    Return list of IMAS paths that have data

    :param ids: IMAS ids

    :param paths: list of paths to load from IMAS

    :param imas_version: IMAS version

    :param skip_ggd: do not load ggd structure

    :param skip_ion_state: do not load ion state structure

    :param verbose: print ids infos

    :return: list of paths that have data
    """
    # if paths is None then figure out what IDS are available and get ready to retrieve everything
    if paths is None:
        requested_paths = [[structure] for structure in list_structures(imas_version=imas_version)]
    else:
        requested_paths = list(map(p2l, paths))

    # fetch relevant IDSs and find available signals
    fetch_paths = []
    dss = numpy.unique([p[0] for p in requested_paths])
    ndss = max([len(d) for d in dss])
    for ds in dss:
        if ds in add_datastructures.keys():
            continue
        if not hasattr(ids, ds):
            if verbose:
                print('| %s IDS of IMAS version %s is unknown' % (ds.ljust(ndss), imas_version))
            continue
        # ids fetching
        printd("ids.%s.get()" % ds, topic='imas_code')
        try:
            getattr(ids, ds).get()
        except ValueError as _excp:
            print('x %s IDS failed on get' % ds.ljust(ndss))  # not sure why some IDSs fail on .get()... it's not about them being empty
            continue
        if getattr(ids, ds).ids_properties.homogeneous_time != -999999999:
            if verbose:
                try:
                    print('* %s IDS has data (%d times)' % (ds.ljust(ndss), len(getattr(ids, ds).time)))
                except Exception as _excp:
                    print('* %s IDS' % ds.ljust(ndss))
                fetch_paths += filled_paths_in_ids(ids, load_structure(ds, imas_version=imas_version)[1], [], [], requested_paths, skip_ggd=skip_ggd, skip_ion_state=skip_ion_state)
        else:
            if verbose:
                print('- %s IDS is empty' % ds.ljust(ndss))
    joined_fetch_paths = list(map(l2i, fetch_paths))
    return fetch_paths, joined_fetch_paths


@codeparams_xml_load
def load_omas_imas(user=os.environ.get('USER', 'dummy_user'), machine=None, pulse=None, run=0, paths=None,
                   imas_version=None, skip_uncertainties=False, skip_ggd=True, skip_ion_state=True, verbose=True):
    """
    Load OMAS data from IMAS

    NOTE: Either specify both or none of `user` and `machine`
    If none of them are specified then use `imasdb` command to set the `MDSPLUS_TREE_BASE_?` environmental variables

    :param user: IMAS username

    :param machine: IMAS machine

    :param pulse: IMAS pulse

    :param run: IMAS run

    :param paths: list of paths to load from IMAS

    :param imas_version: IMAS version (force specific version)

    :param skip_uncertainties: do not load uncertain data

    :param skip_ggd: do not load ggd structure

    :param skip_ion_state: do not load ion_state structure

    :param verbose: print loading progress

    :return: OMAS data set
    """

    if pulse is None or run is None:
        raise Exception('`pulse` and `run` must be specified')

    printd('Loading from IMAS (user:%s machine:%s pulse:%d run:%d, imas_version:%s)' % (user, machine, pulse, run, imas_version), topic='imas')

    try:
        ids = imas_open(user=user, machine=machine, pulse=pulse, run=run, new=False, verbose=verbose)

        if imas_version is None:
            try:
                imas_version = ids.dataset_description.imas_version
                if not imas_version:
                    imas_version = os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version'])
                    if verbose:
                        print('dataset_description.imas_version is missing: assuming IMAS version %s' % imas_version)
                else:
                    print('%s IMAS version detected' % imas_version)
            except Exception:
                raise

    except ImportError:
        if imas_version is None:
            imas_version = os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version'])
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join([omas_rcparams['fake_imas_dir'], '%s_%s_%d_%d_v%s.pkl' % (user, machine, pulse, run, imas_versions.get(imas_version, imas_version))])
        printe('Overloaded load_omas_imas: %s' % filename)
        from . import load_omas_pkl
        ods = load_omas_pkl(filename)

    else:

        try:
            # see what paths have data
            fetch_paths, joined_fetch_paths = infer_fetch_paths(ids, paths=paths, imas_version=imas_version, skip_ggd=skip_ggd, skip_ion_state=skip_ion_state, verbose=verbose)

            # build omas data structure
            ods = ODS(imas_version=imas_version, consistency_check=False)
            for k, path in enumerate(fetch_paths):
                if path[-1].endswith('_error_upper') or path[-1].endswith('_error_lower') or path[-1].endswith('_error_index'):
                    continue
                if verbose and (k % (len(fetch_paths) // 10) == 0 or k == len(fetch_paths) - 1):
                    print('Loading {0:3.1f}%'.format(100 * float(k) / (len(fetch_paths) - 1)))
                # get data from IDS
                data = imas_get(ids, path, None)
                # continue for empty data
                if data is None:
                    continue
                # add uncertainty
                if not skip_uncertainties and l2i(path[:-1] + [path[-1] + '_error_upper']) in joined_fetch_paths:
                    stdata = imas_get(ids, path[:-1] + [path[-1] + '_error_upper'], None)
                    if stdata is not None:
                        try:
                            data = uarray(data, stdata)
                        except uncertainties.core.NegativeStdDev as _excp:
                            printe('Error loading uncertainty for %s: %s' % (l2i(path), repr(_excp)))
                # assign data to ODS
                ods[path] = data

        finally:
            # close connection to IMAS database
            printd("ids.close()", topic='imas_code')
            ids.close()

    if paths is None:
        ods.setdefault('dataset_description.data_entry.user', unicode(user))
        ods.setdefault('dataset_description.data_entry.machine', unicode(machine))
        ods.setdefault('dataset_description.data_entry.pulse', int(pulse))
        ods.setdefault('dataset_description.data_entry.run', int(run))
        ods.setdefault('dataset_description.imas_version', unicode(imas_version))

    try:
        ods.consistency_check = True
    except LookupError as _excp:
        printe(repr(_excp))

    return ods


def browse_imas(user=os.environ.get('USER', 'dummy_user'), pretty=True, quiet=False,
                user_imasdbdir=os.sep.join([os.environ['HOME'], 'public', 'imasdb'])):
    '''
    Browse available IMAS data (machine/pulse/run) for given user

    :param user: user (of list of users) to browse. Browses all users if None.

    :param pretty: express size in MB and time in human readeable format

    :param quiet: print database to screen

    :param user_imasdbdir: directory where imasdb is located for current user (typically $HOME/public/imasdb/)

    :return: hierarchical dictionary with database of available IMAS data (machine/pulse/run) for given user
    '''
    # if no users are specified, find all users
    if user is None:
        user = glob.glob(user_imasdbdir.replace('/%s/' % os.environ['USER'], '/*/'))
        user = list(map(lambda x: x.split(os.sep)[-3], user))
    elif isinstance(user, basestring):
        user = [user]

    # build database for each user
    imasdb = {}
    for username in user:
        imasdb[username] = {}
        imasdbdir = user_imasdbdir.replace('/%s/' % os.environ['USER'], '/%s/' % username).strip()

        # find MDS+ datafiles
        files = list(recursive_glob('*datafile', imasdbdir))

        # extract machine/pulse/run from filename of MDS+ datafiles
        for file in files:
            tmp = file.split(os.sep)
            if not re.match('ids_[0-9]{5,}.datafile', tmp[-1]):
                continue
            pulse_run = tmp[-1].split('.')[0].split('_')[1]
            pulse = int(pulse_run[:-4])
            run = int(pulse_run[-4:])
            machine = tmp[-4]

            # size and data
            st = os.stat(file)
            size = st.st_size
            date = st.st_mtime
            if pretty:
                import time
                size = '%d Mb' % (int(size / 1024 / 1024))
                date = time.strftime('%d/%m/%y - %H:%M', time.localtime(date))

            # build database
            if machine not in imasdb[username]:
                imasdb[username][machine] = {}
            imasdb[username][machine][pulse, run] = {'size': size, 'date': date}

    # print if not quiet
    if not quiet:
        pprint(imasdb)

    # return database
    return imasdb


def load_omas_iter_scenario(pulse, run=0, paths=None,
                            imas_version=os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version']),
                            verbose=True):
    """
    Load OMAS data set from ITER IMAS scenario database

    :param pulse: IMAS pulse

    :param run: IMAS run

    :param paths: list of paths to load from IMAS

    :param imas_version: IMAS version

    :param verbose: print loading progress

    :return: OMAS data set
    """
    # set MDSPLUS_TREE_BASE_? environment variables as per
    # imasdb /work/imas/shared/iterdb/3 ; env | grep MDSPLUS_TREE_BASE
    try:
        bkp_imas_environment = {}
        for k in range(10):
            if 'MDSPLUS_TREE_BASE_%d' % k in os.environ:
                bkp_imas_environment['MDSPLUS_TREE_BASE_%d' % k] = os.environ['MDSPLUS_TREE_BASE_%d' % k]
            os.environ['MDSPLUS_TREE_BASE_%d' % k] = '/work/imas/shared/iterdb/3/%d' % k

        # load data from imas
        ods = load_omas_imas(user=None, machine=None, pulse=pulse, run=run, paths=paths, imas_version=imas_version, verbose=verbose)

    finally:
        # restore existing IMAS environment
        for k in range(10):
            del os.environ['MDSPLUS_TREE_BASE_%d' % k]
            os.environ.update(bkp_imas_environment)

    return ods


def filled_paths_in_ids(ids, ds, path=None, paths=None, requested_paths=None, assume_uniform_array_structures=False, skip_ggd=True, skip_ion_state=True):
    """
    Taverse an IDS and list leaf paths (with proper sizing for arrays of structures)

    :param ids: input ids

    :param ds: hierarchical data schema as returned for example by load_structure('equilibrium')[1]

    :param path: current location

    :param paths: list of paths that are filled

    :param requested_paths: list of paths that are requested

    :param assume_uniform_array_structures: assume that the first structure in an array of structures has data in the same nodes locations of the later structures in the array

    :param skip_ggd: do not traverse ggd structures

    :param skip_ion_state: do not traverse ion state

    :return: returns list of paths in an IDS that are filled
    """
    if path is None:
        path = []

    if paths is None:
        paths = []

    if requested_paths is None:
        requested_paths = []

    # leaf
    if not len(ds):
        # append path if it has data
        if imas_empty(ids) is not None:
            paths.append(path)
        return paths

    # keys
    keys = list(ds.keys())
    if keys[0] == ':':
        keys = range(len(ids))
        if len(keys) and assume_uniform_array_structures:
            keys = [0]

    # kid must be part of this list
    if len(requested_paths):
        request_check = [p[0] for p in requested_paths]

    # traverse
    for kid in keys:

        # skip ggd structures
        if skip_ggd and kid in ['ggd', 'grids_ggd']:
            continue

        # skip ion state structures
        if skip_ion_state and kid in ['state'] and 'ion' in path:
            continue

        propagate_path = copy.copy(path)
        propagate_path.append(kid)

        # generate requested_paths one level deeper
        propagate_requested_paths = requested_paths
        if len(requested_paths):
            if kid in request_check or (isinstance(kid, int) and ':' in request_check):
                propagate_requested_paths = [p[1:] for p in requested_paths if len(p) > 1 and (kid == p[0] or p[0] == ':')]
            else:
                continue

        # recursive call
        try:
            if isinstance(kid, basestring):
                subtree_paths = filled_paths_in_ids(getattr(ids, kid), ds[kid], propagate_path, [], propagate_requested_paths, assume_uniform_array_structures, skip_ggd=skip_ggd, skip_ion_state=skip_ion_state)
            else:
                subtree_paths = filled_paths_in_ids(ids[kid], ds[':'], propagate_path, [], propagate_requested_paths, assume_uniform_array_structures, skip_ggd=skip_ggd, skip_ion_state=skip_ion_state)
        except Exception:
            print('Error traversing %s ! Possible IMAS version mismatch!' % l2o(path + [kid]))
            return paths
        paths += subtree_paths

        # assume_uniform_array_structures
        if assume_uniform_array_structures and keys[0] == 0:
            zero_paths = subtree_paths
            for key in range(1, len(ids)):
                subtree_paths = copy.deepcopy(zero_paths)
                for p in subtree_paths:
                    p[len(path)] = key
                paths += subtree_paths

    return paths


def through_omas_imas(ods, method=['function', 'class_method'][1]):
    """
    Test save and load OMAS IMAS

    :param ods: ods

    :return: ods
    """
    user = os.environ['USER']
    machine = 'ITER'
    pulse = 1
    run = 0

    if method == 'function':
        paths = save_omas_imas(ods, user=user, machine=machine, pulse=pulse, run=run, new=True)
        ods1 = load_omas_imas(user=user, machine=machine, pulse=pulse, run=run, paths=paths)
    else:
        paths = ods.save('imas', user=user, machine=machine, pulse=pulse, run=run, new=True)
        ods1 = ODS().load('imas', user=user, machine=machine, pulse=pulse, run=run, paths=paths)
    return ods1
