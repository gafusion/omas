'''save/load from IMAS routines

-------
'''

from .omas_utils import *
from .omas_core import ODS, codeparams_xml_save, codeparams_xml_load, dynamic_ODS, omas_environment
from .omas_utils import _extra_structures


class IDS:
    def __init__(self, DBentry, occurrence):
        self.DBentry = DBentry
        self.occurrence = occurrence

    def __getattr__(self, key):
        import imas

        printd(f"{key} = imas.{key}()", topic='imas_code')
        tmp = getattr(imas, key)()
        setattr(self, key, tmp)
        return tmp

    def close(self):
        self.DBentry.close()


# --------------------------------------------
# IMAS convenience functions
# --------------------------------------------
def imas_open(user, machine, pulse, run, occurrence={}, new=False, imas_major_version='3', backend='MDSPLUS', verbose=True):
    """
    function to open an IMAS

    :param user: IMAS username

    :param machine: IMAS machine

    :param pulse: IMAS pulse

    :param run: IMAS run id

    :param new: whether the open should create a new IMAS tree

    :param imas_major_version: IMAS major version (string)

    :param backend: one of MDSPLUS, ASCII, HDF5, MEMORY, UDA, NO

    :param verbose: print open parameters

    :return: IMAS ids
    """
    if verbose:
        print(
            'Opening {new} IMAS data for user={user} machine={machine} pulse={pulse} run={run}'.format(
                new=['existing', 'new'][int(new)], user=repr(user), machine=repr(machine), pulse=pulse, run=run
            )
        )

    import imas

    printd(
        f"DBentry = imas.DBEntry(imas.imasdef.{backend}_BACKEND, {repr(machine)}, {pulse}, {run}, {repr(user)}, {repr(imas_major_version)})",
        topic='imas_code',
    )
    DBentry = imas.DBEntry(getattr(imas.imasdef, backend + '_BACKEND'), machine, pulse, run, user, imas_major_version)

    if new:
        printd(f"DBentry.create()", topic='imas_code')
        ret_code = DBentry.create()[0]
    else:
        printd(f"DBentry.open()", topic='imas_code')
        ret_code = DBentry.open()[0]

    if ret_code < 0:
        raise IOError(
            'Error opening imas entry (user:%s machine:%s pulse:%s run:%s imas_major_version:%s backend=%s)'
            % (user, machine, pulse, run, imas_major_version, backend)
        )
    return IDS(DBentry, occurrence)


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
    # handle uncertain data
    if is_uncertain(value):
        path = copy.deepcopy(path)
        tmp = imas_set(ids, path, nominal_values(value), skip_missing_nodes=skip_missing_nodes, allocate=allocate)
        path[-1] = path[-1] + '_error_upper'
        imas_set(ids, path, std_devs(value), skip_missing_nodes=skip_missing_nodes, allocate=allocate)
        return tmp

    ds = path[0]
    path = path[1:]

    # identify data dictionary to use, from this point on `m` points to the IDS
    debug_path = ''
    if hasattr(ids, ds):
        debug_path += '%s' % ds
        m = getattr(ids, ds)
        if hasattr(m, 'time') and not isinstance(m.time, float) and not m.time.size:
            m.time.resize(1)
            m.time[0] = -1.0
    elif l2i(path) == 'ids_properties.occurrence':  # IMAS does not store occurrence info as part of the IDSs
        return
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
        location = l2i([ds] + path[: kp + 1])
        if isinstance(p, str):
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
    """
    Check if value is an IMAS empty
        * array with no size
        * float of value -9E40
        * integer of value -999999999
        * empty string

    :param value: value to check

    :return: None if value is an IMAS empty
    """
    # arrays
    if isinstance(value, numpy.ndarray):
        if not value.size:
            return None
        else:
            return value
    # missing floats
    elif isinstance(value, float):
        if value == -9e40:
            return None
        else:
            return value
    # missing integers
    elif isinstance(value, int):
        if value == -999999999:
            return None
        else:
            return value
    # empty strings
    elif isinstance(value, str):
        if not len(value):
            return None
        else:
            return value
    # anything else is not a leaf
    return None


def imas_get(ids, path, skip_missing_nodes=False, check_empty=True):
    """
    read the value of a path in an open IMAS ids

    :param ids: open IMAS ids to read from

    :param path: ODS path

    :param skip_missing_nodes:  if the IMAS path does not exists:
                             `False` raise an error
                             `True` does not raise error
                             `None` prints a warning message

    :param check_empty: return None if not a leaf or empty leaf

    :return: the value that was read if successful or None otherwise
    """
    printd('fetching: %s' % l2i(path), topic='imas')
    ds = path[0]
    path = path[1:]

    debug_path = ''
    if hasattr(ids, ds):
        debug_path += '%s' % ds
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
        if isinstance(p, str):
            if hasattr(out, p):
                debug_path += '.%s' % p
                out = getattr(out, p)
            elif skip_missing_nodes is not False:
                if skip_missing_nodes is None:
                    printe('WARNING: %s is not part of IMAS' % l2i([ds] + path[: kp + 1]))
                    printe(out.__dict__.keys())
                return None
            else:
                printd(debug_path, topic='imas_code')
                raise AttributeError('%s is not part of IMAS' % l2i([ds] + path[: kp + 1]))
        else:
            debug_path += '[%s]' % p
            out = out[p]

    # handle missing data
    if check_empty:
        out = imas_empty(out)

    printd(debug_path, topic='imas_code')
    return out


# --------------------------------------------
# save and load OMAS to IMAS
# --------------------------------------------
@codeparams_xml_save
def save_omas_imas(ods, user=None, machine=None, pulse=None, run=None, occurrence={}, new=False, imas_version=None, verbose=True):
    """
    Save OMAS data to IMAS

    :param ods: OMAS data set

    :param user: IMAS username (reads ods['dataset_description.data_entry.user'] if user is None and finally fallsback on os.environ['USER'])

    :param machine: IMAS machine (reads ods['dataset_description.data_entry.machine'] if machine is None)

    :param pulse: IMAS pulse (reads ods['dataset_description.data_entry.pulse'] if pulse is None)

    :param run: IMAS run (reads ods['dataset_description.data_entry.run'] if run is None and finally fallsback on 0)

    :param occurrence: dictinonary with the occurrence to save for each IDS

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

    printd('Saving to IMAS (user:%s machine:%s pulse:%d run:%d, imas_version:%s)' % (user, machine, pulse, run, imas_version), topic='imas')

    # ensure requirements for writing data to IMAS are satisfied
    ods.satisfy_imas_requirements()

    # get the list of paths from ODS
    paths = set_paths = ods.paths()

    try:
        # open IMAS tree
        ids = imas_open(user=user, machine=machine, pulse=pulse, run=run, occurrence=occurrence, new=new, verbose=verbose)

    except IOError as _excp:
        raise IOError(str(_excp) + '\nIf this is a new pulse/run then set `new=True`')

    except ImportError:
        # fallback on saving IMAS as NC file if IMAS is not installed
        if not omas_rcparams['allow_fake_imas_fallback']:
            raise
        filename = os.sep.join(
            [
                omas_rcparams['fake_imas_dir'],
                '%s_%s_%d_%d_v%s.pkl' % (user, machine, pulse, run, imas_versions.get(imas_version, imas_version)),
            ]
        )
        printe(f'Overloaded save_omas_imas: {filename}')
        from . import save_omas_pkl

        if not os.path.exists(omas_rcparams['fake_imas_dir']):
            os.makedirs(omas_rcparams['fake_imas_dir'])
        ods['dataset_description.data_entry.user'] = str(user)
        ods['dataset_description.data_entry.machine'] = str(machine)
        ods['dataset_description.data_entry.pulse'] = int(pulse)
        ods['dataset_description.data_entry.run'] = int(run)
        ods['dataset_description.imas_version'] = str(imas_version)
        save_omas_pkl(ods, filename)

    else:

        try:
            # allocate memory
            # NOTE: for how memory allocation works it is important to traverse the tree in reverse
            set_paths = []
            for path in reversed(paths):
                set_paths.append(imas_set(ids, path, ods[path], None, allocate=True))
            set_paths = list(filter(None, set_paths))

            # assign the data
            for path in set_paths:
                printd(f'writing {l2i(path)}')
                imas_set(ids, path, ods[path], True)

            # actual write of IDS data to IMAS database
            for ds in ods.keys():
                occ = ids.occurrence.get(ds, ods.get('ids_properties.occurrence', 0))
                printd(f"{ds}.put({occ}, DBentry)", topic='imas_code')
                getattr(ids, ds).put(occ, ids.DBentry)

        finally:
            # close connection to IMAS database
            printd("DBentry.close()", topic='imas_code')
            ids.DBentry.close()

    return set_paths


def infer_fetch_paths(ids, occurrence, paths, time, imas_version, verbose=True):
    """
    Return list of IMAS paths that have data

    :param ids: IMAS ids

    :param occurrence: dictinonary with the occurrence to load for each IDS

    :param paths: list of paths to load from IMAS

    :param imas_version: IMAS version

    :param time: extract a time slice [expressed in seconds] from the IDS

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
        if not hasattr(ids, ds):
            if verbose:
                print(f'| {ds.ljust(ndss)} IDS of IMAS version {imas_version} is unknown')
            continue

        # retrieve this occurrence for this IDS
        occ = occurrence.get(ds, 0)

        # ids.get()
        if time is None:
            printd(f"{ds}.get({occ}, DBentry)", topic='imas_code')
            try:
                getattr(ids, ds).get(occ, ids.DBentry)
            except ValueError as _excp:
                print(f'x {ds.ljust(ndss)} IDS failed on get')  # not sure why some IDSs fail on .get()... it's not about them being empty
                continue

        # ids.getSlice()
        else:
            printd(f"ids.{ds}.getSlice({time}, 1, {occ}, DBentry)", topic='imas_code')
            try:
                getattr(ids, ds).getSlice(time, 1, occ, ids.DBentry)
            except ValueError as _excp:
                print(f'x {ds.ljust(ndss)} IDS failed on getSlice')
                continue

        # see if the IDS has any data (if so homogeneous_time must be populated)
        if getattr(ids, ds).ids_properties.homogeneous_time != -999999999:
            if verbose:
                try:
                    print(f'* {ds.ljust(ndss)} IDS has data ({len(getattr(ids, ds).time)} times)')
                except Exception as _excp:
                    print(f'* {ds.ljust(ndss)} IDS')
                fetch_paths += filled_paths_in_ids(ids, load_structure(ds, imas_version=imas_version)[1], [], [], requested_paths)

        else:
            if verbose:
                print(f'- {ds.ljust(ndss)} IDS is empty')

    joined_fetch_paths = list(map(l2i, fetch_paths))
    return fetch_paths, joined_fetch_paths


@codeparams_xml_load
def load_omas_imas(
    user=os.environ.get('USER', 'dummy_user'),
    machine=None,
    pulse=None,
    run=0,
    occurrence={},
    paths=None,
    time=None,
    imas_version=None,
    skip_uncertainties=False,
    consistency_check=True,
    verbose=True,
):
    """
    Load OMAS data from IMAS

    :param user: IMAS username

    :param machine: IMAS machine

    :param pulse: IMAS pulse

    :param run: IMAS run

    :param occurrence: dictinonary with the occurrence to load for each IDS

    :param paths: list of paths to load from IMAS

    :param time: time slice [expressed in seconds] to be extracted

    :param imas_version: IMAS version (force specific version)

    :param skip_uncertainties: do not load uncertain data

    :param consistency_check: perform consistency_check

    :param verbose: print loading progress

    :return: OMAS data set
    """

    if pulse is None or run is None:
        raise Exception('`pulse` and `run` must be specified')

    printd(
        'Loading from IMAS (user:%s machine:%s pulse:%d run:%d, imas_version:%s)' % (user, machine, pulse, run, imas_version), topic='imas'
    )

    try:
        ids = imas_open(user=user, machine=machine, pulse=pulse, run=run, occurrence=occurrence, new=False, verbose=verbose)

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
        filename = os.sep.join(
            [
                omas_rcparams['fake_imas_dir'],
                '%s_%s_%d_%d_v%s.pkl' % (user, machine, pulse, run, imas_versions.get(imas_version, imas_version)),
            ]
        )
        printe('Overloaded load_omas_imas: %s' % filename)
        from . import load_omas_pkl

        ods = load_omas_pkl(filename, consistency_check=False)

    else:

        try:
            # see what paths have data
            # NOTE: this is where the IDS.get operation occurs
            fetch_paths, joined_fetch_paths = infer_fetch_paths(
                ids, occurrence=occurrence, paths=paths, time=time, imas_version=imas_version, verbose=verbose
            )
            # build omas data structure
            ods = ODS(imas_version=imas_version, consistency_check=False)
            if verbose and tqdm is not None:
                progress_fetch_paths = tqdm.tqdm(fetch_paths, file=sys.stdout)
            else:
                progress_fetch_paths = fetch_paths
            with omas_environment(ods, dynamic_path_creation='dynamic_array_structures'):
                for k, path in enumerate(progress_fetch_paths):
                    # print progress
                    if verbose:
                        if tqdm is not None:
                            progress_fetch_paths.set_description(joined_fetch_paths[k])
                        elif k % int(numpy.ceil(len(fetch_paths) / 10)) == 0 or k == len(fetch_paths) - 1:
                            print('Loading {0:3.1f}%'.format(100 * float(k) / (len(fetch_paths) - 1)))
                    # uncertain data is loaded as part of the nominal value of the data
                    if path[-1].endswith('_error_upper') or path[-1].endswith('_error_lower') or path[-1].endswith('_error_index'):
                        continue
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
                    # NOTE: here we can use setraw since IMAS data is by definition compliant with IMAS
                    ods.setraw(path, data)

        finally:
            # close connection to IMAS database
            printd("ids.close()", topic='imas_code')
            ids.close()

    # add dataset_description information to this ODS
    if paths is None:
        ods.setdefault('dataset_description.data_entry.user', str(user))
        ods.setdefault('dataset_description.data_entry.machine', str(machine))
        ods.setdefault('dataset_description.data_entry.pulse', int(pulse))
        ods.setdefault('dataset_description.data_entry.run', int(run))
        ods.setdefault('dataset_description.imas_version', str(imas_version))

    # add occurrence information to the ODS
    for ds in ods:
        if 'ids_properties' in ods[ds]:
            ods[ds]['ids_properties.occurrence'] = occurrence.get(ds, 0)

    try:
        ods.consistency_check = consistency_check
    except LookupError as _excp:
        printe(repr(_excp))

    return ods


class dynamic_omas_imas(dynamic_ODS):
    """
    Class that provides dynamic data loading from IMAS
    This class is not to be used by itself, but via the ODS.open() method.
    """

    def __init__(self, user=os.environ.get('USER', 'dummy_user'), machine=None, pulse=None, run=0, occurrence={}, verbose=True):
        self.kw = {'user': user, 'machine': machine, 'pulse': pulse, 'run': run, 'verbose': verbose, 'occurrence': occurrence}
        self.ids = None
        self.active = False
        self.open_ids = []

    def open(self):
        printd('Dynamic open  %s' % self.kw, topic='dynamic')
        self.ids = imas_open(new=False, **self.kw)
        self.active = True
        self.open_ids = []
        return self

    def close(self):
        printd('Dynamic close %s' % self.kw, topic='dynamic')
        self.ids.close()
        self.open_ids = []
        self.ids = None
        self.active = False
        return self

    def __getitem__(self, key):
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        printd('Dynamic read  %s: %s' % (self.kw, key), topic='dynamic')
        return imas_get(self.ids, p2l(key))

    def __contains__(self, key):
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        path = p2l(key)
        ds = path[0]
        if ds not in self.open_ids:
            occ = self.ids.occurrence.get(ds, 0)
            getattr(self.ids, ds).get(occ, self.ids.DBentry)
            self.open_ids.append(ds)
        return imas_empty(imas_get(self.ids, path)) is not None

    def keys(self, location):
        return keys_leading_to_a_filled_path(self.ids, location, os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version']))


def browse_imas(
    user=os.environ.get('USER', 'dummy_user'),
    pretty=True,
    quiet=False,
    user_imasdbdir=os.sep.join([os.environ['HOME'], 'public', 'imasdb']),
):
    """
    Browse available IMAS data (machine/pulse/run) for given user

    :param user: user (of list of users) to browse. Browses all users if None.

    :param pretty: express size in MB and time in human readeable format

    :param quiet: print database to screen

    :param user_imasdbdir: directory where imasdb is located for current user (typically $HOME/public/imasdb/)

    :return: hierarchical dictionary with database of available IMAS data (machine/pulse/run) for given user
    """
    # if no users are specified, find all users
    if user is None:
        user = glob.glob(user_imasdbdir.replace('/%s/' % os.environ['USER'], '/*/'))
        user = list(map(lambda x: x.split(os.sep)[-3], user))
    elif isinstance(user, str):
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


def load_omas_iter_scenario(
    pulse, run=0, paths=None, imas_version=os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version']), verbose=True
):
    """
    Load ODS from ITER IMAS scenario database

    :param pulse: IMAS pulse

    :param run: IMAS run

    :param paths: list of paths to load from IMAS

    :param imas_version: IMAS version

    :param verbose: print loading progress

    :return: OMAS data set
    """
    return load_omas_imas(user='public', machine='iterdb', pulse=pulse, run=run, paths=paths, imas_version=imas_version, verbose=verbose)


def filled_paths_in_ids(
    ids, ds, path=None, paths=None, requested_paths=None, assume_uniform_array_structures=False, stop_on_first_fill=False
):
    """
    Taverse an IDS and list leaf paths (with proper sizing for arrays of structures)

    :param ids: input ids

    :param ds: hierarchical data schema as returned for example by load_structure('equilibrium')[1]

    :param path: current location

    :param paths: list of paths that are filled

    :param requested_paths: list of paths that are requested

    :param assume_uniform_array_structures: assume that the first structure in an array of structures has data in the same nodes locations of the later structures in the array

    :param stop_on_first_fill: return as soon as one path with data hass been found

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
        if kid == 'occurrence' and path[-1] == 'ids_properties':
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
            if isinstance(kid, str):
                subtree_paths = filled_paths_in_ids(
                    getattr(ids, kid), ds[kid], propagate_path, [], propagate_requested_paths, assume_uniform_array_structures
                )
            else:
                subtree_paths = filled_paths_in_ids(
                    ids[kid], ds[':'], propagate_path, [], propagate_requested_paths, assume_uniform_array_structures
                )
        except Exception:
            # check if the issue was that we were trying to load something that was added to the _extra_structures
            if o2i(l2u(propagate_path)) in _extra_structures.get(propagate_path[0], {}):
                # printe('`%s` does not exist in the IMAS data dictionary.
                # Consider opening a JIRA issue asking for its addition: https://jira.iter.org' % l2i(path + [kid]))
                continue
            printe('Error querying IMAS database for `%s` Possible IMAS version mismatch?' % l2i(path + [kid]))
            continue
        paths += subtree_paths

        # assume_uniform_array_structures
        if assume_uniform_array_structures and keys[0] == 0:
            zero_paths = subtree_paths
            for key in range(1, len(ids)):
                subtree_paths = copy.deepcopy(zero_paths)
                for p in subtree_paths:
                    p[len(path)] = key
                paths += subtree_paths

        # if stop_on_first_fill return as soon as a filled path has been found
        if len(paths) and stop_on_first_fill:
            return paths

    return paths


def reach_ids_location(ids, path):
    """
    Traverse IMAS structure until reaching location

    :param ids: IMAS ids

    :param path: path to reach

    :return: requested location in IMAS ids
    """
    out = ids
    for p in path:
        if isinstance(p, str):
            out = getattr(out, p)
        else:
            out = out[p]
    return out


def reach_ds_location(path, imas_version):
    """
    Traverse ds structure until reaching location

    :param path: path to reach

    :param imas_version: IMAS version

    :return: requested location in ds
    """
    ds = load_structure(path[0], imas_version=imas_version)[1]
    out = ds
    for kp, p in enumerate(path):
        if not isinstance(p, str):
            p = ':'
        out = out[p]
    return out


def keys_leading_to_a_filled_path(ids, location, imas_version):
    """
    What keys at a given IMAS location lead to a leaf that has data

    :param ids: IMAS ids

    :param location: location to query

    :param imas_version:  IMAS version

    :return: list of keys
    """
    # if no location is passed, then we see if the IDSs are filled at all
    if not len(location):
        filled_keys = []
        for ds in list_structures(imas_version=imas_version):
            if not hasattr(ids, ds):
                continue
            occ = ids.occurrence.get(ds, 0)
            getattr(ids, ds).get(occ, ids.DBentry)
            if getattr(ids, ds).ids_properties.homogeneous_time != -999999999:
                filled_keys.append(ds)
        return filled_keys

    path = p2l(location)
    ids = reach_ids_location(ids, path)
    ds = reach_ds_location(path, imas_version)

    # always list all arrays of structures
    if list(ds.keys())[0] == ':':
        return list(range(len(ids)))

    # find which keys have at least one filled path underneath
    filled_keys = []
    for kid in ds.keys():
        paths = filled_paths_in_ids(getattr(ids, kid), ds[kid], stop_on_first_fill=True)
        if len(paths):
            filled_keys.append(kid)

    return filled_keys


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
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
    if method == 'function':
        paths = save_omas_imas(ods, user=user, machine=machine, pulse=pulse, run=run, new=True)
        ods1 = load_omas_imas(user=user, machine=machine, pulse=pulse, run=run, paths=paths)
    else:
        paths = ods.save('imas', user=user, machine=machine, pulse=pulse, run=run, new=True)
        ods1 = ODS().load('imas', user=user, machine=machine, pulse=pulse, run=run, paths=paths)
    return ods1


# List IDS fields that have to be present to add datasets in the ITER scenario database
# as defined here: from https://confluence.iter.org/x/kQqOE
iter_scenario_requirements = [
    'equilibrium.time_slice.:.global_quantities.ip',
    'equilibrium.time_slice.:.profiles_2d.:.phi',
    'equilibrium.time_slice.:.profiles_2d.:.psi',
    'equilibrium.time_slice.:.profiles_2d.:.r',
    'equilibrium.time_slice.:.profiles_2d.:.z',
    'equilibrium.vacuum_toroidal_field.r0',
    'equilibrium.vacuum_toroidal_field.b0',
    'core_profiles.global_quantities.beta_pol',
    'core_profiles.global_quantities.beta_tor_norm',
    'core_profiles.global_quantities.current_bootstrap',
    'core_profiles.global_quantities.current_non_inductive',
    'core_profiles.global_quantities.energy_diamagnetic',
    'core_profiles.global_quantities.ip',
    'core_profiles.global_quantities.v_loop',
    'core_profiles.profiles_1d.:.e_field.radial',
    'core_profiles.profiles_1d.:.electrons.density',
    'core_profiles.profiles_1d.:.electrons.pressure',
    'core_profiles.profiles_1d.:.electrons.pressure_fast_parallel',
    'core_profiles.profiles_1d.:.electrons.pressure_fast_perpendicular',
    'core_profiles.profiles_1d.:.electrons.pressure_thermal',
    'core_profiles.profiles_1d.:.electrons.temperature',
    'core_profiles.profiles_1d.:.grid.rho_tor',
    'core_profiles.profiles_1d.:.grid.rho_tor_norm',
    'core_profiles.profiles_1d.:.grid.psi',
    'core_profiles.profiles_1d.:.grid.volume',
    'core_profiles.profiles_1d.:.ion.:.density',
    'core_profiles.profiles_1d.:.ion.:.element.:.a',
    'core_profiles.profiles_1d.:.ion.:.element.:.z_n',
    'core_profiles.profiles_1d.:.ion.:.pressure',
    'core_profiles.profiles_1d.:.ion.:.pressure_fast_parallel',
    'core_profiles.profiles_1d.:.ion.:.pressure_fast_perpendicular',
    'core_profiles.profiles_1d.:.ion.:.pressure_thermal',
    'core_profiles.profiles_1d.:.ion.:.temperature',
    'core_profiles.profiles_1d.:.ion.:.velocity.diamagnetic',
    'core_profiles.profiles_1d.:.ion.:.velocity.poloidal',
    'core_profiles.profiles_1d.:.ion.:.velocity.toroidal',
    'core_profiles.profiles_1d.:.j_bootstrap',
    'core_profiles.profiles_1d.:.j_non_inductive',
    'core_profiles.profiles_1d.:.j_ohmic',
    'core_profiles.profiles_1d.:.j_total',
    'core_profiles.profiles_1d.:.magnetic_shear',
    'core_profiles.profiles_1d.:.pressure_ion_total',
    'core_profiles.profiles_1d.:.pressure_parallel',
    'core_profiles.profiles_1d.:.pressure_perpendicular',
    'core_profiles.profiles_1d.:.pressure_thermal',
    'core_profiles.profiles_1d.:.q',
    'core_profiles.profiles_1d.:.t_i_average',
    'core_profiles.profiles_1d.:.zeff',
    'summary.global_quantities.b0.value',
    'summary.global_quantities.r0.value',
    'summary.global_quantities.beta_pol.value',
    'summary.global_quantities.beta_tor_norm.value',
    'summary.global_quantities.current_bootstrap.value',
    'summary.global_quantities.current_non_inductive.value',
    'summary.global_quantities.current_ohm.value',
    'summary.global_quantities.energy_diamagnetic.value',
    'summary.global_quantities.energy_thermal.value',
    'summary.global_quantities.energy_total.value',
    'summary.global_quantities.h_98.value',
    'summary.global_quantities.h_mode.value',
    'summary.global_quantities.ip.value',
    'summary.global_quantities.tau_energy.value',
    'summary.global_quantities.v_loop.value',
    'summary.local.separatrix.n_e.value',
    'summary.local.separatrix.zeff.value',
]
