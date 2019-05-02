'''naming convention translation and misc utilities

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_setup import *
import sys


# --------------------------------------------
# ODS utilities
# --------------------------------------------
def different_ods(ods1, ods2):
    """
    Checks if two ODSs have any difference and returns the string with the cause of the different

    :param ods1: first ods to check

    :param ods2: second ods to check

    :return: string with reason for difference, or False otherwise
    """
    ods1 = ods1.flat()
    ods2 = ods2.flat()

    k1 = set(ods1.keys())
    k2 = set(ods2.keys())
    differences = []
    for k in k1.difference(k2):
        if not k.startswith('info.'):
            differences.append('DIFF: key `%s` missing in 2nd ods' % k)
    for k in k2.difference(k1):
        if not k.startswith('info.'):
            differences.append('DIFF: key `%s` missing in 1st ods' % k)
    for k in k1.intersection(k2):
        if isinstance(ods1[k], basestring) and isinstance(ods2[k], basestring):
            if ods1[k] != ods2[k]:
                differences.append('DIFF: `%s` differ in value' % k)
        elif type(ods1[k]) != type(ods2[k]):
            differences.append('DIFF: `%s` differ in type (%s,%s)' % (k, type(ods1[k]), type(ods2[k])))
        elif numpy.atleast_1d(is_uncertain(ods1[k])).any() or numpy.atleast_1d(is_uncertain(ods2[k])).any():
            if not numpy.allclose(nominal_values(ods1[k]), nominal_values(ods2[k]), equal_nan=True) or not numpy.allclose(std_devs(ods1[k]), std_devs(ods2[k]), equal_nan=True):
                differences.append('DIFF: `%s` differ in value' % k)
        else:
            if not numpy.allclose(ods1[k], ods2[k], equal_nan=True):
                differences.append('DIFF: `%s` differ in value' % k)
    if len(differences):
        return differences
    else:
        return False


# --------------------------
# general utility functions
# --------------------------
def printd(*objects, **kw):
    """
    debug print
    environmental variable OMAS_DEBUG_TOPIC sets the topic to be printed
    """
    topic = kw.pop('topic', '')
    if isinstance(topic, basestring):
        topic = [topic]
    topic = list(map(lambda x: x.lower(), topic))
    objects = ['DEBUG:'] + list(objects)
    topic_selected = os.environ.get('OMAS_DEBUG_TOPIC', '')
    dump = False
    if topic_selected.endswith('_dump'):
        dump = True
        topic_selected = re.sub('_dump$', '', topic_selected)
    if topic_selected and (topic_selected == '*' or topic_selected in topic or '*' in topic):
        if eval(os.environ.get('OMAS_DEBUG_STDOUT', '1')):
            print(*objects, **kw)
        else:
            printe(*objects, **kw)
        if dump:
            fb = StringIO()
            print(*objects[1:], file=fb)
            with open('omas_dump.txt', 'a') as f:
                f.write(fb.getvalue())
            fb.close()


def printe(*objects, **kw):
    """
    print to stderr
    """
    kw['file'] = sys.stderr
    print(*objects, **kw)


# printw works like printe (this is done to allow mindless copy of some OMFIT functions in OMAS)
printw = printe


def is_uncertain(var):
    '''
    :param var: Variable or array to test

    :return: True if variable is instance of uncertainties or
             array of shape var with elements indicating uncertainty
    '''

    def _uncertain_check(x):
        return isinstance(x, uncertainties.core.AffineScalarFunc)

    if isinstance(var, basestring):
        return False
    elif numpy.iterable(var):
        tmp = numpy.array(var).flat
        tmp = numpy.array(list(map(_uncertain_check, tmp)))
        return numpy.reshape(tmp, numpy.array(var).shape)
    else:
        return _uncertain_check(var)


def is_numeric(value):
    """
    Convenience function check if value is numeric

    :param value: value to check

    :return: True/False
    """
    try:
        0 + value
        return True
    except TypeError:
        return False


def omas_interp1d(x, xp, fp, left=None, right=None, period=None):
    '''
    If xp is not increasing, the results are numpy.interp1d nonsense.
    This function wraps numpy.interp1d but makes sure that the x-coordinate sequence xp is increasing.

    '''
    if not numpy.all(numpy.diff(xp) > 0):
        index = numpy.argsort(xp)
        return numpy.interp(x, xp[index], fp[index], left=left, right=right, period=period)
    return numpy.interp(x, xp, fp, left=left, right=right, period=period)


omas_interp1d.__doc__ += numpy.interp.__doc__


def json_dumper(obj, objects_encode=True):
    """
    Dump objects to json format

    :param obj: input ojbect

    :param objects_encode: how to handle non-standard JSON objects
        * True: encode numpy arrays, complex, and uncertain
        * None: numpy arrays as lists, encode complex, and uncertain
        * False: numpy arrays as lists, fail on complex, and uncertain

    :return: json-compatible object
    """
    from omas import ODS

    if isinstance(obj, ODS):
        return obj.omas_data

    if not objects_encode:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.generic):
            return obj.item()
        else:
            return obj.toJSON()

    else:
        tmp = is_uncertain(obj)
        if numpy.any(numpy.atleast_1d(tmp)):
            if not len(numpy.array(tmp).shape):
                return dict(__ufloat__=nominal_values(obj),
                            __ufloat_std__=std_devs(obj))
            else:
                nomv = nominal_values(obj)
                return dict(__udarray_tolist_avg__=nomv.tolist(),
                            __udarray_tolist_std__=std_devs(obj).tolist(),
                            dtype=str(nomv.dtype),
                            shape=obj.shape)
        elif isinstance(obj, numpy.ndarray):
            if 'complex' in str(obj.dtype).lower():
                return dict(__ndarray_tolist_real__=obj.real.tolist(),
                            __ndarray_tolist_imag__=obj.imag.tolist(),
                            dtype=str(obj.dtype),
                            shape=obj.shape)
            else:
                if objects_encode is None:
                    return obj.tolist()
                else:
                    return dict(__ndarray_tolist__=obj.tolist(),
                                dtype=str(obj.dtype),
                                shape=obj.shape)
        elif isinstance(obj, numpy.generic):
            return obj.item()
        elif isinstance(obj, complex):
            return dict(__complex__=True, real=obj.real, imag=obj.imag)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        else:
            return obj.toJSON()


def json_loader(object_pairs, cls=dict):
    """
    Load json-objects generated by the json_dumper function

    :param object_pairs: json-compatible [dict/list] object

    :param cls: dicitonary class to use

    :return: ojbect
    """
    from omas import ODS
    def convert_int(key):
        try:
            return int(key)
        except ValueError:
            return key

    object_pairs = list(map(lambda o: (convert_int(o[0]), o[1]), object_pairs))

    dct = cls()
    for x, y in object_pairs:
        if isinstance(y, list):
            if len(y) and isinstance(y[0], (dict, ODS)):
                dct[x] = cls()
                for k in range(len(y)):
                    dct[x][k] = y[k]
            else:
                dct[x] = y
        else:
            dct[x] = y

    if '__ndarray_tolist__' in dct:
        return numpy.array(dct['__ndarray_tolist__'], dtype=dct['dtype']).reshape(dct['shape'])
    elif '__ndarray_tolist_real__' in dct and '__ndarray_tolist_imag__' in dct:
        return (numpy.array(dct['__ndarray_tolist_real__'], dtype=dct['dtype']).reshape(dct['shape']) +
                numpy.array(dct['__ndarray_tolist_imag__'], dtype=dct['dtype']).reshape(dct['shape']) * 1j)
    elif '__udarray_tolist_avg__' in dct and '__udarray_tolist_std__' in dct:
        return uarray(numpy.array(dct['__udarray_tolist_avg__'], dtype=dct['dtype']).reshape(dct['shape']),
                      numpy.array(dct['__udarray_tolist_std__'], dtype=dct['dtype']).reshape(dct['shape']))
    elif '__ufloat__' in dct and '__ufloat_std__' in dct:
        return ufloat(dct['__ufloat__'], dct['__ufloat_std__'])
    elif '__ndarray__' in dct:
        import base64
        data = base64.b64decode(dct['__ndarray__'])
        return numpy.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    elif '__complex__' in dct:
        return complex(dct['real'], dct['imag'])
    return dct


def recursive_glob(pattern='*', rootdir='.'):
    """
    Search recursively for files matching a specified pattern within a rootdir

    :param pattern: glob pattern to match

    :param rootdir: top level directory to search under
    """
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def remove_parentheses(inv, replace_with=''):
    '''
    function used to remove/replace top-level matching parenthesis from a string

    :param inv: input string

    :param replace_with: string to replace matching parenthesis with

    :return: input string without first set of matching parentheses
    '''
    k = 0
    lp = ''
    out = ''
    for c in inv:
        # go one level deep
        if c == '(':
            k += 1
            lp = c
        # go one level up
        elif c == ')':
            k -= 1
            lp += c
            if k == 0:
                out += replace_with
        # zero depth: add character to output string
        elif k == 0:
            out += c
    return out


def closest_index(my_list, my_number=0):
    """
    Given a SORTED iterable (a numeric array or list of numbers) and a numeric scalar my_number, find the index of the
    number in the list that is closest to my_number

    :param my_list: Sorted iterable (list or array) to search for number closest to my_number

    :param my_number: Number to get close to in my_list

    :return: Index of my_list element closest to my_number

    :note: If two numbers are equally close, returns the index of the smallest number.
    """
    import bisect

    if not hasattr(my_list, '__iter__'):
        raise TypeError("closestIndex() in utils_math.py requires an iterable as the first argument. Got "
                        "instead: {:}".format(my_list))

    if not is_numeric(my_number):
        if hasattr(my_number, '__iter__') and len(my_number) == 1 and is_numeric(my_number[0]):
            printw('Warning: closestIndex got a len()=1 iterable instead of a scalar for my_number. my_number[0] will '
                   'be used, but please input a scalar next time.')
            # Before, the function would run without an error if given a one element array, but it would not return the
            # correct answer.
            my_number = my_number[0]
            printd('my_number is now {:}'.format(my_number))
        else:
            raise TypeError("closestIndex() in utils_math.py requires a numeric scalar as the second argument. Got "
                            "instead: {:}".format(my_number))

    pos = bisect.bisect_left(my_list, my_number)
    if pos == 0:
        return 0
    if pos == len(my_list):
        return pos - 1
    before = pos - 1
    after = pos
    if my_list[after] - my_number < my_number - my_list[before]:
        return pos
    else:
        return pos - 1


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


# ----------------------------------------------
# handling of OMAS json structures
# ----------------------------------------------
# IMAS structure info organized as flat entries
# * IMAS syntax with `:` for list of structures
# * each entry contains leafs attributes
_structures = {}
# IMAS structure info organized in hierarchical dictionaries
# * list of structures as `:`
# * the leafs are empty dictionaries
_structures_dict = {}
# similar to `_structures_dict` but for use in omas_info
_info_structures = {}
# dictionary that contains all the coordinates defined within the data dictionary
_coordinates = {}

# extra structures that python modules using omas can define
# by setting omas.omas_utils._extra_structures equal to a
# dictionary with the definitions of the quantities that are
# not (yet) available in IMAS. For example:
#
# omas.omas_utils._extra_structures = {
#     'equilibrium': {
#         'equilibrium.time_slice.:.profiles_1d.centroid.r_max': {
#             "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.r_max(:)",
#             "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
#             "data_type": "FLT_1D",
#             "description": "centroid r max",
#             "units": 'm',
#             "cocos_signal": '?'  # optional
#         }
#     }
# }
_extra_structures = {}


def list_structures(imas_version):
    '''
    list names of structures in imas version

    :param imas_version: imas version

    :return: list with names of structures in imas version
    '''
    json_filenames = glob.glob(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '*' + '.json')
    json_filenames = filter(lambda x: os.path.basename(x)[0] != '_', json_filenames)
    structures = sorted(list(map(lambda x: os.path.splitext(os.path.split(x)[1])[0], json_filenames)))
    if not len(structures):
        raise ValueError("Unrecognized IMAS version `%s`. Possible options are:\n%s" % (imas_version, imas_versions.keys()))
    return structures


def dict_structures(imas_version):
    '''
    maps structure names to json filenames

    :param imas_version: imas version

    :return: dictionary maps structure names to json filenames
    '''
    paths = glob.glob(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '*' + '.json')
    if not len(paths):
        raise ValueError("Unrecognized IMAS version `%s`. Possible options are:\n%s" % (imas_version, imas_versions.keys()))
    structures = dict(zip(list(map(lambda x: os.path.splitext(os.path.split(x)[1])[0], paths)), paths))
    return {structure: structures[structure] for structure in structures if not structure.startswith('_')}


def load_structure(filename, imas_version):
    """
    load omas structure from given json filename or IDS name

    :param filename: full path to json file or IDS name

    :param imas_version: imas version to load the data schema of (optional if filename is a full path)

    :return: tuple with structure, hashing mapper, and ods
    """

    from .omas_physics import cocos_signals

    filename0 = filename
    id = (filename0, imas_version)
    if id in _structures and id in _structures_dict:
        return _structures[id], _structures_dict[id]

    if os.sep not in filename:
        filename = dict_structures(imas_version)[filename]

    if filename not in _structures:
        with open(filename, 'r') as f:
            dump_string = f.read()
        # load flat definitions from json file
        _structures[id] = json.loads(dump_string)

        # add _extra_structures definitions
        structure_name = os.path.splitext(os.path.split(filename)[1])[0]
        if structure_name in _extra_structures:
            for item in _extra_structures[structure_name]:
                if item not in _structures[id]:
                    cs = _extra_structures[structure_name][item].pop('cocos_signal', None)
                    _structures[id][item] = _extra_structures[structure_name][item]
                    if cs is not None:
                        cocos_signals[i2o(item)] = cs

        # generate hierarchical structure
        _structures_dict[id] = {}
        for item in _structures[id]:
            h = _structures_dict[id]
            for step in i2o(item).split('.'):
                if step not in h:
                    h[step] = {}
                h = h[step]

    return _structures[id], _structures_dict[id]


def omas_coordinates(imas_version=omas_rcparams['default_imas_version']):
    '''
    return list of coordinates

    :param imas_version: IMAS version to look up

    :return: list of strings with IMAS coordinates
    '''
    # caching
    if imas_version not in _coordinates:
        filename = imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '_coordinates.json'
        with open(filename, 'r') as f:
            _coordinates[imas_version] = json.load(f)
    return _coordinates[imas_version]


_p2l_cache = {}


def p2l(key):
    """
    Converts the many different ways of addressing an ODS path to a list of keys (['bla',0,'bla'])

    :param key: ods location in some format

    :return: list of keys that make the ods path
    """
    if isinstance(key, list):
        return key

    if isinstance(key, tuple):
        return list(key)

    if isinstance(key, (int, numpy.integer)):
        return [int(key)]

    if isinstance(key, basestring) and '.' not in key:
        if len(key):
            return [key]
        else:
            return []

    if key is None:
        raise TypeError('OMAS key cannot be None')

    if isinstance(key, dict):
        raise TypeError('OMAS key cannot be of type dictionary')

    key0 = ''.join(key)
    if key0 in _p2l_cache:
        return copy.deepcopy(_p2l_cache[key0])

    if not isinstance(key, (list, tuple)):
        key = str(key).replace('[', '.').replace(']', '').split('.')

    key = [k for k in key if k]
    for k, item in enumerate(key):
        try:
            key[k] = int(item)
        except ValueError:
            pass

    if len(_p2l_cache) > 1000:
        _p2l_cache.clear()
    _p2l_cache[key0] = copy.deepcopy(key)

    return key


def l2i(path):
    """
    Formats a list (['bla',0,'bla']) into a IMAS path ('bla[0].bla')

    :param path: ODS path format

    :return: IMAS path format
    """
    ipath = path[0]
    for step in path[1:]:
        if isinstance(step, int) or step == ':':
            ipath += "[%s]" % step
        else:
            ipath += '.%s' % step
    return ipath


def l2u(path):
    """
    Formats a list (['bla',0,'bla']) into a universal ODS path format ('bla.:.bla')
    NOTE: a universal ODS path substitutes lists indices with :

    :param path: list of strings and integers

    :return: universal ODS path format
    """
    return o2u(l2o(path))


def l2ut(path):
    """
    Formats IMAS time lists (['bla',0,'time_slice',5,'quantity']) with universal ODS path ('bla.0.time_slice.:.quantity')

    :param path: list of strings and integers

    :return: ODS path format with time lists in universal format
    """
    lpath = p2l(path)
    opath = l2o(lpath)
    upath = o2u(path)
    for k, key in enumerate(lpath):
        if not isinstance(key, int):
            continue
        key = lpath[:k]
        info = omas_info_node(l2u(key))
        if 'coordinates' in info:
            for infoc in info['coordinates']:
                if infoc.endswith('.time'):
                    lpath[k] = ':'
    return l2o(lpath)


def l2o(path):
    """
    Formats a list (['bla',0,'bla']) into an ODS path format ('bla.0.bla')

    :param path: list of strings and integers

    :return: ODS path format
    """
    return '.'.join(filter(None, map(str, path)))


_o2u_pattern = re.compile('\.[0-9:]+')
_o2u_pattern_no_split = re.compile('^[0-9:]+')
_i2o_pattern = re.compile('\[([:0-9]+)\]')
_o2i_pattern = re.compile('\.([:0-9]+)')


def o2u(path):
    '''
    Converts an ODS path format ('bla.0.bla') into a universal path format ('bla.:.bla')

    :param path: ODS path format

    :return: universal ODS path format
    '''
    path = str(path)
    if '.' in path:
        return re.sub(_o2u_pattern, '.:', path)
    else:
        return re.sub(_o2u_pattern_no_split, ':', path)


def i2o(path):
    """
    Formats a IMAS path ('bla[0].bla') format into an ODS path ('bla.0.bla')

    :param path: IMAS path format

    :return: ODS path format
    """
    return re.sub(_i2o_pattern, r'.\1', path)


def o2i(path):
    """
    Formats a ODS path ('bla.0.bla') format into an IMAS path ('bla[0].bla')

    :param path: ODS path format

    :return: IMAS path format
    """
    return re.sub(_o2i_pattern, r'[\1]', path)


def u2o(upath, path):
    '''
    Replaces `:` and integers in `upath` with ':' and integers from in `path`
    e.g. uo2('a.:.b.:.c.1.d.1.e','f.:.g.1.h.1.i.:.k')) becomes ('bla.1.hello.2.bla')

    :param upath: universal ODS path

    :param path: ODS path

    :return: filled in ODS path
    '''
    if upath.startswith('1...'):
        return upath
    ul = p2l(upath)
    ol = p2l(path)
    for k in range(min([len(ul), len(ol)])):
        if (ul[k] == ':' or isinstance(ul[k], int)) and (ol[k] == ':' or isinstance(ol[k], int)):
            ul[k] = ol[k]
        elif ul[k] == ol[k]:
            continue
        else:
            break
    return l2o(ul)


def trim_common_path(p1, p2):
    '''
    return paths in lists format trimmed of the common first path between paths p1 and p2

    :param p1: ODS path

    :param p2: ODS path

    :return: paths in list format trimmed of common part
    '''
    p1 = p2l(p1)
    p2 = p2l(p2)
    both = [x if x[0] == x[1] else None for x in zip(p1, p2)] + [None]
    return p1[both.index(None):], p2[both.index(None):]


def ids_cpo_mapper(ids, cpo=None):
    '''
    translate (some) IDS fields to CPO

    :param ids: input omas data object (IDS format) to read

    :param cpo: optional omas data object (CPO format) to which to write to

    :return: cpo
    '''
    from omas import ODS
    if cpo is None:
        cpo = ODS()
    cpo.consistency_check = False

    for itime in range(len(ids['core_profiles.time'])):

        if 'equilibrium' in ids:
            cpo['equilibrium'][itime]['time'] = ids['equilibrium.time'][itime]
            cpo['equilibrium'][itime]['profiles_1d.q'] = ids['equilibrium.time_slice'][itime]['profiles_1d.q']
            cpo['equilibrium'][itime]['profiles_1d.rho_tor'] = ids['equilibrium.time_slice'][itime]['profiles_1d.rho_tor']
            for iprof in range(len(ids['equilibrium.time_slice'][itime]['profiles_2d'])):
                cpo['equilibrium'][itime]['profiles_2d'][iprof]['psi'] = ids['equilibrium.time_slice'][itime]['profiles_2d'][iprof]['psi']

        if 'core_profiles' in ids:
            cpo['coreprof'][itime]['te.value'] = ids['core_profiles.profiles_1d'][itime]['electrons.temperature']
            cpo['coreprof'][itime]['ne.value'] = ids['core_profiles.profiles_1d'][itime]['electrons.density']
            pdim = len(cpo['coreprof'][itime]['te.value'])
            idim = len(ids['core_profiles.profiles_1d[0].ion'])
            cpo['coreprof'][itime]['ni.value'] = numpy.zeros((pdim, idim))
            cpo['coreprof'][itime]['ti.value'] = numpy.zeros((pdim, idim))
            for iion in range(len(ids['core_profiles.profiles_1d'][itime]['ion'])):
                if 'density' in ids['core_profiles.profiles_1d'][itime]['ion'][iion]:
                    cpo['coreprof'][itime]['ni.value'][:, iion] = ids['core_profiles.profiles_1d'][itime]['ion'][iion]['density']
                cpo['coreprof'][itime]['ti.value'][:, iion] = nominal_values(ids['core_profiles.profiles_1d'][itime]['ion'][iion]['temperature'])

    return cpo


def omas_info(structures, imas_version=omas_rcparams['default_imas_version']):
    '''
    This function returns an ods with the leaf nodes filled with their property informations

    :param structures: list with ids names or string with ids name of which to retrieve the info
                       if not structures, then all structures are returned (slow and big)

    :return: ods
    '''

    if not structures:
        structures = sorted(list(dict_structures(imas_version).keys()))
    elif isinstance(structures, basestring):
        structures = [structures]

    # caching
    if imas_version not in _info_structures:
        from omas import ODS
        _info_structures[imas_version] = ODS(imas_version=imas_version, consistency_check=False)
    ods = _info_structures[imas_version]
    ods.consistency_check = False

    for structure in structures:
        if structure not in ods:
            tmp = load_structure(structure, imas_version)[0]
            lst = sorted(tmp.keys())
            for k, item in enumerate(lst):
                if re.match('.*_error_(index|lower|upper)$', item.split('.')[-1]):
                    continue
                parent = False
                for item1 in lst[k + 1:]:
                    if l2u(item1.split('.')[:-1]).rstrip('[:]') == item:
                        parent = True
                        break
                if parent:
                    continue
                ods[item.replace(':', '0')] = tmp[item]

    return copy.deepcopy(ods)


def omas_info_node(key, imas_version=omas_rcparams['default_imas_version']):
    '''
    return information about a given node

    :param key: IMAS path

    :param imas_version: IMAS version to look up

    :return: dictionary with IMAS information (or an empty dictionary if the node is not found)
    '''
    tmp = {}
    try:
        tmp.update(load_structure(key.split('.')[0], imas_version)[0][o2i(key)])
    except KeyError:
        pass
    return tmp
