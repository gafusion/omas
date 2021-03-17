'''naming convention translation and misc utilities

-------
'''

from .omas_setup import *
from .omas_setup import __version__
import sys

# --------------------------------------------
# ODS utilities
# --------------------------------------------
default_keys_to_ignore = [
    'dataset_description.data_entry.user',
    'dataset_description.data_entry.run',
    'dataset_description.data_entry.machine',
    'dataset_description.ids_properties',
    'dataset_description.imas_version',
    'dataset_description.time',
    'ids_properties.homogeneous_time',
    'ids_properties.occurrence',
    'ids_properties.version_put.data_dictionary',
    'ids_properties.version_put.access_layer',
    'ids_properties.version_put.access_layer_language',
]


def different_ods(ods1, ods2, ignore_type=False, ignore_empty=False, ignore_keys=[], ignore_default_keys=True):
    """
    Checks if two ODSs have any difference and returns the string with the cause of the different

    :param ods1: first ods to check

    :param ods2: second ods to check

    :param ignore_type: ignore object type differences

    :param ignore_empty: ignore emptry nodes

    :param ignore_keys: ignore the following keys

    :param ignore_default_keys: ignores the following keys from the comparison
                            %s

    :return: string with reason for difference, or False otherwise
    """
    from omas import ODS, CodeParameters

    ods1 = ods1.flat(return_empty_leaves=True, traverse_code_parameters=True)
    ods2 = ods2.flat(return_empty_leaves=True, traverse_code_parameters=True)

    keys_to_ignore = []
    keys_to_ignore.extend(ignore_keys)
    if ignore_default_keys:
        keys_to_ignore.extend(default_keys_to_ignore)

    def is_ignored(k):
        return any(o2u(k).endswith(end) for end in keys_to_ignore)

    k1 = set(ods1.keys())
    k2 = set(ods2.keys())
    differences = []
    for k in k1.difference(k2):
        if not k.startswith('info.') and not (ignore_empty and isinstance(ods1[k], ODS) and not len(ods1[k])) and not is_ignored(k):
            differences.append(f'DIFF: key `{k}` missing in 2nd ods')
    for k in k2.difference(k1):
        if not k.startswith('info.') and not (ignore_empty and isinstance(ods2[k], ODS) and not len(ods2[k])) and not is_ignored(k):
            differences.append(f'DIFF: key `{k}` missing in 1st ods')
    for k in k1.intersection(k2):
        try:
            if is_ignored(k):
                pass
            elif ods1[k] is None and ods2[k] is None:
                pass
            elif isinstance(ods1[k], str) and isinstance(ods2[k], str):
                if ods1[k] != ods2[k]:
                    differences.append(f'DIFF: `{k}` differ in value')
            elif not ignore_type and type(ods1[k]) != type(ods2[k]):
                differences.append(f'DIFF: `{f}` differ in type: {type(ods1[k])} vs type(ods2[k])')
            elif is_uncertain(ods1[k]) or is_uncertain(ods2[k]):
                v1 = nominal_values(ods1[k])
                v2 = nominal_values(ods2[k])
                d1 = std_devs(ods1[k])
                d2 = std_devs(ods1[k])
                s1 = v1.shape
                s2 = v2.shape
                if s1 != s2:
                    differences.append(f'DIFF: `{k}` differ in shape: {s1} vs {s2}')
                elif not numpy.allclose(v1, v2, equal_nan=True) or not numpy.allclose(d1, d2, equal_nan=True):
                    differences.append(f'DIFF: `{k}` differ in value')
            else:
                v1 = nominal_values(ods1[k])
                v2 = nominal_values(ods2[k])
                s1 = v1.shape
                s2 = v2.shape
                if v1.shape != v2.shape:
                    differences.append(f'DIFF: `{k}` differ in shape: {s1} vs {s2}')
                elif not numpy.allclose(ods1[k], ods2[k], equal_nan=True):
                    differences.append(f'DIFF: `{k}` differ in value')
        except Exception as _excp:
            raise Exception(f'Error comparing {k}: ' + repr(_excp))
    if len(differences):
        return differences
    else:
        return False


different_ods.__doc__ = different_ods.__doc__ % '\n                            '.join(default_keys_to_ignore)


def different_ods_attrs(ods1, ods2, attrs=None, verbose=False):
    '''
    Checks if two ODSs have any difference in their attributes

    :param ods1: first ods to check

    :param ods2: second ods to check

    :param attrs: list of attributes to compare

    :param verbose: print differences to stdout

    :return: dictionary with list of attriibutes that have differences, or False otherwise
    '''

    if isinstance(attrs, str):
        attrs = [attrs]
    elif attrs is None:
        from .omas_core import omas_ods_attrs

        attrs = omas_ods_attrs

    if '_parent' in attrs:
        attrs.pop(attrs.index('_parent'))

    n = max(list(map(lambda x: len(x), attrs)))
    l1 = set(list(map(lambda x: l2i(x[:-1]), ods1.paths(return_empty_leaves=True, traverse_code_parameters=False))))
    l2 = set(list(map(lambda x: l2i(x[:-1]), ods2.paths(return_empty_leaves=True, traverse_code_parameters=False))))
    paths = sorted(list(l1.intersection(l2)))

    differences = {}
    for item in paths:
        first = True
        try:
            for k in attrs:
                a1 = getattr(ods1[item], k)
                a2 = getattr(ods2[item], k)
                if a1 != a2:
                    if first:
                        if verbose:
                            print('-' * 20)
                            print(item)
                            print('-' * 20)
                        differences[item] = []
                        first = False
                    differences[item].append(k)
                    if verbose:
                        print(k.ljust(n) + ': * %s' % a1)
                        print('`%s * %s' % (' '.ljust(n), a2))
        except:
            raise
    if len(differences):
        return differences
    else:
        return False


# --------------------------
# general utility functions
# --------------------------
_streams = {'DEBUG': sys.stderr, 'STDERR': sys.stderr}


def printd(*objects, **kw):
    """
    debug print

    Use environmental variable $OMAS_DEBUG_TOPIC to set the topic to be printed
    """
    topic = kw.pop('topic', '')
    if isinstance(topic, str):
        topic = [topic]
    topic = list(map(lambda x: x.lower(), topic))
    if len(topic):
        objects = [f'DEBUG ({",".join(topic)}):'] + list(objects)
    else:
        objects = ['DEBUG:'] + list(objects)
    topic_selected = os.environ.get('OMAS_DEBUG_TOPIC', '')
    dump = False
    if topic_selected.endswith('_dump'):
        dump = True
        topic_selected = re.sub('_dump$', '', topic_selected)
    if topic_selected and (topic_selected == '*' or topic_selected in topic or '*' in topic):
        if eval(os.environ.get('OMAS_DEBUG_STDOUT', '0')):
            kw.setdefault('file', sys.stdout)
        else:
            kw.setdefault('file', _streams['DEBUG'])
        print(*objects, **kw)
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
    kw['file'] = _streams['STDERR']
    print(*objects, **kw)


def print_stack():
    return traceback.print_stack(file=sys.__stderr__)


def is_uncertain(var):
    """
    :param var: Variable or array to test

    :return: True if input variable or array is uncertain
    """

    def _uncertain_check(x):
        return isinstance(x, uncertainties.core.AffineScalarFunc)

    if isinstance(var, str):
        return False
    elif numpy.iterable(var) or isinstance(var, numpy.ndarray):  # isinstance needed for 0D arrays from squeeze
        tmp = numpy.array(var)
        if tmp.dtype not in ['O', 'object']:
            return False
        else:
            # the argument of any is a generator object (using a list slows things down)
            return any(_uncertain_check(x) for x in tmp.flat)
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


def omas_interp1d(x, xp, yp, left=None, right=None, period=None, extrapolate=True):
    """
    If xp is not increasing, the results are numpy.interp1d nonsense.
    This function wraps numpy.interp1d but makes sure that the x-coordinate sequence xp is increasing.

    :param extrapolate: linear extrapolation beyond bounds

    """
    if not numpy.all(numpy.diff(xp) > 0):
        index = numpy.argsort(xp)
    else:
        index = numpy.arange(len(xp)).astype(int)
    y = numpy.interp(x, xp[index], yp[index], left=left, right=right, period=period)
    if extrapolate:
        if not period and not left:
            y = numpy.where(
                x < xp[index[0]], yp[index[0]] + (x - xp[index[0]]) * (yp[index[0]] - yp[index[1]]) / (xp[index[0]] - xp[index[1]]), y
            )
        if not period and not right:
            y = numpy.where(
                x > xp[index[-1]],
                yp[index[-1]] + (x - xp[index[-1]]) * (yp[index[-1]] - yp[index[-2]]) / (xp[index[-1]] - xp[index[-2]]),
                y,
            )
    return y


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

    if objects_encode is False:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, (range, map)):
            return list(obj)
        elif isinstance(obj, numpy.generic):
            return obj.item()
        else:
            return obj.toJSON()

    else:
        if is_uncertain(obj):
            if not len(numpy.array(obj).shape):
                return dict(__ufloat__=nominal_values(obj), __ufloat_std__=std_devs(obj))
            else:
                nomv = nominal_values(obj)
                return dict(
                    __udarray_tolist_avg__=nomv.tolist(),
                    __udarray_tolist_std__=std_devs(obj).tolist(),
                    dtype=str(nomv.dtype),
                    shape=obj.shape,
                )
        elif isinstance(obj, numpy.ndarray):
            if 'complex' in str(obj.dtype).lower():
                return dict(
                    __ndarray_tolist_real__=obj.real.tolist(),
                    __ndarray_tolist_imag__=obj.imag.tolist(),
                    dtype=str(obj.dtype),
                    shape=obj.shape,
                )
            else:
                if objects_encode is None:
                    return obj.tolist()
                else:
                    return dict(__ndarray_tolist__=obj.tolist(), dtype=str(obj.dtype), shape=obj.shape)
        elif isinstance(obj, range):
            return list(obj)
        elif isinstance(obj, numpy.generic):
            return obj.item()
        elif isinstance(obj, complex):
            return dict(__complex__=True, real=obj.real, imag=obj.imag)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        else:
            return obj.toJSON()


def convert_int(value):
    """
    Try to convert value to integer and do nothing on error

    :param value: value to try to convert

    :return: value, possibly converted to int
    """
    try:
        return int(value)
    except ValueError:
        return value


def json_loader(object_pairs, cls=dict, null_to=None):
    """
    Load json-objects generated by the json_dumper function

    :param object_pairs: json-compatible [dict/list] object

    :param cls: dicitonary class to use

    :param null_to: convert null to user defined value (None by default)

    :return: ojbect
    """
    from omas import ODS

    object_pairs = list(map(lambda o: (convert_int(o[0]), o[1]), object_pairs))

    dct = cls()
    # for ODSs we can use the setraw() method which does
    # not peform any sort of check, nor tries to parse
    # special OMAS syntaxes and is thus much faster
    if isinstance(dct, ODS):
        for x, y in object_pairs:
            if null_to is not None and y is None:
                y = null_to
            if isinstance(y, list):
                if len(y) and isinstance(y[0], ODS):
                    dct.setraw(x, cls())
                    for k in range(len(y)):
                        dct[x].setraw(k, y[k])
                else:
                    if null_to is not None:
                        for k in range(len(y)):
                            if y[k] is None:
                                y[k] = null_to
                    y = numpy.array(y)  # to handle objects_encode=None as used in OMAS
                    dct.setraw(x, y)
            else:
                dct.setraw(x, y)
    else:
        for x, y in object_pairs:
            if null_to is not None and y is None:
                y = null_to
            if isinstance(y, list):
                if len(y) and isinstance(y[0], ODS):
                    dct[x] = cls()
                    for k in range(len(y)):
                        dct[x][k] = y[k]
                else:
                    if null_to is not None:
                        for k in range(len(y)):
                            if y[k] is None:
                                y[k] = null_to
                    dct[x] = y
            else:
                dct[x] = y

    if "dtype" in dct:  # python2/3 compatibility
        dct["dtype"] = dct["dtype"].replace('S', 'U')
    if '__ndarray_tolist__' in dct:
        return numpy.array(dct['__ndarray_tolist__'], dtype=dct['dtype']).reshape(dct['shape'])
    elif '__ndarray_tolist_real__' in dct and '__ndarray_tolist_imag__' in dct:
        return (
            numpy.array(dct['__ndarray_tolist_real__'], dtype=dct['dtype']).reshape(dct['shape'])
            + numpy.array(dct['__ndarray_tolist_imag__'], dtype=dct['dtype']).reshape(dct['shape']) * 1j
        )
    elif '__udarray_tolist_avg__' in dct and '__udarray_tolist_std__' in dct:
        return uarray(
            numpy.array(dct['__udarray_tolist_avg__'], dtype=dct['dtype']).reshape(dct['shape']),
            numpy.array(dct['__udarray_tolist_std__'], dtype=dct['dtype']).reshape(dct['shape']),
        )
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
    """
    function used to remove/replace top-level matching parenthesis from a string

    :param inv: input string

    :param replace_with: string to replace matching parenthesis with

    :return: input string without first set of matching parentheses
    """
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
    Given a SORTED iterable (a numeric array or list of numbers) and a numeric scalar my_number,
    find the index of the number in the list that is closest to my_number

    :param my_list: Sorted iterable (list or array) to search for number closest to my_number

    :param my_number: Number to get close to in my_list

    :return: Index of my_list element closest to my_number

    :note: If two numbers are equally close, returns the index of the smallest number.
    """
    if not hasattr(my_list, '__iter__'):
        raise TypeError("closestIndex() requires an iterable as the first argument. Got instead: {:}".format(my_list))
    if not is_numeric(my_number):
        raise TypeError("closestIndex() requires a numeric scalar as the second argument. Got instead: {:}".format(my_number))

    import bisect

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


def underline_last(text, offset=0):
    """
    Utility function to underline the last part of a path

    :param text: text to underline

    :param offset: add offset to underling

    :return: original text with underline on a new line
    """
    index = [i for i, x in enumerate(text) if x in ['.', ' ']][-1]
    if text[index] == '.':
        index += 1
    underline = ' ' * (index + offset) + '^' * (len(text) - index)
    return text + '\n' + underline


def function_arguments(f, discard=None, asString=False):
    """
    Returns the arguments that a function takes

    :param f: function to inspect

    :param discard: list of function arguments to discard

    :param asString: concatenate arguments to a string

    :return: tuple of four elements

    * list of compulsory function arguments

    * dictionary of function arguments that have defaults

    * True/False if the function allows variable arguments

    * True/False if the function allows keywords
    """
    import inspect

    the_argspec = inspect.getfullargspec(f)
    the_keywords = the_argspec.varkw

    args = []
    kws = OrderedDict()
    string = ''
    for k, arg in enumerate(the_argspec.args):
        if (discard is not None) and (arg in tolist(discard)):
            continue
        d = ''
        if the_argspec.defaults is not None:
            if (-len(the_argspec.args) + k) >= -len(the_argspec.defaults):
                d = the_argspec.defaults[-len(the_argspec.args) + k]
                kws[arg] = d
                string += arg + '=' + repr(d) + ',\n'
            else:
                args.append(arg)
                string += arg + ',\n'
        else:
            args.append(arg)
            string += arg + ',\n'
        if the_argspec.varargs:
            string += '*[],\n'
        if the_keywords:
            string += '**{},\n'
        string = string.strip()
    if asString:
        return string
    else:
        return args, kws, the_argspec.varargs is not None, the_keywords is not None


def args_as_kw(f, args, kw):
    """
    Move positional arguments to kw arguments

    :param f: function

    :param args: positional arguments

    :param kw: keywords arguments

    :return: tuple with positional arguments moved to keyword arguments
    """
    a, k, astar, kstar = function_arguments(f)
    if len(a) and a[0] == 'self':
        a = a[1:]
    a = a + list(k.keys())
    n = 0
    for name, value in zip(a + list(k.keys()), args):
        if name not in kw:
            kw[name] = value
        n += 1
    return args[n:], kw


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
# cache structures filenames
_structures_filenames = {}
# cache for structure()
_ods_structure_cache = {}
# similar to `_structures_dict` but for use in omas_info
_info_structures = {}
# dictionary that contains all the coordinates defined within the data dictionary
_coordinates = {}
# dictionary that contains all the times defined within the data dictionary
_times = {}
# dictionary that contains all the _global_quantities defined within the data dictionary
_global_quantities = {}

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
    """
    list names of structures in imas version

    :param imas_version: imas version

    :return: list with names of structures in imas version
    """
    json_filenames = glob.glob(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '*' + '.json')
    json_filenames = filter(lambda x: os.path.basename(x)[0] != '_', json_filenames)
    structures = sorted(list(map(lambda x: os.path.splitext(os.path.split(x)[1])[0], json_filenames)))
    if not len(structures):
        raise ValueError("Unrecognized IMAS version `%s`. Possible options are:\n%s" % (imas_version, imas_versions.keys()))
    return structures


def structures_filenames(imas_version):
    """
    Maps structure names to json filenames

    :param imas_version: imas version

    :return: dictionary maps structure names to json filenames
    """
    if imas_version not in _structures_filenames:
        paths = glob.glob(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '*' + '.json')
        if not len(paths):
            raise ValueError("Unrecognized IMAS version `%s`. Possible options are:\n%s" % (imas_version, imas_versions.keys()))
        structures = dict(zip(list(map(lambda x: os.path.splitext(os.path.split(x)[1])[0], paths)), paths))
        _structures_filenames[imas_version] = {
            structure: structures[structure] for structure in structures if not structure.startswith('_')
        }
    return _structures_filenames[imas_version]


def load_structure(filename, imas_version):
    """
    load omas structure from given json filename or IDS name

    :param filename: full path to json file or IDS name

    :param imas_version: imas version to load the data schema of (optional if filename is a full path)

    :return: tuple with structure, hashing mapper, and ods
    """

    from .omas_physics import cocos_signals

    # translate DS to filename
    if os.sep not in filename:
        filename = structures_filenames(imas_version)[filename]

    # check if _structures and _structures_dict already have this in cache
    id = (filename, imas_version)
    if id in _structures and id in _structures_dict:
        return _structures[id], _structures_dict[id]

    else:
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


def imas_structure(imas_version, location):
    '''
    Returns a dictionary with the IMAS structure given a location

    :param imas_version: imas version

    :param location: path in OMAS format

    :return: dictionary as loaded by load_structure() at location
    '''
    if imas_version not in _ods_structure_cache:
        _ods_structure_cache[imas_version] = {}
    ulocation = o2u(location)
    if ulocation not in _ods_structure_cache[imas_version]:
        if not ulocation:
            structure = {k: k for k in list_structures(imas_version=imas_version)}
        else:
            path = p2l(ulocation)
            structure = load_structure(path[0], imas_version=imas_version)[1][path[0]]
            for key in path[1:]:
                structure = structure[key]
        _ods_structure_cache[imas_version][ulocation] = structure
    return _ods_structure_cache[imas_version][ulocation]


def omas_coordinates(imas_version=omas_rcparams['default_imas_version']):
    """
    return list of coordinates

    :param imas_version: IMAS version to look up

    :return: list of strings with IMAS coordinates
    """
    # caching
    if imas_version not in _coordinates:
        filename = imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '_coordinates.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                _coordinates[imas_version] = json.load(f)
        else:
            from .omas_structure import extract_coordinates

            _coordinates[imas_version] = extract_coordinates(imas_version)
    return _coordinates[imas_version]


def omas_times(imas_version=omas_rcparams['default_imas_version']):
    """
    return list of times

    :param imas_version: IMAS version to look up

    :return: list of strings with IMAS times
    """
    # caching
    if imas_version not in _times:
        filename = imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '_times.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                _times[imas_version] = json.load(f)
        else:
            from .omas_structure import extract_times

            _times[imas_version] = extract_times(imas_version)
    return _times[imas_version]


def omas_global_quantities(imas_version=omas_rcparams['default_imas_version']):
    """
    return list of times

    :param imas_version: IMAS version to look up

    :return: list of strings with IMAS times
    """
    # caching
    if imas_version not in _global_quantities:
        filename = imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '_global_quantities.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                _global_quantities[imas_version] = json.load(f)
        else:
            from .omas_structure import extract_global_quantities

            _global_quantities[imas_version] = extract_global_quantities(imas_version)
    return _global_quantities[imas_version]


# only attempt cython if user owns this copy of omas
if os.environ['USER'] != pwd.getpwuid(os.stat(__file__).st_uid).pw_name:
    with open(os.path.split(__file__)[0] + os.sep + 'omas_cython.pyx', 'r') as f:
        exec(f.read(), globals())
else:
    try:
        import pyximport

        pyximport.install(language_level=3)
        from .omas_cython import *
    except Exception as _excp:
        warnings.warn('omas cython failed: ' + str(_excp))
        with open(os.path.split(__file__)[0] + os.sep + 'omas_cython.pyx', 'r') as f:
            exec(f.read(), globals())


def l2ut(path):
    """
    Formats IMAS time lists ['bla',0,'time_slice',5,'quantity'] with universal ODS path 'bla.0.time_slice.:.quantity'

    :param path: list of strings and integers

    :return: ODS path format with time lists in universal format
    """
    lpath = p2l(path)
    opath = l2o(lpath)
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


def omas_info(structures=None, hide_obsolescent=True, cumulative_queries=False, imas_version=omas_rcparams['default_imas_version']):
    """
    This function returns an ods with the leaf nodes filled with their property informations

    :param hide_obsolescent: hide obsolescent entries

    :param structures: list with ids names or string with ids name of which to retrieve the info
                       if None, then all structures are returned

    :param cumulative_queries: return all IDSs that have been queried

    :param imas_version: IMAS version to look up

    :return: ods showcasing IDS structure
    """

    from omas import ODS

    if not structures:
        structures = sorted(list(structures_filenames(imas_version).keys()))
    elif isinstance(structures, str):
        structures = [structures]

    # caching based on imas version and obsolescence
    if (imas_version, hide_obsolescent) not in _info_structures:
        _info_structures[imas_version, hide_obsolescent] = ODS(imas_version=imas_version, consistency_check=False)
    ods = _info_structures[imas_version, hide_obsolescent]

    ods_out = ODS(imas_version=imas_version, consistency_check=False)

    # generate ODS with info
    for structure in structures:
        if structure not in ods:
            tmp = load_structure(structure, imas_version)[0]
            lst = sorted(tmp.keys())
            for k, item in enumerate(lst):
                if re.match('.*_error_(index|lower|upper)$', item.split('.')[-1]):
                    continue
                parent = False
                for item1 in lst[k + 1 :]:
                    if l2u(item1.split('.')[:-1]).rstrip('[:]') == item:
                        parent = True
                        break
                if parent:
                    continue
                if hide_obsolescent and omas_info_node(item).get('lifecycle_status', '') == 'obsolescent':
                    continue
                ods[item.replace(':', '0')] = tmp[item]
        ods_out[structure] = ods[structure]

    # cumulative queries
    if cumulative_queries:
        for structure in ods:
            if structure not in ods_out:
                ods_out[structure] = ods[structure]

    return ods_out


def omas_info_node(key, imas_version=omas_rcparams['default_imas_version']):
    """
    return information about a given node

    :param key: IMAS path

    :param imas_version: IMAS version to look up

    :return: dictionary with IMAS information (or an empty dictionary if the node is not found)
    """
    try:
        return copy.copy(load_structure(key.split('.')[0], imas_version)[0][o2i(key)])
    except KeyError:
        return {}


def recursive_interpreter(me, interpret_method=ast.literal_eval, dict_cls=OrderedDict):
    """
    Traverse dictionaries and list to convert strings to int/float when appropriate

    :param me: root of the dictionary to traverse

    :param interpret_method: method used for conversion (ast.literal_eval by default)

    :param dict_cls: dictionary class to use

    :return: root of the dictionary
    """
    if isinstance(me, list):
        keys = range(len(me))
    elif isinstance(me, dict):
        keys = me.keys()

    for kid in keys:
        if me[kid] is None:
            continue
        elif isinstance(me[kid], (list, dict)):
            if not isinstance(me[kid], dict_cls):
                tmp = me[kid]
                me[kid] = dict_cls()
                me[kid].update(tmp)
            recursive_interpreter(me[kid], interpret_method=interpret_method, dict_cls=dict_cls)
            if isinstance(kid, str) and kid.startswith('__integer_'):
                me[int(re.sub('__integer_([0-9]+)__', r'\1', kid))] = me[kid]
                del me[kid]
        else:
            try:
                me[kid] = interpret_method(me[kid])
            except (ValueError, SyntaxError) as _excp:
                pass
            if isinstance(me[kid], str) and ' ' in me[kid]:
                try:
                    values = []
                    for item in re.split(r'[ |\t]+', me[kid].strip()):
                        values.append(float(item))
                    me[kid] = numpy.array(values)
                except ValueError:
                    pass
    return me


def recursive_encoder(me):
    """
    Traverse dictionaries and list to convert entries as appropriate

    :param me: root of the dictionary to traverse

    :return: root of the dictionary
    """
    if isinstance(me, list):
        keys = range(len(me))
    elif isinstance(me, dict):
        keys = me.keys()

    for kid in keys:
        if me[kid] is None:
            continue
        elif isinstance(me[kid], (list, dict)):
            recursive_encoder(me[kid])
        else:
            if isinstance(me[kid], numpy.ndarray):
                me[kid] = ' '.join([repr(x) for x in me[kid]])
            else:
                me[kid] = str(me[kid])
        # omas encoding of integer keys
        if isinstance(kid, int):
            me['__integer_%d__' % kid] = me[kid]
            del me[kid]
    return me


def get_actor_io_ids(filename):
    """
    Parse IMAS Python actor script and return actor input and output IDSs

    :param filename: filename of the IMAS Python actor

    :return: tuple with list of input IDSs and output IDSs
    """
    import ast

    with open(filename, 'r') as f:
        module = ast.parse(f.read())
    actor = os.path.splitext(os.path.split(filename)[-1])[0]
    function_definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    docstring = ast.get_docstring([f for f in function_definitions if f.name == actor][0])
    ids_in = []
    ids_out = []
    for line in docstring.split('\n'):
        if 'codeparams' in line:
            pass
        elif line.strip().startswith(':param result:'):
            ids_out = list(map(lambda x: x.strip()[:-1], line.split(':')[2].strip(', ').split(',')))
            break
        elif line.strip().startswith(':param '):
            ids_in.append(line.split(':')[2].strip())
    return ids_in, ids_out


class UnittestCaseOmas(unittest.TestCase):
    """
    Base class for unittest.TestCase within OMAS
    """

    def setUp(self):
        name = self.__class__.__name__ + '.' + self._testMethodName
        print('')
        print('~' * len(name))
        print(name)
        print('~' * len(name))
