'''save/load from ASCII routines

-------
'''

from .omas_utils import *
from .omas_core import ODS, ODC, force_imas_type


def identify_imas_type(value):
    if isinstance(value, (str, numpy.string_, numpy.unicode_, numpy.str_)):
        dtype = dict(type='50 (CHAR_DATA)', dim=1, size=(len(value),))
    elif isinstance(value, (float, numpy.floating)):
        dtype = dict(type='52 (DOUBLE_DATA)', dim=0)
    elif isinstance(value, (int, numpy.integer)):
        dtype = dict(type='51 (INTEGER_DATA)', dim=0)
    elif isinstance(value, numpy.ndarray):
        if 'str' in value.dtype.name:
            dtype = dict(type='50 (CHAR_DATA)', dim=len(value.shape), size=value.shape)
        elif 'float' in value.dtype.name:
            dtype = dict(type='52 (DOUBLE_DATA)', dim=len(value.shape), size=value.shape)
        elif 'int' in value.dtype.name:
            dtype = dict(type='51 (INTEGER_DATA)', dim=len(value.shape), size=value.shape)
        else:
            raise ValueError(str(value.dtype.name) + ' is not a valid IMAS data type')
    elif isinstance(value, ODS):
        dtype = dict(dim=len(value))
    else:
        raise ValueError(str(type(value)) + ' is not a valid IMAS data type')
    return dtype


imas_nan = -9e40


def imas_fmt(value):
    if isinstance(value, (float, numpy.floating)):
        if numpy.isnan(value):
            return '%5.16e' % imas_nan
        else:
            return '%5.16e' % value
    else:
        return '%s' % value


def imas_eval(value):
    tmp = ast.literal_eval(value)
    if tmp == imas_nan:
        return numpy.nan
    else:
        return tmp


def imas_ascii_key_sorter(keys, ods_has_location):
    new_order = [
        'ids_properties.homogeneous_time',
        'ids_properties.version_put.data_dictionary',
        'ids_properties.version_put.access_layer',
        'ids_properties.version_put.access_layer_language',
    ]

    if ods_has_location:

        for key in reversed(new_order):
            if key in keys:
                keys.insert(0, keys.pop(keys.index(key)))

    return keys


# ---------------------------
# save and load OMAS to ASCII
# ---------------------------
def save_omas_ascii(ods, filename, machine=None, pulse=None, run=None, dir=None):
    """
    Save an ODS to ASCII (follows IMAS ASCII_BACKEND convention)

    :param ods: OMAS data set

    :param filename: filename or file descriptor to save to
                     use `None` to save IDSs to multiple files based on machine, pulse, run

    :param machine: machine name to build filename for saving IDSs to multiple files

    :param pulse: pulse number to build filename for saving IDSs to multiple files

    :param run: run number to build filename for saving IDSs to multiple files

    :param dir: directory where to save multiple IDSs files
    """

    if filename is None and machine is not None and pulse is not None and run is not None:
        for ds in ods:
            filename = f'{machine}_{pulse}_{run}_{ds}.ids'
            if dir:
                filename = dir + os.sep + filename
            save_omas_ascii(ods[ds], filename, machine=None, pulse=None, run=None)
        return

    elif filename is not None and machine is None and pulse is None and run is None:
        pass

    else:
        raise ValueError('Either `filename` or `machine, pulse, run` can be specified')

    printd('Saving OMAS data to ASCII: %s' % filename, topic='ascii')

    ods.satisfy_imas_requirements()

    ascii_string = []
    for path in imas_ascii_key_sorter(ods.pretty_paths(include_structures=True), ods.location):
        value = ods[path]
        if isinstance(value, ODS) and not isinstance(value.omas_data, list):
            continue
        value = force_imas_type(value)
        info = identify_imas_type(value)
        tokens = []
        if ods.location:
            tokens.append(ods.location + '.' + path.replace('.', '/'))
        else:
            tokens.append(path.replace('.', '/'))
        if 'type' in info:
            tokens.append('	type: ' + info['type'])
        if 'dim' in info:
            tokens.append('	dim: %d' % info['dim'])
        if 'size' in info:
            tokens.append('	size: %s' % (' '.join(map(str, info['size']))))
        if isinstance(value, ODS):
            pass
        elif not isinstance(value, numpy.ndarray):
            tokens.append(imas_fmt(value))
        elif len(value.shape) == 1:
            tokens.append(' '.join(map(imas_fmt, value)))
        elif len(value.shape) == 2:
            for row in value.T:
                tokens.append(' '.join(map(imas_fmt, row)))
        else:
            raise ValueError(f'{path} not implemented ASCII support for number of dimensions >2')

        ascii_string.extend(tokens)

    ascii_string = '\n'.join(ascii_string)

    if isinstance(filename, str):
        with open(filename, 'w') as f:
            f.write(ascii_string)
    else:
        f = filename
        f.write(ascii_string)


def load_omas_ascii(
    filename, machine=None, pulse=None, run=None, dir=None, consistency_check=True, imas_version=omas_rcparams['default_imas_version']
):
    """
    Load an ODS from ASCII (follows IMAS ASCII_BACKEND convention)

    :param filename: filename or file descriptor to load from
                     use `None` to load IDSs from multiple files based on machine, pulse, run

    :param machine: machine name to build filename for loading IDSs from multiple files

    :param pulse: pulse number to build filename for loading IDSs from multiple files

    :param run: run number to build filename for loading IDSs from multiple files

    :param dir: directory from where to load multiple IDSs files

    :param consistency_check: verify that data is consistent with IMAS schema

    :param imas_version: imas version to use for consistency check

    :return: OMAS data set
    """

    if filename is None and machine is not None and pulse is not None and run is not None:
        filename = f'{machine}_{pulse}_{run}_*.ids'
        if dir:
            filename = dir + os.sep + filename
        ods = ODS(consistency_check=True, imas_version=omas_rcparams['default_imas_version'])
        for filename in glob.glob(filename):
            ods.update(load_omas_ascii(filename, machine=None, pulse=None, run=None))
        return ods

    elif filename is not None and machine is None and pulse is None and run is None:
        pass

    else:
        raise ValueError('Either `filename` or `machine, pulse, run` can be specified')

    printd('Loading OMAS data from ASCCI: %s' % filename, topic='ascii')

    if isinstance(filename, str):
        with open(filename, 'r') as f:
            ascii_string = f.read()
    else:
        ascii_string = filename.read()

    tokens = OrderedDict()
    value_lines = 0
    path = None
    token = None
    for line in ascii_string.split('\n'):
        if path is None:
            path = line.strip().replace('/', '.')
            tokens[path] = token = {'path': path}
            continue
        elif value_lines:
            token.setdefault('value', []).append(line)
            value_lines -= 1
            if value_lines == 0:
                path = None
            continue
        elif line.startswith('	type: '):
            token['type'] = line.split('type:')[1].strip()
        elif line.startswith('	dim: '):
            token['dim'] = int(line.split('dim:')[1].strip())
        elif line.startswith('	size: '):
            token['size'] = tuple(map(int, line.split('size:')[1].strip().split()))

        # scalar INT or FLOAT
        if 'type' in token and 'dim' in token and token['dim'] == 0:
            value_lines = 1
        # string
        elif 'type' in token and token['type'] == '50 (CHAR_DATA)' and 'dim' in token and token['dim'] == 1 and 'size' in token:
            value_lines = 1
        # 1D arrays
        elif 'type' in token and 'dim' in token and token['dim'] == 1 and 'size' in token:
            value_lines = 1
        # 2D arrays
        elif 'type' in token and 'dim' in token and token['dim'] == 2 and 'size' in token:
            value_lines = token['size'][1]
        # ODS
        elif 'type' not in token and 'dim' in token:
            value_lines = 0
            path = None

    ods = ODS(imas_version=imas_version, consistency_check=consistency_check)

    for token in tokens.values():
        path = token['path']
        # scalar INT or FLOAT
        if 'type' in token and 'dim' in token and token['dim'] == 0:
            value = imas_eval(token['value'][0])
        # string
        elif 'type' in token and token['type'] == '50 (CHAR_DATA)' and 'dim' in token and token['dim'] == 1:
            value = token['value'][0]
        # 1D arrays
        elif 'type' in token and 'dim' in token and token['dim'] == 1 and 'size' in token:
            value = numpy.array(list(map(imas_eval, token['value'][0].split())))
        # 2D arrays
        elif 'type' in token and 'dim' in token and token['dim'] == 2 and 'size' in token:
            value = numpy.array([list(map(imas_eval, row.split())) for row in token['value']]).T
        # ODS
        elif 'type' not in token and 'dim' in token:
            continue

        ods[path] = value

    return ods


def through_omas_ascii(ods, method=['function', 'class_method'][1], one_or_many_files=['one', 'many'][1]):
    """
    Test save and load OMAS ASCII

    :param ods: ods

    :return: ods
    """
    if one_or_many_files == 'one':
        filename = omas_testdir(__file__) + os.sep + 'test.ids'
        machine = None
        pulse = None
        run = None
        dir = None
    else:
        if method == 'function':
            filename = None
        else:
            filename = 'ascii'
        machine = 'TEST'
        pulse = 1
        run = 0
        dir = omas_testdir(__file__)
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
    if method == 'function':
        save_omas_ascii(ods, filename, machine, pulse, run, dir)
        ods1 = load_omas_ascii(filename, machine, pulse, run, dir)
    else:
        ods.save(filename, machine, pulse, run, dir)
        ods1 = ODS().load(filename, machine, pulse, run, dir)
    return ods1
