# https://confluence.iter.org/display/IMP/UDA+data+mapping+tutorial

import urllib
from .omas_utils import *
from . import ODS, omas_environment, omas_info_node, imas_json_dir, omas_rcparams
from .omas_core import dynamic_ODS
from .omas_physics import cocos_signals

__all__ = ['expression_types', 'machine_to_omas', 'machines', 'machine_mappings', 'load_omas_machine']

_mapping_functions_namespace = {}

url_dir = os.sep.join([omas_rcparams['tmp_omas_dir'], 'machine_mappings', '{branch}', 'omas_machine_mappings_url'])

expression_types = ['VALUE', 'ENVIRON', 'PYTHON', 'TDI', 'eval2TDI']


def mapping_functions_namespace(branch):
    # return cached mapping function namespace
    if _mapping_functions_namespace.get('__branch__', False) == branch:
        return _mapping_functions_namespace

    # clear mapping function namespace
    _mapping_functions_namespace.clear()

    # remove old machine_mappings modules
    if 'omas.machine_mappings' in sys.modules:
        del sys.modules['omas.machine_mappings']
    if 'omas.machine_mappings.mapping_functions' in sys.modules:
        del sys.modules['omas.machine_mappings.mapping_functions']

    # get local mapping functions
    if branch is None:
        exec('from omas.machine_mappings.mapping_functions import *', _mapping_functions_namespace)
    # get mapping functions from GitHub
    else:
        printd(f'omas machine mappings functions from branch: `{branch}`', topic='machine')

        # setup temporary import directory
        dir = url_dir.format(branch=branch)
        if not os.path.exists(dir):
            os.makedirs(dir)
        open(dir + os.sep + '__init__.py', 'w').close()

        # download the mapping_functions.py and write it to file
        url = f"https://raw.githubusercontent.com/gafusion/omas/{branch}/omas/machine_mappings/mapping_functions.py"
        contents = urllib.request.urlopen(url).read().decode("utf-8")
        with open(dir + os.sep + 'mapping_functions.py', 'w') as f:
            f.write(contents)

        # import from temporary directory
        if dir + os.sep + '..' not in sys.path:
            sys.path.insert(0, dir + os.sep + '..')
        exec('from omas_machine_mappings_url.mapping_functions import *', _mapping_functions_namespace)

        # alias
        sys.modules['omas.machine_mappings'] = sys.modules['omas_machine_mappings_url']
        sys.modules['omas.machine_mappings.mapping_functions'] = sys.modules['omas_machine_mappings_url.mapping_functions']

    _mapping_functions_namespace['__branch__'] = branch
    return _mapping_functions_namespace


def update_mapping(machine, location, value):
    new_raw_mappings = machine_mappings(machine, None, None, return_raw_mappings=True)
    new_raw_mappings[location] = value
    filename, branch = machines(machine, None)
    with open(filename, 'w') as f:
        json.dump(new_raw_mappings, f, indent=1, separators=(',', ': '), sort_keys=True)
    print(f"Updated {filename}")


def machine_to_omas(ods, machine, pulse, location, options={}, branch=None, user_machine_mappings=None):
    '''
    Routine to convert machine data to ODS

    :param ods: input ODS to populate

    :param machine: machine name

    :param pulse: pulse number

    :param location: ODS location to be populated

    :param options: dictionary with options to use when loadinig the data

    :param branch: load machine mappings and mapping functions from a specific GitHub branch

    :param user_mappings: allow specification of external mappings

    :return: updated ODS and data before being assigned to the ODS
    '''
    try:
        from classes.omfit_mds import OMFITmdsValue
    except (ModuleNotFoundError, ImportError):
        from omfit.classes.omfit_mds import OMFITmdsValue

    if user_machine_mappings is None:
        user_machine_mappings = {}

    location = l2o(p2l(location))
    mappings = machine_mappings(machine, branch, user_machine_mappings)
    options_with_defaults = copy.copy(mappings['__options__'])
    options_with_defaults.update(options)
    options_with_defaults.update({'machine': machine, 'pulse': pulse, 'location': location})
    mapped = mappings[location]

    cocosio = None

    # CONSTANT VALUE
    if 'VALUE' in mapped:
        data0 = data = mapped['VALUE']
        if isinstance(data0, str):
            data0 = data = data0.format(**options_with_defaults)

    # ENVIRONMENTAL VARIABLE
    elif 'ENVIRON' in mapped:
        data0 = data = os.environ[mapped['ENVIRON'].format(**options_with_defaults)]

    # PYTHON
    elif 'PYTHON' in mapped:
        namespace = {}
        namespace.update(_namespace_mappings[machine])
        namespace['ods'] = ODS()
        exec(mapped['PYTHON'].format(**options_with_defaults), namespace)
        data0 = namespace[mapped.get('RETURN','ods')]
        data = data0[location]

        for loc in numpy.unique(list(map(o2u, data0.flat().keys()))):
            if loc not in mappings:
                update_mapping(machine, loc, mapped)

    # MDS+
    elif 'TDI' in mapped:
        try:
            TDI = mapped['TDI'].format(**options_with_defaults)
            treename = mapped['treename'].format(**options_with_defaults) if 'treename' in mapped else None
            data0 = data = OMFITmdsValue(server=machine, shot=pulse, treename=treename, TDI=TDI).data()
            if data is None:
                raise ValueError('data is None')
        except Exception:
            printe(mapped['TDI'].format(**options_with_defaults).replace('\\n', '\n'))
            raise

    else:
        raise ValueError(f"Could not fetch data for {location}. Must define one of {expression_types}")

    # handle size definition for array of structures
    if location.endswith(':'):
        if 'TDI' in mapped:
            data = data[0]
        return int(data), {'raw_data': data0, 'processed_data': data, 'cocosio': cocosio}

    # transpose manipulation
    if mapped.get('TRANSPOSE', False):
        data = numpy.transpose(data, mapped['TRANSPOSE'])

    # transpose filter
    nanfilter = lambda x: x
    if mapped.get('NANFILTER', False):
        nanfilter = lambda x: x[~numpy.isnan(x)]

    # cocos
    if mapped.get('COCOSIO', False):
        if 'VALUE' in mapped:
            cocosio = mapped['COCOSIO']
        else:
            TDI = mapped['COCOSIO'].format(**options_with_defaults)
            cocosio = int(OMFITmdsValue(server=machine, shot=pulse, treename=treename, TDI=TDI).data()[0])

    # assign data to ODS
    if not hasattr(data, 'shape'):
        ods[location] = data
    else:
        with omas_environment(ods, cocosio=cocosio):
            csize = mapped.get('COORDINATES', [])
            osize = len([c for c in mapped.get('COORDINATES', []) if c != '1...N'])
            dsize = len(data.shape)
            if dsize - osize == 0 or ':' not in location:
                if data.size == 1:
                    data = data.item()
                ods[location] = nanfilter(data)
            else:
                for k in range(data.shape[0]):
                    ods[u2n(location, [k] + [0] * 10)] = nanfilter(data[k, ...])

    return ods, {'raw_data': data0, 'processed_data': data, 'cocosio': cocosio}


def load_omas_machine(
        machine,
        pulse,
        options={},
        consistency_check=True,
        imas_version=omas_rcparams['default_imas_version'],
        cls=ODS,
        branch=None,
        user_machine_mappings=None,
):
    printd('Loading from %s' % machine, topic='machine')
    ods = cls(imas_version=imas_version, consistency_check=consistency_check)
    for location in [location for location in machine_mappings(machine, branch, user_machine_mappings) if not location.startswith('__')]:
        if location.endswith(':'):
            continue
        print(location)
        machine_to_omas(ods, machine, pulse, location, options, branch)
    return ods


class dynamic_omas_machine(dynamic_ODS):
    """
    Class that provides dynamic data loading from machine mappings
    This class is not to be used by itself, but via the ODS.open() method.
    """

    def __init__(self, machine, pulse, options={}, branch=None, user_machine_mappings=None, verbose=True):
        self.kw = {'machine': machine, 'pulse': pulse, 'options': options, 'branch': branch, 'user_machine_mappings': user_machine_mappings}
        self.active = False
        self.cache = {}

    def open(self):
        printd('Dynamic open  %s' % self.kw, topic='dynamic')
        self.active = True
        return self

    def close(self):
        printd('Dynamic close %s' % self.kw, topic='dynamic')
        self.active = False
        self.cache.clear()
        return self

    def __getitem__(self, key):
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        if o2u(key) not in self.cache:
            printd('Dynamic read  %s: %s' % (self.kw, key), topic='dynamic')
            ods, _ = machine_to_omas(ODS(), self.kw['machine'], self.kw['pulse'], o2u(key), self.kw['options'], self.kw['branch'])
            self.cache[o2u(key)] = ods
        if isinstance(self.cache[o2u(key)], int):
            return self.cache[o2u(key)]
        else:
            return self.cache[o2u(key)][key]

    def __contains__(self, location):
        ulocation = o2u(location)
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        if ulocation.endswith(':'):
            return False
        return ulocation in machine_mappings(self.kw['machine'], self.kw['branch'], self.kw['user_machine_mappings'])

    def keys(self, location):
        ulocation = o2u(location)
        if ulocation + '.:' in machine_mappings(self.kw['machine'], self.kw['branch'], self.kw['user_machine_mappings']):
            return list(range(self[ulocation + '.:']))
        else:
            return numpy.unique(
                [
                    convert_int(k[len(ulocation):].lstrip('.').split('.')[0])
                    for k in machine_mappings(self.kw['machine'], self.kw['branch'], self.kw['user_machine_mappings'])
                    if k.startswith(ulocation)
                ]
            )


_machines_dict = {}


def machines(machine=None, branch=None):
    '''
    Function to get machines that have their mappings defined

    :param machine: string with machine name or None

    :param branch: GitHub branch from which to load the machine mapping information

    :return: if `machine==None` returns dictionary with list of machines and their json mapping files
             if `machine` is a string, then returns json mapping filename and branch
    '''
    # return cached results
    if '__branch__' in _machines_dict and _machines_dict['__branch__'] == branch:
        if machine is None:
            return _machines_dict
        elif machine in _machines_dict:
            return _machines_dict[machine]

    # local machines
    for filename in glob.glob(imas_json_dir + '/../machine_mappings/*.json'):
        _machines_dict[os.path.splitext(os.path.split(filename)[1])[0]] = os.path.abspath(filename)

    # return list of supported machines
    if machine is None:
        if branch is None:
            return _machines_dict
        else:
            raise NotImplementedError(f'Cannot list machine mappings on GitHub branches')

    # return filename with mappings for this machine
    else:
        # try `master` branch if not in local
        if machine not in _machines_dict and branch is None:
            branch = 'master'

        # get remote mappings
        if branch is not None:
            try:
                # setup temporary machine_mappings directory
                dir = url_dir.format(branch=branch)
                if not os.path.exists(dir):
                    os.makedirs(dir)

                # download machine.json and write it to file
                url = f"https://raw.githubusercontent.com/gafusion/omas/{branch}/omas/machine_mappings/{machine}.json"
                contents = urllib.request.urlopen(url).read().decode("utf-8")
                filename = dir + os.sep + f'{machine}.json'
                with open(filename, 'w') as f:
                    f.write(contents)
                _machines_dict[machine] = filename
            except Exception as _excp:
                printd(f'No machine mappings for `{machine}` from branch `{branch}`: {repr(_excp)}', topic='machine')
                raise KeyError(f'No machine mapping for `{machine}`. Valid machines are: {list(_machines_dict.keys())}')
        return _machines_dict[machine], branch


_machine_mappings = {}
_namespace_mappings = {}
_user_machine_mappings = {}


def machine_mappings(machine, branch, user_machine_mappings=None, return_raw_mappings=False):
    '''
    Function to load the json mapping files (local or remote)
    Allows for merging external mapping rules defined by users.
    This function sanity-checks and the mapping file and adds extra info required for mapping

    :param machine: machine for which to load the mapping files

    :param branch: GitHub branch from which to load the machine mapping information

    :return: dictionary with mapping transformations
    '''
    if user_machine_mappings is None:
        user_machine_mappings = {}

    if (
            return_raw_mappings
            or machine not in _machine_mappings
            or list(_user_machine_mappings.keys()) + list(user_machine_mappings.keys())
            != _machine_mappings[machine]['__user_machine_mappings__']
    ):

        # figure out mapping file
        filename, branch = machines(machine, branch)

        # load mappings from file
        with open(filename, 'r') as f:
            mappings = json.load(f)
        for item in ['__cocos_rules__', '__options__']:
            mappings.setdefault(item, {})

        # merge mappings and user_machine_mappings
        mappings['__user_machine_mappings__'] = []
        for umap in [_user_machine_mappings, user_machine_mappings]:
            umap = copy.copy(umap)
            mappings['__user_machine_mappings__'].extend(list(umap.keys()))
            for item in ['__cocos_rules__', '__options__']:
                mappings[item].update(umap.pop(item, {}))
            mappings.update(umap)

        # return raw json mappings if so requested
        if return_raw_mappings:
            mappings.pop('__user_machine_mappings__')
            return mappings

        _namespace_mappings[machine] = {}
        if os.path.exists(os.path.splitext(filename)[0] + '.py'):
            with open(os.path.splitext(filename)[0] + '.py', 'r') as f:
                exec(f.read(), _namespace_mappings[machine])

        # generate TDI for cocos_rules
        for item in mappings['__cocos_rules__']:
            if 'eval2TDI' in mappings['__cocos_rules__'][item]:
                try:
                    mappings['__cocos_rules__'][item]['TDI'] = eval(
                        mappings['__cocos_rules__'][item]['eval2TDI'].replace('\\', '\\\\'), mapping_functions_namespace(branch)
                    )
                except:
                    print(mappings['__cocos_rules__'][item]['eval2TDI'])

        # generate TDI and sanity check mappings
        for location in mappings:
            # sanity check format
            if l2o(p2l(location)) != location:
                raise ValueError(f'{location} mapping should be specified as {l2o(p2l(location))}')

            # generate DTI functions based on eval2DTI
            if 'eval2TDI' in mappings[location]:
                mappings[location]['TDI'] = eval(mappings[location]['eval2TDI'].replace('\\', '\\\\'), mapping_functions_namespace(branch))

            # make sure required coordinates info are present in the mapping
            info = omas_info_node(location)
            if 'coordinates' in info:
                mappings[location]['COORDINATES'] = list(map(i2o, info['coordinates']))
                for coordinate in mappings[location]['COORDINATES']:
                    if coordinate == '1...N':
                        continue
                    elif coordinate not in mappings:
                        raise ValueError(f'missing coordinate {coordinate} for {location}')

            # add cocos transformation info
            has_COCOS = o2u(location) in cocos_signals and cocos_signals[o2u(location)] is not None
            if 'COCOSIO' not in mappings[location] and has_COCOS:
                cocos_defined = False
                for cocos in mappings['__cocos_rules__']:
                    if 'TDI' in mappings[location] and re.findall(cocos, mappings[location]['TDI']):
                        mappings[location]['COCOSIO'] = mappings['__cocos_rules__'][cocos]['TDI']
                        cocos_defined = True
                        break
                if not cocos_defined:
                    raise ValueError(f'{location} must have COCOS specified')
            if 'COCOSIO' in mappings[location] and not has_COCOS:
                raise ValueError(f'{location} should not have COCOS specified, or COCOS definition should be added to omas_cocos file')
        # cache
        _machine_mappings[machine] = mappings

    return _machine_mappings[machine]
