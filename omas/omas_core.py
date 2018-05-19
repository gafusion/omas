from __future__ import print_function, division, unicode_literals

from collections import MutableMapping

from .omas_utils import *

__version__ = open(os.path.abspath(str(os.path.dirname(__file__)) + os.sep + 'version'), 'r').read().strip()

__all__ = [
    'ODS', 'ods_sample', 'different_ods',
    'save_omas',      'load_omas',      'test_omas_suite',
    'save_omas_pkl',  'load_omas_pkl',  'test_omas_pkl',
    'save_omas_json', 'load_omas_json', 'test_omas_json',
    'save_omas_hdc',  'load_omas_hdc',  'test_omas_hdc',
    'save_omas_nc',   'load_omas_nc',   'test_omas_nc',
    'save_omas_imas', 'load_omas_imas', 'test_omas_imas',
    'save_omas_itm',  'load_omas_itm',  'test_omas_itm',
    'save_omas_s3',   'load_omas_s3',   'test_omas_s3', 'list_omas_s3', 'del_omas_s3',
    'omas_scenario_database',
    'generate_xml_schemas', 'create_json_structure', 'create_html_documentation',
    'imas_json_dir', 'default_imas_version', 'ids_cpo_mapper', 'omas_info',
    'cocos_environment', 'cocos_transform', 'define_cocos',
    'omas_rcparams', 'rcparams_environment', '__version__'
]


def _omas_key_dict_preprocessor(key):
    """
    converts a omas string path to a list of keys that make the path

    :param key: omas string path

    :return: list of keys that make the path
    """
    if not isinstance(key, (list, tuple)):
        key = str(key)
        key = re.sub('\]', '', re.sub('\[', '.', key)).split('.')
    else:
        key = list(map(str, key))
    for k,item in enumerate(key):
        try:
            key[k] = int(item)
        except ValueError:
            pass
    return key


class ODS(MutableMapping):
    """
    OMAS class
    """

    def __init__(self,
                 imas_version=default_imas_version,
                 consistency_check=omas_rcparams['consistency_check'],
                 dynamic_path_creation=omas_rcparams['dynamic_path_creation'],
                 location='',
                 cocos=omas_rcparams['cocos'],
                 cocosin=omas_rcparams['cocosin'],
                 cocosout=omas_rcparams['cocosout'],
                 structure=None):
        """
        :param imas_version: IMAS version to use as a constrain for the nodes names

        :param consistency_check: whether to enforce consistency with IMAS schema

        :param dynamic_path_creation: whether to dynamically create the path when setting an item

        :param location: string with location of this object relative to IMAS schema

        :param cocos: internal COCOS representation (this can only be set when the object is created)

        :param cocosin: COCOS representation of the data that is written to the ODS

        :param cocosout: COCOS representation of the data that is read from the ODS

        :param structure: IMAS schema to use
        """
        self.omas_data = None
        self._consistency_check = consistency_check
        self._dynamic_path_creation = dynamic_path_creation
        self.imas_version = imas_version
        self.location = location
        self._cocos = cocos
        self.cocosin = cocosin
        self.cocosout = cocosout
        if structure is None:
            structure = {}
        self.structure = structure

    @property
    def consistency_check(self):
        """
        property that sets whether consistency with IMAS schema is enabled or not

        :return: True/False
        """
        if not hasattr(self,'_consistency_check'):
            self._consistency_check=omas_rcparams['consistency_check']
        return self._consistency_check

    @consistency_check.setter
    def consistency_check(self, value):
        self._consistency_check = value
        for item in self.keys():
            if isinstance(self[item], ODS):
                self[item].consistency_check = value

    @property
    def cocos(self):
        """
        property that tells in what COCOS format the data is stored internally of the ODS
        (NOTE: this parameter can only be set when the object is created)

        :return: cocosin value
        """
        if not hasattr(self,'_cocos'):
            self._cocos=omas_rcparams['cocos']
        return self._cocos

    @cocos.setter
    def cocos(self, value):
        raise(AttributeError('cocos parameter is readonly!'))

    @property
    def cocosin(self):
        """
        property that tells in what COCOS format the data will be input

        :return: cocosin value
        """
        if not hasattr(self,'_cocosin'):
            self._cocosin=omas_rcparams['cocosin']
        return self._cocosin

    @cocosin.setter
    def cocosin(self, value):
        self._cocosin = value
        for item in self.keys():
            if isinstance(self[item], ODS):
                self[item].cocosin = value

    @property
    def cocosout(self):
        """
        property that tells in what COCOS format the data should be output

        :return: cocosout value
        """
        if not hasattr(self,'_cocosout'):
            self._cocosout=omas_rcparams['cocosout']
        return self._cocosout

    @cocosout.setter
    def cocosout(self, value):
        self._cocosout = value
        for item in self.keys():
            if isinstance(self[item], ODS):
                self[item].cocosout = value

    @property
    def dynamic_path_creation(self):
        """
        property that sets whether dynamic path creation is enabled or not

        :return: True/False
        """
        if not hasattr(self,'_dynamic_path_creation'):
            self._dynamic_path_creation=True
        return self._dynamic_path_creation

    @dynamic_path_creation.setter
    def dynamic_path_creation(self, value):
        self._dynamic_path_creation = value
        for item in self.keys():
            if isinstance(self[item], ODS):
                self[item].dynamic_path_creation = value

    def _validate(self, value, structure):
        """
        validate that the value is consistent with the provided structure field

        :param value: sub-tree to be checked

        :param structure: reference structure
        """
        for key in value.keys():
            structure_key = re.sub('^[0-9:]+$', ':', str(key))
            if isinstance(value[key], ODS) and value[key].consistency_check:
                value._validate(value[key], structure[structure_key])
            else:
                structure[structure_key]

    def __setitem__(self, key, value):
        # handle individual keys as well as full paths
        key = _omas_key_dict_preprocessor(key)

        # non-scalar data is saved as numpy arrays
        if isinstance(value, list):
            value = numpy.array(value)

        # if the user has entered path rather than a single key
        if len(key) > 1:
            pass_on_value = value
            value = ODS(imas_version=self.imas_version,
                        consistency_check=self.consistency_check,
                        dynamic_path_creation=self.dynamic_path_creation,
                        cocos=self.cocos, cocosin=self.cocosin, cocosout=self.cocosout)

        # full path where we want to place the data
        location = l2o([self.location, key[0]])

        # handle cocos transformations coming in
        if self.cocosin != self.cocos and location in omas_physics.cocos_signals:
            value = value * omas_physics.cocos_transform(self.cocosin, self.cocos)[omas_physics.cocos_signals[location]]

        # perform consistency check with IMAS structure
        if self.consistency_check:
            structure = {}
            structure_key = list(map(lambda x: re.sub('^[0-9:]+$', ':', str(x)), key))
            try:
                if isinstance(value, ODS):
                    if not self.structure:
                        # load the json structure file
                        structure = load_structure(key[0], imas_version=self.imas_version)[1][key[0]]
                    else:
                        structure = self.structure[structure_key[0]]
                        if not len(structure):
                            raise(ValueError('`%s` has no data'%location))
                    # check that tha data will go in the right place
                    self._validate(value, structure)
                else:
                    self.structure[structure_key[0]]

            except (LookupError, TypeError):
                if self.consistency_check=='warn':
                    printe('`%s` is not a valid IMAS %s location' % (location, self.imas_version))
                    if isinstance(value,ODS):
                        value.consistency_check=False
                elif self.consistency_check:
                    options = list(self.structure.keys())
                    if len(options) == 1 and options[0] == ':':
                        options = 'A numerical index is needed with n>=0'
                    else:
                        options = 'Did you mean: %s' % options
                    spaces = ' '*len('LookupError')+'  '+' ' * (len(self.location) + 2)
                    raise LookupError('`%s` is not a valid IMAS %s location\n' % (location, self.imas_version) +
                        spaces + '^' * len(structure_key[0]) + '\n' + '%s' % options)

        # check what container type is required and if necessary switch it
        if isinstance(key[0], int) and not isinstance(self.omas_data, list):
            if not self.omas_data or not len(self.omas_data):
                self.omas_data = []
            else:
                raise (Exception('Cannot convert from dict to list once ODS has data'))
        if not isinstance(key[0], int) and not isinstance(self.omas_data, dict):
            if not self.omas_data or not len(self.omas_data):
                self.omas_data = {}
            else:
                raise (Exception('Cannot convert from list to dict once ODS has data'))

        # now that all checks are completed we can assign the structure information
        if self.consistency_check:
            if isinstance(value, ODS):
                value.structure = structure

        if isinstance(value, ODS):
            value.location = location

        # if the user has entered a path rather than a single key
        if len(key) > 1:
            dynamically_created = False
            if key[0] not in self.keys():
                dynamically_created = True
                if isinstance(self.omas_data, dict):
                    self.omas_data[key[0]] = value
                elif key[0] == len(self.omas_data):
                    self.omas_data.append(value)
                else:
                    raise (IndexError('%s[:] index is at %d' % (self.location, len(self) - 1)))
            try:
                self[key[0]][l2o(key[1:])] = pass_on_value
            except LookupError:
                if dynamically_created:
                    del self[key[0]]
                raise
        elif isinstance(self.omas_data, dict):
            self.omas_data[key[0]] = value
        elif key[0] in self.omas_data:
            self.omas_data[key[0]] = value
        elif key[0] == len(self.omas_data):
            self.omas_data.append(value)
        else:
            raise IndexError('%s[:] index is at %d' % (self.location, len(self.omas_data) - 1))

    def __getitem__(self, key):
        # handle individual keys as well as full paths
        key = _omas_key_dict_preprocessor(key)

        dynamically_created = False

        # data slicing
        if key[0] == ':':
            data = []
            for k in self.keys():
                data.append(self[l2o([k] + key[1:])])
            return numpy.array(data)

        # dynamic path creation
        elif key[0] not in self.keys():
            if self.dynamic_path_creation:
                dynamically_created=True
                self.__setitem__(key[0], ODS(imas_version=self.imas_version,
                                              consistency_check=self.consistency_check,
                                              dynamic_path_creation=self.dynamic_path_creation,
                                              cocos=self.cocos, cocosin=self.cocosin, cocosout=self.cocosout))
            else:
                location = l2o([self.location, key[0]])
                raise(LookupError('Dynamic path creation is disabled, hence `%s` needs to be manually created'%location))

        if len(key) > 1:
            # if the user has entered path rather than a single key
            try:
                return self.omas_data[key[0]][l2o(key[1:])]
            except ValueError:
                if dynamically_created:
                    del self[key[0]]
                raise
        else:
            location = l2o([self.location, key[0]])
            value=self.omas_data[key[0]]
            # handle cocos transformations going out
            if self.cocosout != self.cocos and location in omas_physics.cocos_signals:
                value = value * omas_physics.cocos_transform(self.cocos, self.cocosout)[omas_physics.cocos_signals[location]]
            return value

    def __delitem__(self, key):
        # handle individual keys as well as full paths
        key = _omas_key_dict_preprocessor(key)
        if len(key) > 1:
            # if the user has entered path rather than a single key
            del self[key[0]][l2o(key[1:])]
        else:
            return self.omas_data.__delitem__(key[0])

    def paths(self, **kw):
        """
        Traverse the ods and return paths that have data

        :return: list of paths that have data
        """
        paths = kw.setdefault('paths', [])
        path = kw.setdefault('path', [])
        for kid in self.keys():
            if isinstance(self[kid], ODS):
                self[kid].paths(paths=paths, path=path + [kid])
            else:
                paths.append(path + [kid])
        return paths

    def flat(self):
        """
        :return: flat dictionary representation of the data
        """
        tmp = OrderedDict()
        for path in self.paths():
            tmp[l2o(path)] = self[path]
        return tmp

    def __getnewargs__(self):
        # tells pickle.dumps to pickle the omas object in such a way that a pickle.loads
        # back from that string will use omas.__new__ with consistency_check=False and dynamic_path_creation=True
        return (False,True)

    def __len__(self):
        return len(self.omas_data)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        key = _omas_key_dict_preprocessor(key)
        h = self
        for k in key:
            # h.omas_data is None when dict/list behaviour is not assigned
            if h.omas_data is not None and k in h.keys():
                h = h[k]
                continue  # continue to the next key
            else:
                return False
        # return False if checking existance of a leaf and the leaf exists but is unassigned
        if isinstance(h, ODS) and h.omas_data is None:
            return False
        return True

    def keys(self):
        if isinstance(self.omas_data, dict):
            # map keys with str to get strings and not unicode when working with Python 2.7
            return list(map(str,self.omas_data.keys()))
        elif isinstance(self.omas_data, list):
            return range(len(self.omas_data))
        else:
            return []

    def values(self):
        return [self[item] for item in self.keys()]

    def __str__(self):
        return str(self.omas_data)

    def __repr__(self):
        return repr(self.omas_data)

    def get(self, key, default=None):
        """
        Check if key is present and if not return default value without creating value in omas data structure

        :param key: key to get

        :param default: default value

        :return: return default if key is not found
        """
        if key not in self:
            return default
        else:
            return self[key]

    def setdefault(self, key, value=None):
        """
        Set value if key is not present

        :param key: key to get

        :param value: value to set

        :return: value
        """
        if key not in self:
            self[key] = value
        return self[key]

    def __getstate__(self):
        tmp=copy.copy(self.__dict__)
        for item in list(tmp.keys()):
            if item not in omas_dictstate:
                del tmp[item]
        return tmp

    def copy(self):
        '''
        :return: copy.deepcopy of current ODS object
        '''
        return copy.deepcopy(self)

    def clear(self):
        '''
        remove data from a branch

        :return: current ODS object
        '''
        if isinstance(self.omas_data, dict):
            self.omas_data.clear()
        elif isinstance(self.omas_data, list):
            self.omas_data[:]=[]
        return self

    def copy_attrs_from(self, ods):
        '''
        copy omas_ods_attrs ['_consistency_check','_dynamic_path_creation','imas_version','location','structure','_cocos','_cocosin','_cocosout'] attributes from input ods

        :param ods: input ods

        :return: self
        '''
        for item in omas_ods_attrs:
            setattr(self,item,getattr(ods,item))
        return self

    def prune(self):
        '''
        Prune ODS branches that are leafless

        :return: number of branches that were pruned
        '''
        n=0
        for item in self.keys():
            if isinstance(self[item], ODS):
                n+=self[item].prune()
                if not len(self[item].keys()):
                    n+=1
                    del self[item]
        return n
# --------------------------------------------
# import physics functions and add them as ODS methods
# --------------------------------------------
try:
    from . import omas_physics
    from .omas_physics import cocos_environment, cocos_transform, define_cocos
    __all__.append('omas_physics')
    for item in omas_physics.__all__:
        setattr(ODS, 'physics_' + item, getattr(omas_physics, item))
except ImportError as _excp:
    printe('OMAS physics function are not available: ' + repr(_excp))


# --------------------------------------------
# import plotting functions and add them as ODS methods
# --------------------------------------------
try:
    from . import omas_plot
    __all__.append('omas_plot')
    for item in omas_plot.__all__:
        setattr(ODS, 'plot_' + item, getattr(omas_plot, item))
except ImportError as _excp:
    printe('OMAS plotting function are not available: ' + repr(_excp))

omas_ods_attrs=['_consistency_check','_dynamic_path_creation','imas_version','location','structure','_cocos','_cocosin','_cocosout']
omas_dictstate=dir(ODS)
omas_dictstate.extend(['omas_data']+omas_ods_attrs)
omas_dictstate=sorted(list(set(omas_dictstate)))

# --------------------------------------------
# save and load OMAS with Python pickle
# --------------------------------------------
def save_omas_pkl(ods, filename, **kw):
    """
    Save OMAS data set to Python pickle

    :param ods: OMAS data set

    :param filename: filename to save to

    :param kw: keywords passed to pickle.dump function
    """
    printd('Saving to %s' % filename, topic='pkl')

    kw.setdefault('protocol', pickle.HIGHEST_PROTOCOL)

    with open(filename, 'wb') as f:
        pickle.dump(ods, f, **kw)


def load_omas_pkl(filename):
    """
    Load OMAS data set from Python pickle

    :param filename: filename to save to

    :returns: ods OMAS data set
    """
    printd('Loading from %s' % filename, topic='pkl')

    with open(filename, 'rb') as f:
        return pickle.load(f)


def test_omas_pkl(ods):
    """
    test save and load Python pickle

    :param ods: ods

    :return: ods
    """
    filename = 'test.pkl'
    save_omas_pkl(ods, filename)
    ods1 = load_omas_pkl(filename)
    return ods1


# --------------------------------------------
# tools
# --------------------------------------------
def ods_sample():
    """
    create sample ODS data
    """
    ods = ODS()

    #check effect of disabling dynamic path creation
    try:
        ods.dynamic_path_creation = False
        ods['info.user']
    except LookupError:
        ods['info'] = ODS()
        ods['info.user'] = unicode(os.environ['USER'])
    else:
        raise(Exception('OMAS error handling dynamic_path_creation=False'))
    finally:
        ods.dynamic_path_creation = True

    #check that accessing leaf that has not been set raises a ValueError, even with dynamic path creation turned on
    try:
        ods['info.machine']
    except ValueError:
        pass
    else:
        raise(Exception('OMAS error querying leaf that has not been set'))

    # info ODS is used for keeping track of IMAS metadata
    ods['info.machine'] = 'ITER'
    ods['info.imas_version'] = default_imas_version
    ods['info.shot'] = 1
    ods['info.run'] = 0

    # check .get() method
    assert (ods.get('info.shot') == ods['info.shot'])
    assert (ods.get('info.bad', None) is None)

    # check that keys is an iterable (so that Python 2/3 work the same way)
    keys = ods.keys()
    keys[0]

    # check that dynamic path creation during __getitem__ does not leave empty fields behind
    try:
        print(ods['wall.description_2d.0.limiter.unit.0.outline.r'])
    except ValueError:
        assert 'wall.description_2d.0.limiter.unit.0.outline' not in ods

    ods['equilibrium']['time_slice'][0]['time'] = 1000.
    ods['equilibrium']['time_slice'][0]['global_quantities']['ip'] = 1.5

    ods2 = copy.deepcopy(ods)
    ods2['equilibrium']['time_slice'][1] = ods['equilibrium']['time_slice'][0]
    ods2['equilibrium']['time_slice'][2] = ods['equilibrium']['time_slice'][0]

    printd(ods2['equilibrium']['time_slice'][0]['global_quantities'].location, topic='sample')
    printd(ods2['equilibrium']['time_slice'][2]['global_quantities'].location, topic='sample')

    ods2['equilibrium.time_slice.1.time'] = 2000.
    ods2['equilibrium.time_slice.1.global_quantities.ip'] = 2.
    ods2['equilibrium.time_slice[2].time'] = 3000.

    # uncertain scalar
    ods2['equilibrium.time_slice[2].global_quantities.ip'] = ufloat(3,0.1)

    # uncertain array
    ods2['equilibrium.time_slice[2].profiles_1d.q'] = uarray([0.,1.,2.,3.],[0,.1,.2,.3])

    # check different ways of addressing data
    printd(ods2['equilibrium.time_slice']['1.global_quantities.ip'], topic='sample')
    printd(ods2[['equilibrium', 'time_slice', 1, 'global_quantities', 'ip']], topic='sample')
    printd(ods2[('equilibrium', 'time_slice', '1', 'global_quantities', 'ip')], topic='sample')
    printd(ods2['equilibrium.time_slice.1.global_quantities.ip'], topic='sample')
    printd(ods2['equilibrium.time_slice[1].global_quantities.ip'], topic='sample')

    ods2['equilibrium.time_slice.0.profiles_1d.psi'] = numpy.linspace(0, 1, 10)

    # pprint(ods.paths())
    # pprint(ods2.paths())

    # check data slicing
    printd(ods2['equilibrium.time_slice[:].global_quantities.ip'], topic='sample')

    ckbkp = ods.consistency_check
    tmp = pickle.dumps(ods2)
    ods2 = pickle.loads(tmp)
    if ods2.consistency_check != ckbkp:
        raise (Exception('consistency_check attribute changed'))

    # check picking
    save_omas_pkl(ods2, 'test.pkl')
    ods2 = load_omas_pkl('test.pkl')

    # check flattening
    tmp = ods2.flat()
    # pprint(tmp)

    # check deepcopy
    ods3=ods2.copy()

    return ods3


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
    for k in k1.difference(k2):
        if not k.startswith('info.'):
            return 'DIFF: key `%s` missing in 2nd ods' % k
    for k in k2.difference(k1):
        if not k.startswith('info.'):
            return 'DIFF: key `%s` missing in 1st ods' % k
    for k in k1.intersection(k2):
        if isinstance(ods1[k], basestring) and isinstance(ods2[k], basestring):
            if ods1[k] != ods2[k]:
                return 'DIFF: `%s` differ in value' % k
        elif type(ods1[k]) != type(ods2[k]):
            return 'DIFF: `%s` differ in type (%s,%s)' % (k, type(ods1[k]), type(ods2[k]))
        elif numpy.atleast_1d(is_uncertain(ods1[k])).any() or numpy.atleast_1d(is_uncertain(ods2[k])).any():
            if not numpy.allclose(nominal_values(ods1[k]), nominal_values(ods2[k])) and not numpy.allclose(std_devs(ods1[k]), std_devs(ods2[k])):
                return 'DIFF: `%s` differ in value' % k
        else:
            if not numpy.allclose(ods1[k], ods2[k]):
                return 'DIFF: `%s` differ in value' % k
    return False


_tests = ['pkl', 'json', 'nc', 's3', 'imas', 'hdc']


def test_omas_suite(ods=None, test_type=None, do_raise=False):
    """
    :param ods: omas structure to test. If None this is set to ods_sample

    :param test_type: None tests all suite, otherwise choose among %s

    :param do_raise: raise error if something goes wrong
    """

    if ods is None:
        ods = ods_sample()

    if test_type in _tests:
        os.environ['OMAS_DEBUG_TOPIC'] = test_type
        ods1 = globals()['test_omas_' + test_type](ods)
        check = different_ods(ods, ods1)
        if not check:
            print('OMAS data got saved and loaded correctly')
        else:
            print(check)

    else:

        os.environ['OMAS_DEBUG_TOPIC'] = '*'
        printd('OMAS is using IMAS data structure version `%s` as default' % default_imas_version, topic='*')

        print('=' * 20)

        results = numpy.zeros((len(_tests), len(_tests)))

        for k1, t1 in enumerate(_tests):
            failed1 = False
            try:
                ods1 = globals()['test_omas_' + t1](ods)
            except Exception as _excp:
                failed1 = _excp
                if do_raise:
                    raise
            for k2, t2 in enumerate(_tests):
                try:
                    if failed1:
                        raise failed1
                    ods2 = globals()['test_omas_' + t2](ods1)

                    different = different_ods(ods1, ods2)
                    if not different:
                        print('FROM %s TO %s : OK' % (t1.center(5), t2.center(5)))
                        results[k1, k2] = 1.0
                    else:
                        print('FROM %s TO %s : NO --> %s' %
                              (t1.center(5), t2.center(5), different))
                        results[k1, k2] = -1.0

                except Exception as _excp:
                    print('FROM %s TO %s : NO --> %s' %
                          (t1.center(5), t2.center(5), repr(_excp)))
                    if do_raise:
                        raise
        print('=' * 20)
        print(results.astype(int))
        print('=' * 20)


test_omas_suite.__doc__ = test_omas_suite.__doc__ % _tests


# --------------------------------------------
# save and load OMAS with default saving method
# --------------------------------------------
def save_omas(ods, filename):
    """
    Save omas data to filename. The file extension defines format to use.

    :param ods: OMAS data set

    :param filename: filename to save to
    """
    if os.path.splitext(filename)[1].lower() == '.json':
        return save_omas_json(ods, filename)
    elif os.path.splitext(filename)[1].lower() == '.nc':
        return save_omas_nc(ods, filename)
    else:
        return save_omas_pkl(ods, filename)


def load_omas(filename):
    """
    Load omas data from filename. The file extension defines format to use.

    :param filename: filename to load from

    :returns: ods OMAS data set
    """
    if os.path.splitext(filename)[1].lower() == '.json':
        return load_omas_json(filename)
    elif os.path.splitext(filename)[1].lower() == '.nc':
        return load_omas_nc(filename)
    else:
        return load_omas_pkl(filename)


# --------------------------------------------
# import other omas tools and methods in this namespace
# --------------------------------------------
from .omas_imas import *
from .omas_s3 import *
from .omas_nc import *
from .omas_json import *
from .omas_structure import *
from .omas_itm import *
from .omas_hdc import *
