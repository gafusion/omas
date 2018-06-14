from __future__ import print_function, division, unicode_literals

from collections import MutableMapping

from .omas_utils import *

__version__ = open(os.path.abspath(str(os.path.dirname(__file__)) + os.sep + 'version'), 'r').read().strip()

__all__ = [
    'ODS', 'ods_sample', 'different_ods',
    'save_omas_pkl',  'load_omas_pkl',  'through_omas_pkl',
    'save_omas_json', 'load_omas_json', 'through_omas_json',
    'save_omas_hdc',  'load_omas_hdc',  'through_omas_hdc',
    'save_omas_nc',   'load_omas_nc',   'through_omas_nc',
    'save_omas_imas', 'load_omas_imas', 'through_omas_imas',
    'save_omas_itm',  'load_omas_itm',  'through_omas_itm',
    'save_omas_s3',   'load_omas_s3',   'through_omas_s3', 'list_omas_s3', 'del_omas_s3',
    'generate_xml_schemas', 'create_json_structure', 'create_html_documentation',
    'imas_json_dir', 'default_imas_version', 'ids_cpo_mapper', 'omas_info', 'omas_info_node',
    'cocos_environment', 'cocos_transform', 'define_cocos',
    'omas_rcparams', 'rcparams_environment', '__version__'
]

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

    def homogeneous_time(self, key=''):
        '''
        return whether time is homogeneous or not

        :param key: ods location

        :return:
        '''
        extra_info={}
        self.time(key=key,extra_info=extra_info)
        return extra_info['homogeneous_time']

    def time(self, key='', extra_info=None):
        """
        Return the time information for a given ODS location

        :param key: ods location

        :param extra_info: dictionary that will be filled in place with extra information about time

        :return: time information for a given ODS location (scalar or array)
        """

        def add_is_homogeneous_info(time):
            if time is None:
                extra_info['homogeneous_time'] = None
            elif len(numpy.atleast_1d(time)) <= 2:
                extra_info['homogeneous_time'] = None
            else:
                tmp = numpy.diff(time)
                if numpy.sum(numpy.abs(tmp - tmp[0])) < 1E-6:
                    extra_info['homogeneous_time'] = True
                else:
                    extra_info['homogeneous_time'] = False
            return time

        # extra
        if extra_info is None:
            extra_info = {}

        # process the key
        key = p2l(key)
        tmp = self[key]
        if not isinstance(tmp, ODS):
            key = key[:-1]
            tmp = self[key]

        # this ODS has a children with 'time' information
        if isinstance(tmp.omas_data, dict):
            if 'time' in tmp:
                extra_info['location'] = self.location + separator + 'time'
                return add_is_homogeneous_info(tmp['time'])
            # this node should have time filled, but the user did not do their job
            elif 'time' in tmp.structure:
                # try to assemble time information by looking in the children
                for item in tmp.structure:
                    if item in tmp and ':' in tmp.structure[item] and 'time' in tmp.structure[item][':']:
                        return add_is_homogeneous_info(tmp.time(item, extra_info=extra_info))

        # this ODS is an array of structures (which may or may not have time information)
        elif isinstance(tmp.omas_data, list):
            # assemble time array information from the children
            times = []
            for item in tmp:
                times.append(tmp[item].time(extra_info=extra_info))
            # if any time information was found, return it
            if len(list(filter(None, times))):
                return add_is_homogeneous_info(numpy.array(times))

        # ODS not yet assigned
        else:
            return add_is_homogeneous_info(None)

        # traverse tree upstream looking for the first parent that has time information
        while len(key):
            key.pop()
            time = self.time(key, extra_info=extra_info)
            if time is not None:
                # if the parent with time information is an array of structures
                # then return the time of the element that we are asking for
                if isinstance(time, numpy.ndarray):
                    time = time[key[-1]]
                return add_is_homogeneous_info(time)

    def slice_at_time(self, time=None, time_index=None):
        '''
        method for selecting a time slice from an time-dependent ODS (NOTE: this method operates in place)

        :param time: time value to select

        :param time_index: time index to select (NOTE: time_index has precedence over time)

        :return: modified ODS
        '''

        # set time_index for parent and children
        if 'time' in self:
            if time_index is None:
                time_index = numpy.where(self['time'] == time)[0]
            else:
                raise (ValueError(
                    'time info is defined both in %s as well as upstream' % (self.location + separator + 'time')))

        # loop over items
        for item in self.keys():
            # time (if present) is treated last
            if item == 'time':
                continue

            # identify time-dependent data
            info = omas_info_node(self.location + separator + item)
            if 'coordinates' in info and any([k.endswith(separator + 'time') for k in info['coordinates']]):

                # time-dependent arrays
                if not isinstance(self[item], ODS):
                    self[item] = numpy.atleast_1d(self[item][time_index])

                # time-depentend list of ODSs
                elif isinstance(self[item].omas_data, list) and len(self[item]) and 'time' in self[item][0]:
                    for k in self[item].keys()[::-1]:
                        if k != time_index:
                            del self[item][k]

            # go deeper inside ODSs that do not have time info
            elif isinstance(self[item], ODS):
                self[item].slice_at_time(time=time, time_index=time_index)

        # treat time
        if 'time' in self:
            self['time'] = numpy.atleast_1d(self['time'][time_index])

        return self

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
        key = p2l(key)

        # non-scalar data is saved as numpy arrays
        if isinstance(value, list):
            value = numpy.array(value)
        # floats as python floats
        elif isinstance(value, numpy.float64):
            value = float(value)

        # if the user has entered path rather than a single key
        if len(key) > 1:
            pass_on_value = value
            value = ODS(imas_version=self.imas_version,
                        consistency_check=self.consistency_check,
                        dynamic_path_creation=self.dynamic_path_creation,
                        cocos=self.cocos, cocosin=self.cocosin, cocosout=self.cocosout)

        # full path where we want to place the data
        location = l2u([self.location, key[0]])

        # handle cocos transformations coming in
        if self.cocosin != self.cocos and separator in location and location in omas_physics.cocos_signals:
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
            else:
                info = omas_info_node(location)
                # check consistency for scalar entries
                if 'data_type' in info and '_0D' in info['data_type'] and isinstance(value, numpy.ndarray):
                    printe('%s must be a scalar of type %s' % (location, info['data_type']))
                # check consistency for number of dimensions
                elif 'coordinates' in info and len(info['coordinates']):
                    if not isinstance(value, numpy.ndarray) or len(value.shape) != len(info['coordinates']):
                        # may want to raise a ValueError in the future
                        printe('%s must be an array with dimensions: %s' % (location, info['coordinates']))

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
        key = p2l(key)

        if not len(key):
            return self

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
                location = l2u([self.location, key[0]])
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
            location = l2u([self.location, key[0]])
            value=self.omas_data[key[0]]
            # handle cocos transformations going out
            if self.cocosout != self.cocos and location in omas_physics.cocos_signals:
                value = value * omas_physics.cocos_transform(self.cocos, self.cocosout)[omas_physics.cocos_signals[location]]
            return value

    def __delitem__(self, key):
        # handle individual keys as well as full paths
        key = p2l(key)
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
        key = p2l(key)
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

        :param key: ods location

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

        :param key: ods location

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

    def set_time_array(self, key, time_index, value):
        '''
        Convenience function for setting time dependent arrays

        :param key: ods location to edit

        :param time_index: time index of the value to set

        :param value: value to set

        :return: time dependent array
        '''

        key = p2l(key)

        orig_value = []
        if key in self:
            orig_value = numpy.atleast_1d(self[key]).tolist()

        # substitute
        if time_index < len(orig_value):
            orig_value[time_index] = value
        # append
        elif time_index == len(orig_value):
            orig_value = orig_value + [value]
        else:
            raise (IndexError('%s has length and time_index %d is bejond range' % (l2o(key), len(orig_value), time_index)))

        self[key] = numpy.atleast_1d(orig_value)
        return orig_value

# --------------------------------------------
# import sample functions and add them as ODS methods
# --------------------------------------------
try:
    from . import omas_sample
    from .omas_sample import ods_sample
    __all__.append('omas_sample')
    for item in omas_sample.__all__:
        setattr(ODS, 'sample_' + item, getattr(omas_sample, item))
except ImportError as _excp:
    printe('OMAS sample function are not available: ' + repr(_excp))

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


def through_omas_pkl(ods):
    """
    test save and load Python pickle

    :param ods: ods

    :return: ods
    """
    if not os.path.exists(tempfile.gettempdir()+'/OMAS_TESTS/'):
        os.makedirs(tempfile.gettempdir()+'/OMAS_TESTS/')
    filename = tempfile.gettempdir()+'/OMAS_TESTS/test.pkl'
    save_omas_pkl(ods, filename)
    ods1 = load_omas_pkl(filename)
    return ods1


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
