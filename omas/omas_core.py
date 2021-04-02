'''ODS class and save/load from pickle routines

-------
'''

from .omas_utils import *
from .omas_utils import __version__, _extra_structures

# fmt: off
__all__ = [
    'ODS', 'ODC', 'ODX',
    'CodeParameters', 'codeparams_xml_save', 'codeparams_xml_load',
    'ods_sample', 'different_ods', 'omas_structure',
    'save_omas_pkl', 'load_omas_pkl', 'through_omas_pkl',
    'save_omas_json', 'load_omas_json', 'through_omas_json',
    'save_omas_mongo', 'load_omas_mongo', 'through_omas_mongo',
    'save_omas_hdc', 'load_omas_hdc', 'through_omas_hdc',
    'save_omas_nc', 'load_omas_nc', 'through_omas_nc',
    'save_omas_h5', 'load_omas_h5', 'through_omas_h5',
    'save_omas_ascii', 'load_omas_ascii', 'through_omas_ascii',
    'save_omas_ds', 'load_omas_ds', 'through_omas_ds',
    'load_omas_dx', 'save_omas_dx', 'through_omas_dx', 'ods_2_odx', 'odx_2_ods',
    'save_omas_imas', 'load_omas_imas', 'through_omas_imas', 'load_omas_iter_scenario', 'browse_imas',
    'save_omas_s3', 'load_omas_s3', 'through_omas_s3', 'list_omas_s3', 'del_omas_s3',
    'machine_expression_types', 'machines', 'machine_mappings', 'load_omas_machine',
    'machine_mapping_function', 'run_machine_mapping_functions', 'mdstree', 'mdsvalue',
    'imas_json_dir', 'imas_versions', 'latest_imas_version', 'omas_info', 'omas_info_node', 'get_actor_io_ids',
    'omas_rcparams', 'rcparams_environment', 'omas_testdir', '__version__',
    'latexit'
]
# fmt: on

# List of functions that can be added by third-party Python
# packages for processing input data that will go in an ODS
# This is necessary because ODSs should only contain [int (arrays), floats (arrays), strings]
# It is used for example by OMFIT to process OMFITexpressions
input_data_process_functions = []


def force_imas_type(value):
    """
    IMAS supports (arrays of) integers, floats and strings

    :param value: input value

    :return: input value converted to be IMAS compatible
    """
    # lists are saved as numpy arrays, and 0D numpy arrays as scalars
    for function in input_data_process_functions:
        value = function(value)
    if isinstance(value, list):
        value = numpy.array(value)
    if 'DataArray' in value.__class__.__name__:
        import xarray

        if isinstance(value, xarray.DataArray):
            value = value.values
    if isinstance(value, numpy.ndarray) and not len(value.shape):
        value = value.item()
    if isinstance(value, (numpy.string_, numpy.unicode_, numpy.str_)):
        value = value.item()
    elif isinstance(value, (float, numpy.floating)):
        value = float(value)
    elif isinstance(value, (int, numpy.integer)):
        value = int(value)
    if isinstance(value, str):
        pass
    elif isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    return value


_consistency_warnings = {}


def consistency_checker(location, value, info, consistency_check, imas_version):
    """
    Print warnings or raise errors if object does not satisfy IMAS data dictionary
    Converts numeric data to INT/FLOAT depending on IMAS specifications

    :param value: value to check consistency of

    :param info: output of omas_info_node

    :param consistency_check: True, False, 'warn'

    :param imas_version: IMAS version

    :return: value
    """
    # force type consistent with data dictionary
    txt = ''
    if is_uncertain(value) or 'data_type' not in info:
        pass
    elif isinstance(value, numpy.ndarray):
        if 'STRUCT_ARRAY' in info['data_type'] and not len(value):
            value = ODS()
            value.omas_data = []
        elif 'FLT' in info['data_type']:
            value = value.astype(float)
        elif 'INT' in info['data_type']:
            value = value.astype(int)
        elif 'STR' in info['data_type']:
            value = value.astype(str)
    elif isinstance(value, (int, float, numpy.integer, numpy.floating)):
        if 'FLT' in info['data_type']:
            value = float(value)
        elif 'INT' in info['data_type']:
            value = int(value)
        elif 'STR' in info['data_type']:
            value = str(value)
    elif isinstance(value, bytes):
        if 'STR' in info['data_type']:
            value = b2s(value)

    # structure type is respected check type
    if 'data_type' in info and info['data_type'] in ['STRUCTURE', 'STRUCT_ARRAY'] and not isinstance(value, ODS):
        txt = f'{location} is of type {type(value)} but this should be an ODS'
    # check type
    elif not (
        isinstance(value, (int, float, str, numpy.ndarray, uncertainties.core.Variable))
        or value is None
        or isinstance(value, (CodeParameters, ODS))
    ):
        txt = f'{location} is of type {type(value)} but supported types are: string, float, int, array'
    # check consistency for scalar entries
    elif 'data_type' in info and '_0D' in info['data_type'] and isinstance(value, numpy.ndarray):
        txt = f'{location} is of type {type(value)} must be a scalar of type {info["data_type"]}'
    # check consistency for number of dimensions
    elif (
        'coordinates' in info
        and len(info['coordinates'])
        and (not isinstance(value, numpy.ndarray) or len(value.shape) != len(info['coordinates']))
    ):
        txt = f'{location} shape {numpy.asarray(value).shape} is inconsistent with coordinates: {info["coordinates"]}'

    if len(txt) and consistency_check is True:
        raise ValueError(txt)
    elif 'lifecycle_status' in info and info['lifecycle_status'] in ['obsolescent']:
        txt = f'{o2u(location)} is in {info["lifecycle_status"].upper()} state for IMAS {imas_version}'
        if consistency_check and imas_version not in _consistency_warnings or txt not in _consistency_warnings[imas_version]:
            _consistency_warnings.setdefault(imas_version, []).append(txt)
        else:
            txt = ''

    return value, txt


def _handle_extension(*args, **kw):
    if args[0] == 'ascii':
        ext = 'ascii'
        args = list(args)
        args[0] = None
        args = tuple(args)
    elif '/' not in args[0] and '.' not in os.path.split(args[0])[1]:
        ext = args[0]
        args = args[1:]
    else:
        ext = os.path.splitext(args[0])[-1].strip('.')
        if not ext:
            ext = 'pkl'
    # use `ids` extension for IMAS ASCII format
    if ext == 'ids':
        ext = 'ascii'
    # understand machine names as `machine` extension
    if ext in machines(None):
        args = tuple([ext] + list(args))
        ext = 'machine'
    return ext, args


omas_ods_attrs = ['_consistency_check', '_imas_version', '_cocos', '_cocosio', '_coordsio', '_unitsio', '_dynamic', '_parent']


class ODS(MutableMapping):
    """
    OMAS Data Structure class
    """

    def __init__(
        self,
        imas_version=omas_rcparams['default_imas_version'],
        consistency_check=omas_rcparams['consistency_check'],
        cocos=omas_rcparams['cocos'],
        cocosio=None,
        coordsio=None,
        unitsio=None,
        dynamic=None,
    ):
        """
        :param imas_version: IMAS version to use as a constrain for the nodes names

        :param consistency_check: whether to enforce consistency with IMAS schema

        :param cocos: internal COCOS representation (this can only be set when the object is created)

        :param cocosio: COCOS representation of the data that is read/written from/to the ODS

        :param coordsio: ODS with coordinates to use for the data that is read/written from/to the ODS

        :param unitsio: ODS will return data with units if True

        :param dynamic: internal keyword used for dynamic data loading
        """
        self.omas_data = None
        self._consistency_check = consistency_check
        if consistency_check and imas_version not in imas_versions:
            raise ValueError("Unrecognized IMAS version `%s`. Possible options are:\n%s" % (imas_version, imas_versions.keys()))
        self._parent = None
        self.imas_version = imas_version
        self.cocos = cocos
        self.cocosio = cocosio
        self.coordsio = coordsio
        self.unitsio = unitsio
        self.dynamic = dynamic

    def homogeneous_time(self, key='', default=True):
        """
        Dynamically evaluate whether time is homogeneous or not
        NOTE: this method does not read ods['ids_properties.homogeneous_time'] instead it uses the time info to figure it out

        :param default: what to return in case no time basis is defined

        :return: True/False or default value (True) if no time basis is defined
        """
        if not len(self.location) and not len(key):
            raise ValueError('homogeneous_time() can not be called on a top-level ODS')
        extra_info = {}
        self.time(key=key, extra_info=extra_info)
        homogeneous_time = extra_info['homogeneous_time']
        if homogeneous_time is None:
            return default
        else:
            return homogeneous_time

    def time(self, key='', extra_info=None):
        """
        Return the time information for a given ODS location

        :param key: ods location

        :param extra_info: dictionary that will be filled in place with extra information about time

        :return: time information for a given ODS location (scalar or array)
        """

        if extra_info is None:
            extra_info = {}

        location = self.location

        # subselect on requested key
        subtree = p2l(location + '.' + key)

        # get time nodes from data structure definitions
        loc = p2l(subtree)
        if not loc:
            raise LookupError('Must specify a location in the ODS to get the time of')
        utimes_ds = [i2o(k) for k in omas_times(self.imas_version) if k.startswith(loc[0] + '.')]

        # get time nodes with actual numbers for indexes of arrays of structures and identify time index
        times_ds = list(map(lambda item: u2o(item, l2o(subtree)), utimes_ds))
        try:
            time_array_index = int(re.sub('.*\.([0-9]+)\.time.*', r'\1', ' '.join(times_ds)))
        except Exception:
            time_array_index = None

        try:
            try:
                # traverse ODS upstream until time information is found
                time = {}
                for sub in [subtree[:k] for k in range(len(subtree), 0, -1)]:
                    times_sub_ds = [k for k in utimes_ds if k.startswith(l2u(sub))]
                    this_subtree = l2o(sub)

                    # get time data from ods
                    times = {}
                    n = len(location)
                    for item in times_sub_ds:
                        otem = u2o(item, this_subtree)[n:]
                        if otem.replace(':', '0') not in self:
                            continue
                        try:
                            time = self.__getitem__(otem, None)  # traverse ODS
                            if isinstance(time, numpy.ndarray):
                                if time.size == 0:
                                    continue
                                # if time is still multidimensional
                                # (eg. because we are querying the time across different diagnostic channels)
                                # squash the multidimensional time arrays if they are all the same
                                if len(time.shape) > 1:
                                    time = numpy.reshape(time, (-1, time.shape[-1]))
                                    if all(numpy.allclose(time[0], t) for t in time[1:]):
                                        time = time[0]
                            times[item] = time
                        except ValueError as _excp:
                            if 'has no data' in repr(_excp):
                                pass
                            else:
                                # return False if time is not homogeneous
                                extra_info['homogeneous_time'] = False
                                return None
                    times_values = list(times.values())
                    extra_info['location'] = times.keys()

                    # no time data defined
                    if not len(times_values):
                        extra_info['homogeneous_time'] = None
                        return None

                    # We crossed [:] or something and picked up a 2D time array
                    elif any(len(numpy.asarray(time).shape) > 1 for time in times_values):
                        # Make a 1D reference time0 that can be comapred against other time arrays
                        time0 = list(times.values())[0]
                        # Collapse extra dimensions, assuming time is the last one. If it isn't, this will fail.
                        while len(time0.shape) > 1:
                            time0 = numpy.take(time0, 0, axis=0)
                        if all(time.size == time0.size for time in times.values()):
                            for time in times.values():
                                # Make sure all time arrays are close to the time0 we identified
                                assert abs(time - time0).max() < 1e-7
                            extra_info['homogeneous_time'] = True
                            if isinstance(time0, (float, int)):
                                return time0
                            elif time_array_index is not None:
                                return time0[time_array_index]
                            return time0
                        else:  # Similar to ValueError exception caught above
                            extra_info['homogeneous_time'] = False
                            return None

                    # if the time entries that are all consistent with one another
                    elif all([len(numpy.asarray(time).shape) == 1] for time in times_values):
                        # if the time entries that are all consistent with one another
                        if all(
                            [
                                numpy.array_equiv(times_values[0], time) and numpy.allclose(times_values[0], time)
                                for time in times_values[1:]
                            ]
                        ):
                            time = times_values[0]
                            extra_info['homogeneous_time'] = True
                            if isinstance(time, (float, int)):
                                return time
                            elif time_array_index is not None:
                                return time[time_array_index]
                            return time
                        # if the time entries are not consistent with one another
                        else:
                            extra_info['homogeneous_time'] = False
                            return None

                    # We should never end up here
                    else:
                        raise ValueError(
                            f'Error handling time in OMAS for `{l2o(subtree)}`:\n' + '\n'.join([f'{k}:{v}' for k, v in times.items()])
                        )

            except Exception:
                raise

        except Exception as _excp:
            raise _excp.__class__(f'Error setting time for `{l2o(subtree)}`')

        # We should never end up here
        raise ValueError(f'Error handling time in OMAS for `{l2o(subtree)}`')

    def slice_at_time(self, time=None, time_index=None):
        """
        method for selecting a time slice from an time-dependent ODS (NOTE: this method operates in place)

        :param time: time value to select

        :param time_index: time index to select (NOTE: time_index has precedence over time)

        :return: modified ODS
        """

        # set time_index for parent and children
        if 'time' in self and isinstance(self['time'], numpy.ndarray):
            if time_index is None:
                time_index = numpy.argmin(abs(self['time'] - time))
                if (time - self['time'][time_index]) != 0.0:
                    printe('%s sliced at %s instead of requested time %s' % (self.location, self['time'][time_index], time))
                time = self['time'][time_index]
            if time is None:
                time = self['time'][time_index]

        # loop over items
        for item in self.keys():
            # time (if present) is treated last
            if item == 'time':
                continue

            # identify time-dependent data
            info = omas_info_node(o2u(self.ulocation + '.' + str(item)))
            if 'coordinates' in info and any(k.endswith('.time') for k in info['coordinates']):

                # time-dependent arrays
                if not isinstance(self.getraw(item), ODS):
                    self[item] = numpy.atleast_1d(self[item][time_index])

                # time-depentend list of ODSs
                elif isinstance(self[item].omas_data, list) and len(self[item]) and 'time' in self[item][0]:
                    if time_index is None:
                        raise ValueError('`time` array is not set for `%s` ODS' % self.ulocation)
                    tmp = self[item][time_index]
                    self.getraw(item).clear()
                    self.getraw(item)[0] = tmp

            # go deeper inside ODSs that do not have time info
            elif isinstance(self.getraw(item), ODS):
                self.getraw(item).slice_at_time(time=time, time_index=time_index)

        # treat time
        if 'time' in self:
            self['time'] = numpy.atleast_1d(self['time'][time_index])

        return self

    @property
    def parent(self):
        try:
            return self._parent()
        except TypeError:
            return None
        except AttributeError:
            self._parent = None
            return None

    @parent.setter
    def parent(self, value):
        if value is None:
            self._parent = None
        else:
            self._parent = weakref.ref(value)

    @property
    def location(self):
        '''
        Property which returns instance of parent ODS
        '''
        parent = self.parent
        if isinstance(parent, ODC):
            return ''
        elif parent is None:
            return ''
        else:
            parent_location = parent.location
            loc = None
            if isinstance(parent.omas_data, list):
                for k, value in enumerate(parent.omas_data):
                    if value is self:
                        loc = k
                        break
            elif isinstance(parent.omas_data, dict):
                for key in parent.omas_data:
                    if parent.omas_data[key] is self:
                        loc = key
                        break
            if loc is None:
                return ''
            if parent_location:
                return parent_location + '.' + str(loc)
            else:
                return str(loc)

    @property
    def top(self):
        '''
        Property which returns instance of top level ODS
        '''
        top = self
        parent = self.parent
        while parent is not None:
            top = parent
            parent = parent.parent
        return top

    @property
    def structure(self, location):
        '''
        Property which returns structure of current ODS
        '''
        return imas_structure(self.imas_version, self.location)

    @property
    def imas_version(self):
        """
        Property that returns the imas_version of this ods

        :return: string with imas_version
        """
        if not hasattr(self, '_imas_version'):
            self._imas_version = omas_rcparams['default_imas_version']
        top = self.top
        if top._imas_version is None:
            top._imas_version = omas_rcparams['default_imas_version']
        self._imas_version = top._imas_version
        return self._imas_version

    @imas_version.setter
    def imas_version(self, imas_version_value):
        if imas_version_value is None:
            imas_version_value = omas_rcparams['default_imas_version']
        self.top._imas_version = imas_version_value

    @property
    def consistency_check(self):
        """
        property that returns whether consistency with IMAS schema is enabled or not

        :return: True/False/'warn'/'drop'/'strict' or a combination of those strings
        """
        if not hasattr(self, '_consistency_check'):
            self._consistency_check = omas_rcparams['consistency_check']
        return self._consistency_check

    @consistency_check.setter
    def consistency_check(self, consistency_value):
        """
        property that sets whether consistency with IMAS schema is enabled or not

        :param consistency_value: True/False/'warn'/'drop'/'strict' or a combination of those strings
        """
        if not consistency_value and not self._consistency_check:
            return

        old_consistency_check = self._consistency_check
        try:
            # set ._consistency_check for this ODS
            self._consistency_check = consistency_value
            # set .consistency_check
            for item in list(self.keys(dynamic=0)):
                if isinstance(self.getraw(item), ODS) and 'code.parameters' in self.getraw(item).location:
                    # consistency_check=True makes sure that code.parameters is of type CodeParameters
                    if consistency_value:
                        tmp = CodeParameters()
                        tmp.update(self.getraw(item))
                        self.setraw(item, tmp)
                    else:
                        continue
                else:
                    consistency_value_propagate = consistency_value
                    if consistency_value:
                        location = self.location
                        structure = imas_structure(self.imas_version, location)
                        if location.endswith('.ids_properties') and item == 'occurrence':
                            continue
                        else:
                            structure_key = item if not isinstance(item, int) else ':'
                            strict_fail = False
                            if (
                                isinstance(consistency_value, str)
                                and 'strict' in consistency_value
                                and structure_key in structure
                                and p2l(location + '.%s' % item)[0] in _extra_structures
                                and o2i(o2u(location + '.%s' % item)) in _extra_structures[p2l(location + '.%s' % item)[0]]
                            ):
                                strict_fail = True
                            if not strict_fail and structure_key in structure:
                                structure[structure_key]
                            else:
                                options = list(structure.keys())
                                if len(options) == 1 and options[0] == ':':
                                    options = 'A numerical index is needed with n>=0'
                                else:
                                    if len(options) > 5:
                                        options = {
                                            option: difflib.SequenceMatcher(None, structure_key, option).ratio() for option in options
                                        }
                                        index = numpy.argsort(list(options.values())).astype(int)
                                        options = list(numpy.array(list(options.keys()))[index[-5:]][::-1]) + ['...']
                                    options = 'Did you mean: ' + ', '.join(options)
                                txt = 'IMAS %s location: %s' % (self.imas_version, location + '.' + structure_key)
                                if isinstance(consistency_value, str) and ('warn' in consistency_value or 'drop' in consistency_value):
                                    if 'warn' in consistency_value:
                                        if 'drop' in consistency_value:
                                            printe(f'Dropping invalid {txt}')
                                        else:
                                            printe(f'Invalid {txt}')
                                    consistency_value_propagate = False
                                else:
                                    raise LookupError(underline_last(f'Invalid {txt}', len('LookupError: ')) + '\n' + options)
                                if isinstance(consistency_value, str) and 'drop' in consistency_value:
                                    del self[item]
                                    continue
                        # check that value is consistent
                        if not isinstance(self.getraw(item), ODS):
                            location = l2o([location] + [item])
                            info = omas_info_node(o2u(location), imas_version=self.imas_version)
                            value, txt = consistency_checker(location, self.getraw(item), info, consistency_value, self.imas_version)
                            if not len(txt):
                                pass
                            elif isinstance(consistency_value, str) and ('warn' in consistency_value or 'drop' in consistency_value):
                                if 'warn' in consistency_value:
                                    if 'drop' in consistency_value:
                                        printe(f'Dropping invalid {txt}')
                                    else:
                                        printe(f'Invalid {txt}')
                                if isinstance(consistency_value, str) and 'drop' in consistency_value:
                                    del self[item]
                                    continue
                            if value is not self.getraw(item):
                                self.setraw(item, value)
                    # propagate consistency check
                    if isinstance(self.getraw(item), ODS):
                        self.getraw(item).consistency_check = consistency_value_propagate

        except Exception as _excp:
            # restore existing consistency_check value in case of error
            if old_consistency_check != consistency_value:
                for item in self.keys(dynamic=0):
                    if isinstance(self.getraw(item), ODS):
                        self.getraw(item).consistency_check = old_consistency_check
                self.consistency_check = old_consistency_check
            raise  # (LookupError('Consistency check failed: %s' % repr(_excp)))

    @property
    def cocos(self):
        """
        property that tells in what COCOS format the data is stored internally of the ODS
        """
        if not hasattr(self, '_cocos'):
            self._cocos = omas_rcparams['cocos']
        top = self.top
        if top._cocos is None:
            top._cocos = omas_rcparams['cocos']
        self._cocos = top._cocos
        return self._cocos

    @cocos.setter
    def cocos(self, cocos_value):
        if cocos_value is None:
            cocos_value = omas_rcparams['cocos']
        self.top._cocos = cocos_value

    @property
    def cocosio(self):
        """
        property that tells in what COCOS format the data will be input/output
        """
        if not hasattr(self, '_cocosio'):
            self._cocosio = omas_rcparams['cocos']
        top = self.top
        if top._cocosio is None:
            top._cocosio = omas_rcparams['cocos']
        self._cocosio = top._cocosio
        return self._cocosio

    @cocosio.setter
    def cocosio(self, cocosio_value):
        if cocosio_value is None:
            cocosio_value = omas_rcparams['cocos']
        self.top._cocosio = cocosio_value

    @property
    def unitsio(self):
        """
        property that if data should be returned with units or not
        """
        if not hasattr(self, '_unitsio'):
            self._unitsio = {}
        top = self.top
        if top._unitsio is None:
            top._unitsio = {}
        self._unitsio = top._unitsio
        return self._unitsio

    @unitsio.setter
    def unitsio(self, unitsio_value):
        if unitsio_value is None:
            unitsio_value = {}
        self.top._unitsio = unitsio_value

    @property
    def coordsio(self):
        """
        property that tells in what COCOS format the data will be input/output
        """
        if not hasattr(self, '_coordsio'):
            self._coordsio = {}
        top = self.top
        if top._coordsio is None:
            top._coordsio = {}
        self._coordsio = top._coordsio
        return self._coordsio

    @coordsio.setter
    def coordsio(self, coordsio_value):
        if coordsio_value is None:
            coordsio_value = {}
        self.top._coordsio = coordsio_value

    @property
    def dynamic(self):
        """
        property that point to dynamic_ODS object
        """
        if not hasattr(self, '_dynamic'):
            self._coordsio = None
        top = self.top
        self._dynamic = top._dynamic
        return self._dynamic

    @property
    def active_dynamic(self):
        """
        property that point to dynamic_ODS object and says if it is active
        """
        dynamic = self.dynamic
        if dynamic and dynamic.active:
            return dynamic

    @dynamic.setter
    def dynamic(self, dynamic_value):
        self.top._dynamic = dynamic_value

    @property
    def ulocation(self):
        """
        :return: string with location of this object in universal ODS path format
        """
        return o2u(self.location)

    def _validate(self, value, structure, go_deep=False):
        """
        Validate that the value is consistent with the provided structure field

        :param value: sub-tree to be checked

        :param structure: reference structure

        :param go_deep: check value up to its leaves
        """
        for key in value.keys(dynamic=0):
            structure_key = o2u(key)
            if go_deep and isinstance(value[key], ODS) and value[key].consistency_check:
                value._validate(value[key], structure[structure_key])
            else:
                structure[structure_key]

    def __setitem__(self, key, value):
        # handle individual keys as well as full paths
        key = p2l(key)

        if not len(key):
            return self

        if isinstance(key[0], int):
            # negative numbers are used to address arrays of structures from the end
            if key[0] < 0:
                if self.omas_data is None:
                    key[0] = 0
                elif isinstance(self.omas_data, list):
                    if not len(self.omas_data):
                        key[0] = 0
                    else:
                        key[0] = len(self.omas_data) + key[0]
        else:
            # '+' is used to append new entry in array structure
            if key[0] == '+':
                if self.omas_data is None:
                    key[0] = 0
                elif isinstance(self.omas_data, list):
                    key[0] = len(self.omas_data)
            # ':' is used to slice
            elif ':' in key[0]:
                if not self.keys():
                    if key[0] == ':':
                        key[0] = [0]
                    else:
                        key[0] = list(range(slice(*map(lambda x: int(x.strip()) if x.strip() else None, key[0].split(':'))).stop))
                else:
                    key[0] = slice(*map(lambda x: int(x.strip()) if x.strip() else None, key[0].split(':')))

        # handle data slicing on write
        if isinstance(key[0], list):
            for k in key[0]:
                self.__setitem__([k] + key[1:], value)
            return
        elif isinstance(key[0], slice):
            for k in self.keys()[key[0]]:
                self.__setitem__([k] + key[1:], value)
            return

        # handle dynamic path creation for .code.parameters leaf
        if (
            len(key) == 1
            and key[0] == 'parameters'
            and (self.location.endswith('.code') or not self.location)
            and not isinstance(value, str)
        ):
            pass_on_value = value
            value = CodeParameters()
            value.update(pass_on_value)
        # if the user has entered path rather than a single key
        elif len(key) > 1:
            pass_on_value = value
            if key[0] == 'parameters' and (self.location.endswith('.code') or not self.location) and not isinstance(value, str):
                value = CodeParameters()
                value[key[1:]] = pass_on_value
            else:
                value = self.same_init_ods(cls=ODS)

        # full path where we want to place the data
        location = l2o([self.location, key[0]])

        # perform consistency check with IMAS structure
        if self.consistency_check and '.code.parameters.' not in location:
            structure_key = key[0] if not isinstance(key[0], int) else ':'
            try:
                structure = imas_structure(self.imas_version, location)
                if isinstance(value, ODS):
                    if value.omas_data is None and not len(structure) and '.code.parameters' not in location:
                        raise ValueError('`%s` has no data' % location)
                    self._validate(value, structure)
                    if value.omas_data is None:
                        if ':' in structure:
                            value.omas_data = []
                        elif len(structure):
                            value.omas_data = {}

            except (LookupError, TypeError):
                txt = 'Not a valid IMAS %s location: %s' % (self.imas_version, location)
                if isinstance(self.consistency_check, str) and 'warn' in self.consistency_check:
                    printe(txt)
                    if isinstance(value, ODS):
                        value.consistency_check = False
                elif self.consistency_check:
                    try:
                        options = list(imas_structure(self.imas_version, self.location).keys())
                    except KeyError:
                        raise LookupError(txt)
                    if len(options) == 1 and options[0] == ':':
                        options = 'A numerical index is needed with n>=0'
                    else:
                        if len(options) > 5:
                            options = {option: difflib.SequenceMatcher(None, structure_key, option).ratio() for option in options}
                            index = numpy.argsort(list(options.values())).astype(int)
                            options = list(numpy.array(list(options.keys()))[index[-5:]][::-1]) + ['...']
                        options = 'Did you mean: ' + ', '.join(options)
                    raise LookupError(underline_last(txt, len('LookupError: ')) + '\n' + options)

        # check what container type is required and if necessary switch it
        if isinstance(self, ODC):
            pass
        elif not self.omas_data or not len(self.omas_data):
            if isinstance(key[0], int):
                if not isinstance(self.omas_data, list):
                    self.omas_data = []
            else:
                if not isinstance(self.omas_data, dict):
                    self.omas_data = {}
        elif isinstance(key[0], int) and not isinstance(self.omas_data, list):
            raise TypeError('Cannot convert from dict to list once ODS has data')
        elif isinstance(key[0], str) and not isinstance(self.omas_data, dict):
            raise TypeError('Cannot convert from list to dict once ODS has data')

        # if the value is not an ODS strucutre
        if not isinstance(value, ODS):

            # convert simple dict of code.parameters to CodeParameters instances
            if '.code.parameters' in location and not isinstance(value, CodeParameters) and isinstance(value, (dict, ODS)):
                tmp = value
                value = CodeParameters()
                value.update(tmp)

            # now that all checks are completed we can assign the structure information
            if self.consistency_check and '.code.parameters.' not in location:
                ulocation = o2u(location)

                # handle cocos transformations coming in
                if (
                    self.cocosio
                    and self.cocosio != self.cocos
                    and '.' in location
                    and ulocation in omas_physics.cocos_signals
                    and not isinstance(value, ODS)
                ):
                    transform = omas_physics.cocos_signals[ulocation]
                    if transform == '?':
                        if isinstance(self.consistency_check, str) and 'warn' in self.consistency_check:
                            printe('COCOS translation has not been setup: %s' % ulocation)
                            norm = 1.0
                        else:
                            raise ValueError('COCOS translation has not been setup: %s' % ulocation)
                    else:
                        norm = omas_physics.cocos_transform(self.cocosio, self.cocos)[transform]
                    value = value * norm

                # get node information
                info = omas_info_node(ulocation, imas_version=self.imas_version)

                # handle units (Python pint package)
                if str(value.__class__).startswith("<class 'pint."):
                    import pint

                    if (
                        'units' in info
                        and isinstance(value, pint.quantity._Quantity)
                        or (
                            isinstance(value, numpy.ndarray)
                            and value.size
                            and isinstance(numpy.atleast_1d(value).flat[0], pint.quantity._Quantity)
                        )
                    ):
                        value = value.to(info['units']).magnitude

                # coordinates interpolation
                ods_coordinates = self.top
                input_coordinates = self.coordsio
                if input_coordinates:
                    all_coordinates = []
                    coordinates = []
                    if len(input_coordinates) and 'coordinates' in info:
                        all_coordinates = list(map(lambda x: u2o(x, self.location), info['coordinates']))
                        coordinates = list(filter(lambda coord: not coord.startswith('1...'), all_coordinates))
                    if len(coordinates):
                        # add any missing coordinate that were input
                        for coordinate in coordinates:
                            if coordinate not in ods_coordinates and coordinate in input_coordinates:
                                printd('Adding %s coordinate to ods' % (coordinate), topic='coordsio')
                                ods_coordinates[coordinate] = input_coordinates.__getitem__(coordinate, False)

                        # if all coordinates information is present
                        if all(coord in input_coordinates and coord in ods_coordinates for coord in coordinates):
                            # if there is any coordinate that does not match
                            if any(
                                [
                                    len(input_coordinates.__getitem__(coord, None)) != len(ods_coordinates.__getitem__(coord, None))
                                    or (
                                        not numpy.allclose(
                                            input_coordinates.__getitem__(coord, False), ods_coordinates.__getitem__(coord, False)
                                        )
                                    )
                                    for coord in coordinates
                                ]
                            ):

                                # for the time being omas interpolates only 1D quantities
                                if len(info['coordinates']) > 1:
                                    raise Exception('coordio does not support multi-dimentional interpolation just yet')

                                # if the (first) coordinate is in input_coordinates
                                coordinate = coordinates[0]
                                if len(input_coordinates.__getitem__(coordinate, None)) != len(value):
                                    raise Exception(
                                        'coordsio %s.shape=%s does not match %s.shape=%s'
                                        % (coordinate, input_coordinates.__getitem__(coordinate, False).shape, location, value.shape)
                                    )
                                printd('Adding %s interpolated to input %s coordinate' % (self.location, coordinate), topic='coordsio')
                                value = omas_interp1d(
                                    ods_coordinates.__getitem__(coordinate, None), input_coordinates.__getitem__(coordinate, None), value
                                )
                            else:
                                printd('%s ods and coordsio match' % (coordinates), topic='coordsio')
                        else:
                            printd('Adding `%s` without knowing coordinates `%s`' % (self.location, all_coordinates), topic='coordsio')

                    elif not self.active_dynamic and ulocation in omas_coordinates(self.imas_version) and location in ods_coordinates:
                        value = ods_coordinates.__getitem__(location, None)

            # lists are saved as numpy arrays, and 0D numpy arrays as scalars
            value = force_imas_type(value)

            # check that dimensions and data types are consistent with IMAS specifications
            if self.consistency_check and '.code.parameters.' not in location:
                value, txt = consistency_checker(location, value, info, self.consistency_check, self.imas_version)
                if not len(txt):
                    pass
                elif isinstance(self.consistency_check, str) and ('warn' in self.consistency_check or 'drop' in self.consistency_check):
                    if 'warn' in self.consistency_check:
                        if 'drop' in self.consistency_check:
                            printe(f'Dropping invalid {txt}')
                        else:
                            printe(f'Invalid {txt}')
                    if 'drop' in self.consistency_check:
                        return

        # check if the branch/node was dynamically created
        dynamically_created = False
        if key[0] not in self.keys(dynamic=0) and len(key) > 1:
            dynamically_created = True

        # assign values to this ODS
        if key[0] not in self.keys(dynamic=0) or len(key) == 1:
            self.setraw(key[0], value)

        # pass the value one level deeper
        # and cleanup dynamically created branches if necessary (eg. if consistency check fails)
        if len(key) > 1:
            try:
                self.getraw(key[0])[key[1:]] = pass_on_value
            except LookupError:
                if dynamically_created:
                    del self[key[0]]
                raise

        # if the value is an ODS strucutre
        if isinstance(value, ODS) and value.omas_data is not None and len(value.keys(dynamic=0)):
            # we purposly do not force value.consistency_check = self.consistency_check
            # because sub-ODSs could be shared among ODSs that have different settings of consistency_check
            if False and value.consistency_check != self.consistency_check:
                value.consistency_check = self.consistency_check

    def getraw(self, key):
        """
        Method to access data stored in ODS with no processing of the key, and it is thus faster than the ODS.__getitem__(key)
        Effectively behaves like a pure Python dictionary/list __getitem__.
        This method is mostly meant to be used in the inner workings of the ODS class.
        NOTE: ODS.__getitem__(key, False) can be used to access items in the ODS with disabled cocos and coordinates processing but with support for different syntaxes to access data

        :param key: string or integer

        :return: ODS value
        """

        return self.omas_data[key]

    def same_init_ods(self, cls=None):
        """
        Initializes a new ODS with the same attributes as this one

        :return: new ODS
        """
        if cls is None:
            cls = self.__class__
        return cls(
            imas_version=self.imas_version,
            consistency_check=self._consistency_check,
            cocos=self._cocos,
            cocosio=self._cocosio,
            coordsio=self._coordsio,
            dynamic=self.dynamic,
        )

    def setraw(self, key, value):
        """
        Method to assign data to an ODS with no processing of the key, and it is thus faster than the ODS.__setitem__(key, value)
        Effectively behaves like a pure Python dictionary/list __setitem__.
        This method is mostly meant to be used in the inner workings of the ODS class.

        :param key: string, integer or a list of these

        :param value: value to assign

        :return: value
        """
        # accept path as list of keys
        if isinstance(key, list):
            if len(key) > 1:
                if key[0] not in self.keys(dynamic=0):
                    self.setraw(key[0], self.same_init_ods())
                return self.getraw(key[0]).setraw(key[1:], value)
            else:
                key = key[0]

        # set .parent
        if isinstance(value, ODC):
            pass
        elif isinstance(value, ODS):
            if value.parent is not None:
                value = copy.deepcopy(value)
            value.parent = self

        # structure
        if isinstance(key, str) or isinstance(self, ODC):
            if self.omas_data is None:
                self.omas_data = {}
            self.omas_data[key] = value

        # arrays of structures
        else:
            if self.omas_data is None:
                self.omas_data = []

            # dynamic array structure creation
            if key > len(self.omas_data) and omas_rcparams['dynamic_path_creation'] == 'dynamic_array_structures':
                for k in range(len(self.omas_data), key):
                    self.setraw(k, self.same_init_ods())
            # index exists
            if key < len(self.omas_data):
                self.omas_data[key] = value
            # next index creation
            elif key == len(self.omas_data):
                self.omas_data.append(value)
            # missing index
            else:
                if not len(self.omas_data):
                    raise IndexError('`%s[%d]` but ods has no data' % (self.location, key))
                else:
                    raise IndexError(
                        '`%s[%d]` but maximum index is %d\nPerhaps you want to use `with omas_environment(ods, dynamic_path_creation=\'dynamic_array_structures\')'
                        % (self.location, key, len(self.omas_data) - 1)
                    )
        return value

    def __getitem__(self, key, cocos_and_coords=True):
        """
        ODS getitem method allows support for different syntaxes to access data

        :param key: different syntaxes to access data, for example:
              * ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']   # standard Python dictionary syntax
              * ods['equilibrium.time_slice[0].profiles_2d[0].psi']            # IMAS hierarchical tree syntax
              * ods['equilibrium.time_slice.0.profiles_2d.0.psi']              # dot separated string syntax
              * ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]      # list of nodes syntax

            NOTE: Python3.6+ f-strings can be very handy when looping over arrays of structures. For example:
            for time_index in range(len(ods[f'equilibrium.time_slice'])):
                for grid_index in range(len(ods[f'equilibrium.time_slice.{time_index}.profiles_2d'])):
                    print(ods[f'equilibrium.time_slice.{time_index}.profiles_2d.{grid_index}.psi'])

        :param cocos_and_coords: processing of cocos transforms and coordinates interpolations [True/False/None]
              * True: enabled COCOS and enabled interpolation
              * False: enabled COCOS and disabled interpolation
              * None: disabled COCOS and disabled interpolation

        :return: ODS value
        """

        # handle pattern match
        if isinstance(key, str) and key.startswith('@'):
            key = self.search_paths(key, 1, '@')[0]

        # handle individual keys as well as full paths
        key = p2l(key)

        if not len(key):
            return self

        # negative numbers are used to address arrays of structures from the end
        if isinstance(key[0], int) and key[0] < 0:
            if self.omas_data is None:
                key[0] = 0
            elif isinstance(self.omas_data, list):
                if not len(self.omas_data):
                    key[0] = 0
                else:
                    key[0] = len(self.omas_data) + key[0]
        # '+' is used to append new entry in array structure
        if key[0] == '+':
            if self.omas_data is None:
                key[0] = 0
            elif isinstance(self.omas_data, list):
                key[0] = len(self.omas_data)
        # slice
        elif isinstance(key[0], str) and ':' in key[0]:
            key[0] = slice(*map(lambda x: int(x.strip()) if x.strip() else None, key[0].split(':')))

        dynamically_created = False

        # data slicing
        if isinstance(key[0], slice):
            data0 = []
            for k in self.keys(dynamic=1)[key[0]]:
                try:
                    data0.append(self.__getitem__([k] + key[1:], cocos_and_coords))
                except ValueError:
                    data0.append([])
            # raise an error if no data is returned
            if not len(data0):
                raise ValueError('`%s` has no data' % self.location)

            # if they are filled but do not have the same number of dimensions
            shapes = [numpy.asarray(item).shape for item in data0 if numpy.asarray(item).size]
            if not len(shapes):
                return numpy.asarray(data0)
            if not all(len(shape) == len(shapes[0]) for shape in shapes[1:]):
                return data0

            # find maximum shape
            max_shape = []
            for shape in shapes:
                for k, s in enumerate(shape):
                    if len(max_shape) < k + 1:
                        max_shape.append(s)
                    else:
                        max_shape[k] = max(max_shape[k], s)
            max_shape = tuple([len(data0)] + max_shape)

            # find types
            dtypes = [numpy.asarray(item).dtype for item in data0 if numpy.asarray(item).size]
            if not len(dtypes):
                return numpy.asarray(data0)
            if not all(dtype.char == dtypes[0].char for dtype in dtypes[1:]):
                return data0
            dtype = dtypes[0]

            # array of strings
            if dtype.char in 'U':
                return numpy.asarray(data0)

            # define an empty array of shape max_shape
            if dtype.char in 'iIl':
                data = numpy.full(max_shape, 0)
            elif dtype.char in 'df':
                data = numpy.full(max_shape, numpy.nan)
            elif dtype.char in 'O':
                data = numpy.full(max_shape, object())
            else:
                raise ValueError('Not an IMAS data type %s' % dtype.char)

            # place the data withing the the empty array
            if len(max_shape) == 1:
                for k, item in enumerate(data0):
                    data[k] = item
            else:
                for k, item in enumerate(data0):
                    if not sum(numpy.squeeze(item).shape):
                        if len(numpy.atleast_1d(numpy.squeeze(item))):
                            data[k, 0] = numpy.atleast_1d(numpy.squeeze(item))[0]
                    else:
                        data[k, : len(item)] = item

            return data

        # dynamic path creation
        elif key[0] not in self.keys(dynamic=0):
            if omas_rcparams['dynamic_path_creation']:
                if self.active_dynamic:
                    location = l2o([self.location, key[0]])
                if self.active_dynamic and self.dynamic.__contains__(location):
                    value = self.dynamic.__getitem__(location)
                    self.__setitem__(key[0], value)
                elif self.active_dynamic and o2u(location).endswith(':'):
                    dynamically_created = True
                    for k in self.keys(dynamic=1):
                        if k not in self.keys(dynamic=0):
                            self.__setitem__(k, self.same_init_ods())
                else:
                    dynamically_created = True
                    self.__setitem__(key[0], self.same_init_ods())
            else:
                location = l2o([self.location, key[0]])
                raise LookupError('Dynamic path creation is disabled, hence `%s` needs to be manually created' % location)

        value = self.omas_data[key[0]]
        if len(key) > 1:
            # if the user has entered a path rather than a single key
            try:
                if isinstance(value, ODS):
                    return value.__getitem__(key[1:], cocos_and_coords)
                else:
                    return value[l2o(key[1:])]
            except ValueError:  # ValueError is raised when nodes have no data
                if dynamically_created:
                    del self[key[0]]
                raise
        else:

            if cocos_and_coords is not None and self.consistency_check and not isinstance(value, ODS):

                location = l2o([self.location, key[0]])
                ulocation = o2u(location)

                # handle cocos transformations going out
                if self.cocosio and self.cocosio != self.cocos and '.' in location and ulocation in omas_physics.cocos_signals:
                    transform = omas_physics.cocos_signals[ulocation]
                    if transform == '?':
                        if self.consistency_check == 'warn':
                            printe('COCOS translation has not been setup: %s' % ulocation)
                            norm = 1.0
                        else:
                            raise ValueError('COCOS translation has not been setup: %s' % ulocation)
                    else:
                        norm = omas_physics.cocos_transform(self.cocos, self.cocosio)[transform]
                    value = value * norm

                # get node information
                info = omas_info_node(ulocation, imas_version=self.imas_version)

                # coordinates interpolation
                ods_coordinates = self.top
                output_coordinates = self.coordsio
                if cocos_and_coords and output_coordinates:
                    all_coordinates = []
                    coordinates = []
                    if len(output_coordinates) and 'coordinates' in info:
                        all_coordinates = list(map(lambda x: u2o(x, self.location), info['coordinates']))
                        coordinates = list(filter(lambda coord: not coord.startswith('1...'), all_coordinates))
                    if len(coordinates):
                        # if all coordinates information is present
                        if all(coord in output_coordinates and coord in ods_coordinates for coord in coordinates):
                            # if there is any coordinate that does not match
                            if any(
                                [
                                    len(output_coordinates.__getitem__(coord, None)) != len(ods_coordinates.__getitem__(coord, None))
                                    or (
                                        not numpy.allclose(
                                            output_coordinates.__getitem__(coord, None), ods_coordinates.__getitem__(coord, None)
                                        )
                                    )
                                    for coord in coordinates
                                ]
                            ):

                                # for the time being omas interpolates only 1D quantities
                                if len(info['coordinates']) > 1:
                                    raise Exception('coordio does not support multi-dimentional interpolation just yet')

                                # if the (first) coordinate is in output_coordinates
                                coordinate = coordinates[0]
                                if len(ods_coordinates.__getitem__(coordinate, None)) != len(value):
                                    raise Exception(
                                        'coordsio %s.shape=%s does not match %s.shape=%s'
                                        % (coordinate, ods_coordinates.__getitem__(coordinate, False).shape, location, value.shape)
                                    )
                                printd('Returning %s interpolated to output %s coordinate' % (location, coordinate), topic='coordsio')
                                try:
                                    value = omas_interp1d(
                                        output_coordinates.__getitem__(coordinate, None),
                                        ods_coordinates.__getitem__(coordinate, None),
                                        value,
                                    )
                                except TypeError:
                                    if is_uncertain(value):
                                        v = omas_interp1d(
                                            output_coordinates.__getitem__(coordinate, None),
                                            ods_coordinates.__getitem__(coordinate, None),
                                            nominal_values(value),
                                        )
                                        s = omas_interp1d(
                                            output_coordinates.__getitem__(coordinate, None),
                                            ods_coordinates.__getitem__(coordinate, None),
                                            std_devs(value),
                                        )
                                        value = unumpy.uarray(v, s)
                            else:
                                printd('%s ods and coordsio match' % (coordinates), topic='coordsio')
                        else:
                            printd(
                                'Getting `%s` without knowing some of the coordinates `%s`' % (self.location, all_coordinates),
                                topic='coordsio',
                            )

                    elif ulocation in omas_coordinates(self.imas_version) and location in output_coordinates:
                        value = output_coordinates.__getitem__(location, False)

                # handle units (Python pint package)
                if 'units' in info and self.unitsio:
                    import pint
                    from .omas_setup import ureg

                    if ureg[0] is None:
                        import pint

                        ureg[0] = pint.UnitRegistry()
                    value = value * getattr(ureg[0], info['units'])

            return value

    def __delitem__(self, key):
        # handle individual keys as well as full paths
        key = p2l(key)
        if len(key) > 1:
            # if the user has entered path rather than a single key
            del self.getraw(key[0])[key[1:]]
        else:
            return self.omas_data.__delitem__(key[0])

    def paths(self, return_empty_leaves=False, traverse_code_parameters=True, include_structures=False, dynamic=True, **kw):
        """
        Traverse the ods and return paths to its leaves

        :param return_empty_leaves: if False only return paths to leaves that have data
                                    if True also return paths to empty leaves

        :param traverse_code_parameters: traverse code parameters

        :param include_structures: include paths leading to the leaves

        :param dynamic: traverse paths that are not loaded in a dynamic ODS

        :return: list of paths that have data
        """
        paths = kw.setdefault('paths', [])
        path = kw.setdefault('path', [])
        for kid in self.keys(dynamic=dynamic):
            if isinstance(self.getraw(kid), ODS):
                if include_structures:
                    paths.append(path + [kid])
                self.getraw(kid).paths(
                    return_empty_leaves=return_empty_leaves,
                    traverse_code_parameters=traverse_code_parameters,
                    include_structures=include_structures,
                    dynamic=dynamic,
                    paths=paths,
                    path=path + [kid],
                )
            elif traverse_code_parameters and isinstance(self.getraw(kid), CodeParameters):
                if include_structures:
                    paths.append(path)
                self.getraw(kid).paths(paths=paths, path=path + [kid])
            else:
                paths.append(path + [kid])
        if not len(self.keys(dynamic=dynamic)) and return_empty_leaves:
            paths.append(path)
        return paths

    def pretty_paths(self, **kw):
        r"""
        Traverse the ods and return paths that have data formatted nicely

        :param \**kw: extra keywords passed to the path() method

        :return: list of paths that have data formatted nicely
        """
        return list(map(l2i, self.paths(**kw)))

    def full_paths(self, **kw):
        r"""
        Traverse the ods and return paths from root of ODS that have data

        :param \**kw: extra keywords passed to the path() method

        :return: list of paths that have data
        """
        location = p2l(self.location)
        return [location + path for path in self.paths(**kw)]

    def full_pretty_paths(self, **kw):
        r"""
        Traverse the ods and return paths from root of ODS that have data formatted nicely

        :param \**kw: extra keywords passed to the full_paths() method

        :return: list of paths that have data formatted nicely
        """
        return list(map(l2i, self.full_paths(**kw)))

    def flat(self, **kw):
        r"""
        Flat dictionary representation of the data

        :param \**kw: extra keywords passed to the path() method

        :return: OrderedDict with flat representation of the data
        """
        tmp = OrderedDict()
        for path in self.paths(**kw):
            tmp[l2o(path)] = self[path]
        return tmp

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        key = p2l(key)
        h = self

        for c, k in enumerate(key):
            # h.omas_data is None when dict/list behaviour is not assigned
            if h.omas_data is not None and k in h.keys(dynamic=0):
                h = h.__getitem__(k, False)
                continue  # continue to the next key

            else:
                # dynamic loading
                if self.active_dynamic:
                    if k in self.dynamic.keys(l2o(p2l(self.location) + key[:c])):
                        continue  # continue to the next key

                # handle ':' by looking under each array of structure
                # and returning True if any of them have the entry we are loooking for
                if isinstance(k, str) and ':' in k:
                    s = slice(*map(lambda x: int(x.strip()) if x.strip() else None, k.split(':')))
                    for k in range(len(h.omas_data))[s]:
                        if key[c + 1 :] in h.omas_data[k]:
                            return True

                return False

        # return False if checking existance of a leaf and the leaf exists but is unassigned
        if not self.active_dynamic and isinstance(h, ODS) and h.omas_data is None:
            return False

        return True

    def keys(self, dynamic=True):
        '''
        Return list of keys

        :param dynamic: whether dynamic loaded key should be shown.
                        This is `True` by default because this should be the case for calls that are facing the user.
                        Within the inner workings of OMAS we thus need to be careful and keep track of when this should not be the case.
                        Throughout the library we use `dynamic=1` or `dynamic=0` for debug purposes, since one can place a conditional
                        breakpoint in this function checking if `dynamic is True and self.dynamic` to verfy that indeed the `dynamic=True`
                        calls come from the user and not from within the library itself.

        :return: list of keys
        '''
        if dynamic and self.active_dynamic:
            if isinstance(self.omas_data, dict):
                dynamic_keys = list(self.dynamic.keys(self.location))
                return sorted(numpy.unique(list(self.omas_data.keys()) + dynamic_keys).tolist())
            elif isinstance(self.omas_data, list):
                # the first time dynamic data is loaded, empty ODS structures will populate self.omas_data
                if len(self.omas_data):
                    return list(range(len(self.omas_data)))
                else:
                    dynamic_keys = list(self.dynamic.keys(self.location))
                    if ':' in dynamic_keys:
                        raise Exception(
                            f'{self.dynamic.__class__.__name__}.keys() at `{self.location}` did not return a range with number of structures'
                        )
                    return dynamic_keys
            elif not self.location:
                return self.dynamic.keys(self.location)
            else:
                return []
        else:
            if isinstance(self.omas_data, dict):
                return list(self.omas_data.keys())
            elif isinstance(self.omas_data, list):
                return list(range(len(self.omas_data)))
            else:
                return []

    def values(self, dynamic=True):
        return [self[item] for item in self.keys(dynamic=dynamic)]

    def items(self, dynamic=True):
        return [(item, self[item]) for item in self.keys(dynamic=dynamic)]

    def __str__(self):
        return self.location

    def __repr__(self):
        return repr(self.omas_data)

    def __tree_repr__(self):
        """
        OMFIT tree representation
        """
        if not self.location:
            return self, []
        s = '--{%d}--' % len(self)
        if 'dataset_description.data_entry' in self:
            s += ' ' + ' '.join(['%s:%s' % (k, v) for k, v in self['dataset_description']['data_entry'].items() if v not in ['None', None]])
        if 'summary.ids_properties.comment' in self:
            s += ' ' + repr(self['summary.ids_properties.comment'])
        return s, []

    def __tree_keys__(self):
        """
        OMFIT tree keys display dynamic
        """
        return self.keys(dynamic=1)

    def get(self, key, default=None):
        r"""
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
        state = {}
        for item in ['omas_data'] + omas_ods_attrs:
            if item in self.__dict__:
                # we do not want to carry with us this information
                if item in ['_cocosio', '_coordsio', '_unitsio', '_parent', '_dynamic']:
                    state[item] = None
                else:
                    state[item] = self.__dict__[item]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for item in omas_ods_attrs:
            self.__dict__.setdefault(item, None)
        if isinstance(self.omas_data, list):
            for value in self.omas_data:
                if isinstance(value, ODS):
                    value.parent = self
        elif isinstance(self.omas_data, dict):
            for key in self.omas_data:
                if isinstance(self.omas_data[key], ODS):
                    self.omas_data[key].parent = self
        return self

    def __deepcopy__(self, memo):
        tmp = self.same_init_ods()
        memo[id(self)] = tmp
        if self.omas_data is None:
            return tmp
        elif isinstance(self.omas_data, list):
            tmp.omas_data = []
            for k, value in enumerate(self.omas_data):
                tmp.omas_data.append(value.__deepcopy__(memo=memo))
                tmp.omas_data[k].parent = tmp
        else:
            tmp.omas_data = {}
            for key in self.omas_data:
                if isinstance(self.omas_data[key], ODS):
                    tmp.omas_data[key] = self[key].__deepcopy__(memo=memo)
                    tmp.omas_data[key].parent = tmp
                else:
                    tmp.omas_data[key] = copy.deepcopy(self[key], memo=memo)
        return tmp

    def copy(self):
        """
        :return: copy.deepcopy of current ODS object
        """
        return copy.deepcopy(self)

    def clear(self):
        """
        remove data from a branch

        :return: current ODS object
        """
        if isinstance(self.omas_data, dict):
            self.omas_data.clear()
        elif isinstance(self.omas_data, list):
            self.omas_data[:] = []
        return self

    def copy_attrs_from(self, ods):
        """
        copy omas_ods_attrs attributes from input ods

        :param ods: input ods

        :return: self
        """
        for item in omas_ods_attrs:
            if item not in ['_parent', '_dynamic']:
                setattr(self, item, getattr(ods, item, None))
        return self

    def prune(self):
        """
        Prune ODS branches that are leafless

        :return: number of branches that were pruned
        """
        n = 0
        for item in self.keys(dynamic=0):
            if isinstance(self.getraw(item), ODS):
                n += self.getraw(item).prune()
                if not len(self.getraw(item).keys()):
                    n += 1
                    del self[item]
        return n

    def set_time_array(self, key, time_index, value):
        """
        Convenience function for setting time dependent arrays

        :param key: ODS location to edit

        :param time_index: time index of the value to set

        :param value: value to set

        :return: time dependent array
        """

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
            key = p2l(key)
            raise IndexError('%s has length %d and time_index %d is bejond current range' % (l2o(key), len(orig_value), time_index))

        self[key] = numpy.atleast_1d(orig_value)
        return orig_value

    def update(self, ods2):
        """
        Adds ods2's key-values pairs to the ods

        :param ods2: dictionary or ODS to be added into the ODS
        """
        if isinstance(ods2, ODS):
            for item in ods2.paths():
                self[item] = ods2[item]
        else:
            with omas_environment(self, dynamic_path_creation='dynamic_array_structures'):
                for item in ods2.keys():
                    self[item] = ods2[item]
        return self

    def list_coordinates(self, absolute_location=True):
        """
        return dictionary with coordinates in a given ODS

        :param absolute_location: return keys as absolute or relative locations

        :return: dictionary with coordinates
        """
        coords = {}

        n = len(self.location)
        for full_path in self.full_paths():
            if l2u(full_path) in omas_coordinates(self.imas_version):
                if absolute_location:
                    coords[l2o(full_path)] = self[l2o(full_path)[n:].strip('.')]
                else:
                    coords[l2o(full_path)[n:].strip('.')] = self[l2o(full_path)[n:].strip('.')]

        return coords

    def coordinates(self, key=None):
        """
        return dictionary with coordinates of a given ODS location

        :param key: ODS location to return the coordinates of
                    Note: both the key location and coordinates must have data

        :return: OrderedDict with coordinates of a given ODS location
        """
        coords = OrderedDict()

        if key is None:
            return self.list_coordinates(absolute_location=False)

        self.__getitem__(key, False)  # raise an error if data is not there

        location = self.location
        uxlocation = l2u(p2l(location) + p2l(key))
        info = omas_info_node(uxlocation)
        if 'coordinates' not in info:
            raise ValueError('ODS location `%s` has no coordinates information' % uxlocation)
        coordinates = list(map(lambda x: u2o(x[len(location) :].strip('.'), key), info['coordinates']))
        for coord in coordinates:
            coords[coord] = self[coord]  # this will raise an error if the coordinates data is not there

        return coords

    def search_paths(self, search_pattern, n=None, regular_expression_startswith=''):
        """
        Find ODS locations that match a pattern

        :param search_pattern: regular expression ODS location string

        :param n: raise an error if a number of occurrences different from n is found

        :param regular_expression_startswith: indicates that use of regular expressions
               in the search_pattern is preceeded by certain characters.
               This is used internally by some methods of the ODS to force users
               to use '@' to indicate access to a path by regular expression.

        :return: list of ODS locations matching search_pattern pattern
        """
        if not isinstance(search_pattern, str):
            return [search_pattern]

        elif regular_expression_startswith:
            if not search_pattern.startswith(regular_expression_startswith):
                return [search_pattern]
            else:
                search_pattern = search_pattern[len(regular_expression_startswith) :]

        search = re.compile(search_pattern)
        matches = []
        for path in map(l2o, self.full_paths()):
            if re.match(search, path):
                matches.append(path)
        if n is not None and len(matches) != n:
            raise ValueError(
                'Found %d matches of `%s` instead of the %d requested\n%s' % (len(matches), search_pattern, n, '\n'.join(matches))
            )
        return matches

    def xarray(self, key):
        """
        Returns data of an ODS location and correspondnig coordinates as an xarray dataset
        Note that the Dataset and the DataArrays have their attributes set with the ODSs structure info

        :param key: ODS location

        :return: xarray dataset
        """
        key = self.search_paths(key, 1, '@')[0]

        import xarray

        key = p2l(key)
        location = p2l(self.location)
        full_key = location + key

        info = omas_info_node(l2u(full_key))
        coords = self.coordinates(key)

        short_coords = OrderedDict()
        for coord in coords:
            short_coords[p2l(coord)[-1]] = coords[coord]

        ds = xarray.Dataset()
        ds[key[-1]] = xarray.DataArray(self[key], coords=short_coords, dims=short_coords.keys(), attrs=info)
        ds.attrs['y'] = key[-1]
        ds.attrs['y_rel'] = l2o(key)
        ds.attrs['y_full'] = l2o(full_key)

        ds.attrs['x'] = []
        ds.attrs['x_rel'] = []
        ds.attrs['x_full'] = []
        for coord in coords:
            coord = p2l(coord)
            info = omas_info_node(l2u(coord))
            ds[coord[-1]] = xarray.DataArray(coords[l2o(coord)], dims=coord[-1], attrs=info)
            ds.attrs['x'].append(coord[-1])
            ds.attrs['x_rel'].append(l2o(coord))
            ds.attrs['x_full'].append(l2o(location + coord))
        return ds

    def dataset(self, homogeneous=[False, 'time', 'full', None][-1]):
        """
        Return xarray.Dataset representation of a whole ODS

        Forming the N-D labeled arrays (tensors) that are at the base of xarrays,
        requires that the number of elements in the arrays do not change across
        the arrays of data structures.

        :param homogeneous: * False: flat representation of the ODS
                                      (data is not collected across arrays of structures)
                            * 'time': collect arrays of structures only along the time dimension
                                      (always valid for homogeneous_time=True)
                            * 'full': collect arrays of structures along all dimensions
                                      (may be valid in many situations, especially related to
                                       simulation data with homogeneous_time=True and where
                                       for example number of ions, sources, etc. do not vary)
                            * None: smart setting, uses homogeneous='time' if homogeneous_time=True else False

        :return: xarray.Dataset
        """
        import xarray

        if not self.location:
            DS = xarray.Dataset()
            for ds in self.keys(dynamic=0):
                DS.update(self[ds].dataset(homogeneous=homogeneous))
            return DS

        def arraystruct_indexnames(key):
            """
            return list of strings with a name for each of the arrays of structures indexes

            :param key: ods location

            :return: list of strings
            """
            base = key.split('.')[0]
            coordinates = []
            counter = 0
            for c in [':'.join(key.split(':')[: k + 1]).strip('.') for k, struct in enumerate(key.split(':'))]:
                info = omas_info_node(o2u(c))
                if 'coordinates' in info:
                    for infoc in info['coordinates']:
                        if infoc == '1...N':
                            infoc = c
                        coordinates.append(
                            '__'
                            + '_'.join(
                                [base, infoc.split('.')[-1]]
                                + [str(k) for k in p2l(key) if isinstance(k, int) or k == ':']
                                + ['index%d__' % counter]
                            )
                        )
                        counter += 1
            return coordinates

        # Generate paths with ':' for the arrays of structures
        # that we want to collect across
        paths = self.paths()
        if self.location:
            fpaths = list(map(lambda key: [self.location] + key, paths))

        if homogeneous is None:
            homogeneous = 'time' if self.homogeneous_time() else False
        if not homogeneous:
            fupaths = list(map(l2o, fpaths))
        elif homogeneous == 'time':
            fupaths = numpy.unique(list(map(l2ut, fpaths)))
        elif homogeneous == 'full':
            fupaths = numpy.unique(list(map(l2u, fpaths)))
        else:
            raise ValueError("OMAS dataset homogeneous attribute can only be [False, 'time', 'full', None]")
        upaths = fupaths
        if self.location:
            n = len(self.location)
            upaths = list(map(lambda key: key[n + 1 :], fupaths))

        # Figure out coordinate indexes
        # NOTE: We use coordinates indexes instead of proper coordinates
        #       since in IMAS these are time dependent quantities
        #       Eg. 'equilibrium.time_slice[:].profiles_1d.psi'
        coordinates = {}
        for fukey, ukey in zip(fupaths, upaths):
            coordinates[fukey] = arraystruct_indexnames(fukey)

        # Generate dataset
        DS = xarray.Dataset()
        for fukey, ukey in zip(fupaths, upaths):
            if not len(omas_info_node(o2u(fukey))):
                printe(f'WARNING: {o2i(fukey)} is not part of IMAS')
                continue
            data = self[ukey]  # OMAS data slicing at work
            for k, c in enumerate(coordinates[fukey]):
                if c not in DS:
                    DS[c] = xarray.DataArray(numpy.arange(data.shape[k]), dims=c)
            try:
                try:
                    DS[fukey] = xarray.DataArray(data, dims=coordinates[fukey])
                except Exception:
                    raise
            except Exception as _excp:
                raise ValueError(
                    f'Error collecting `{fukey}` which has coordinates {coordinates[fukey]}. Are coordinates not homogeneous?\n'
                    + str(_excp)
                )
        return DS

    def satisfy_imas_requirements(self, attempt_fix=True, raise_errors=True):
        """
        Assign .time and .ids_properties.homogeneous_time info for top-level structures
        since these are required for writing an IDS to IMAS

        :param attempt_fix: fix dataset_description and wall IDS to have 0 times if none is set

        :param raise_errors: raise errors if could not satisfy IMAS requirements

        :return: `True` if all is good, `False` if requirements are not satisfied, `None` if fixes were applied
        """
        status = self.physics_consistent_times(attempt_fix=attempt_fix, raise_errors=raise_errors)
        self.physics_imas_info()
        return status

    def save(self, *args, **kw):
        r"""
        Save OMAS data

        :param filename: filename.XXX where the extension is used to select save format method (eg. 'pkl','nc','h5','ds','json','ids')
                         set to `imas`, `s3`, `hdc`, `mongo` for load methods that do not have a filename with extension

        :param \*args: extra arguments passed to save_omas_XXX() method

        :param \**kw: extra keywords passed to save_omas_XXX() method

        :return: return from save_omas_XXX() method
        """
        # figure out format used
        ext, args = _handle_extension(*args)
        # save
        return eval('save_omas_' + ext)(self, *args, **kw)

    def load(self, *args, **kw):
        r"""
        Load OMAS data

        :param filename: filename.XXX where the extension is used to select load format method (eg. 'pkl','nc','h5','ds','json','ids')
                         set to `imas`, `s3`, `hdc`, `mongo` for save methods that do not have a filename with extension

        :param consistency_check: perform consistency check once the data is loaded

        :param \*args: extra arguments passed to load_omas_XXX() method

        :param \**kw: extra keywords passed to load_omas_XXX() method

        :return: ODS with loaded data
        """
        # figure out format used
        ext, args = _handle_extension(*args)

        # manage consistency_check logic
        if 'consistency_check' in kw:
            consistency_check = kw['consistency_check']
        else:
            consistency_check = self.consistency_check
        if self.location:
            kw['consistency_check'] = False
        else:
            kw['consistency_check'] = consistency_check

        # load the data
        results = eval('load_omas_' + ext)(*args, **kw)

        # mongoDB may return more than one result, or none
        if ext in ['mongo']:
            if not len(results):
                raise RuntimeError(ext + ' query returned no result!')
            elif len(results) > 1:
                raise RuntimeError(ext + ' query returned more than one result!')
            else:
                results = list(results.values())[0]

        # update the data
        self.omas_data = results.omas_data
        if isinstance(self.omas_data, list):
            for value in self.omas_data:
                value.omas_data.parent = self
        elif isinstance(self.omas_data, dict):
            for key in self.omas_data:
                if isinstance(self.omas_data[key], ODS):
                    self.omas_data[key].parent = self

        # for pickle we can copy attrs over
        if ext == 'pkl':
            self.copy_attrs_from(results)

        # apply consistency checks
        if consistency_check != self.consistency_check or consistency_check != results.consistency_check:
            self.consistency_check = consistency_check

        return self

    def open(self, *args, **kw):
        r"""
        Dynamically load OMAS data for seekable storage formats

        :param filename: filename.XXX where the extension is used to select load format method (eg. 'nc','h5','ds','json','ids')
                         set to `imas`, `s3`, `hdc`, `mongo` for save methods that do not have a filename with extension

        :param consistency_check: perform consistency check once the data is loaded

        :param \*args: extra arguments passed to dynamic_omas_XXX() method

        :param \**kw: extra keywords passed to dynamic_omas_XXX() method

        :return: ODS with loaded data
        """
        # manage consistency_check logic
        if 'consistency_check' in kw:
            consistency_check = kw.pop('consistency_check')
        else:
            consistency_check = self.consistency_check
        if self.location:
            consistency_check = False

        # without args/kw re-connect to a dynamic ODS
        if not len(args) and not len(kw):
            if self.dynamic:
                return self.dynamic.open()
            else:
                raise ValueError('ods was not previously open. Must specify .open() arguments.')

        # figure out format used
        ext, args = _handle_extension(*args)

        if ext in ['nc', 'imas', 'machine']:
            # apply consistency checks
            if consistency_check != self.consistency_check:
                self.consistency_check = consistency_check

            if ext == 'nc':
                from omas.omas_nc import dynamic_omas_nc

                self.dynamic = dynamic_omas_nc(*args, **kw)
            elif ext == 'imas':
                from omas.omas_imas import dynamic_omas_imas

                self.dynamic = dynamic_omas_imas(*args, **kw)
            elif ext == 'machine':
                from omas.omas_machine import dynamic_omas_machine

                self.dynamic = dynamic_omas_machine(*args, **kw)

            self.dynamic.open()
            return self
        else:
            raise ValueError(ext + ' OMAS storage does not support dynamic loading')

    def close(self):
        if self.dynamic:
            self.dynamic.close()

    def diff(self, ods, ignore_type=False, ignore_empty=False, ignore_keys=[], ignore_default_keys=True):
        """
        return differences between this ODS and the one passed

        :param ods: ODS to compare against

        :param ignore_type: ignore object type differences

        :param ignore_empty: ignore empty nodes

        :return: dictionary with differences
        """
        return different_ods(
            self, ods, ignore_type=ignore_type, ignore_empty=ignore_empty, ignore_keys=ignore_keys, ignore_default_keys=ignore_default_keys
        )

    def diff_attrs(self, ods, attrs=omas_ods_attrs, verbose=False):
        '''
        Checks if two ODSs have any difference in their attributes

        :param ods: ODS to compare against

        :param attrs: list of attributes to compare

        :param verbose: print differences to stdout

        :return: dictionary with list of attriibutes that have differences, or False otherwise
        '''
        return different_ods_attrs(self, ods, attrs=attrs, verbose=verbose)

    def from_structure(self, structure, depth=0):
        """
        Generate an ODS starting from a hierarchical structure made of dictionaries and lists

        :param structure: input structure

        :return: self
        """
        if isinstance(structure, dict):
            keys = list(map(str, structure.keys()))
        elif isinstance(structure, list):
            keys = list(range(len(structure)))
        else:
            raise ValueError('from_structure must be fed either a structure made of dictionaries and lists')

        for item in keys:
            if isinstance(structure[item], dict):
                self[item].from_structure(structure[item], depth=depth + 1)
            elif isinstance(structure[item], list):
                # identify if this is a leaf
                if (len(structure[item]) and not isinstance(structure[item][0], dict)) or not len(structure[item]):
                    self.setraw(item, numpy.array(structure[item]))
                # or a node in the IMAS tree
                else:
                    self[item].from_structure(structure[item], depth=depth + 1)
            else:
                self.setraw(item, copy.deepcopy(structure[item]))
            if depth == 0 and isinstance(self[item], ODS):
                self[item].consistency_check = self.consistency_check
        return self

    def codeparams2xml(self):
        """
        Convert code.parameters to a XML string
        """
        if not self.location:
            for item in self.keys(dynamic=0):
                self[item].codeparams2xml()
            return
        elif 'code.parameters' in self and isinstance(self['code.parameters'], CodeParameters):
            self['code.parameters'] = self['code.parameters'].to_string()
        elif 'parameters' in self and isinstance(self['parameters'], CodeParameters):
            self['parameters'] = self['parameters'].to_string()

    def codeparams2dict(self):
        """
        Convert code.parameters to a CodeParameters dictionary object
        """
        import xml

        if not self.location:
            for item in self.keys(dynamic=0):
                self[item].codeparams2dict()
            return
        try:
            if 'code.parameters' in self and isinstance(self['code.parameters'], str):
                self['code.parameters'] = CodeParameters().from_string(self['code.parameters'])
            elif 'parameters' in self and isinstance(self['parameters'], str):
                self['parameters'] = CodeParameters().from_string(self['parameters'])
        except xml.parsers.expat.ExpatError:
            printe('%s.code.parameters is not formatted as XML' % self.location)
        except Exception as _excp:
            printe('Issue with %s.code.parameters: %s' % (self.location, repr(_excp)))

    def sample(self, ntimes=1, homogeneous_time=None):
        """
        Populates the ods with sample data

        :param ntimes: number of time slices to generate

        :param homogeneous_time: only return samples that have ids_properties.homogeneous_time either True or False

        :return: self
        """
        for func in omas_sample.__ods__:
            printd(f'Adding {func} sample data to ods', topic='sample')
            args, kw, _, _ = function_arguments(getattr(self, 'sample_' + func))
            if 'time_index' in kw:
                for k in range(ntimes):
                    getattr(self, 'sample_' + func)(time_index=k)
            else:
                getattr(self, 'sample_' + func)()

        # remove IDSs that do not have the same homogeneous_time property as requested
        if homogeneous_time is not None:
            self.physics_consistent_times()  # to add homogeneous_time info
            for ds in list(self.keys()):
                if self[ds]['ids_properties']['homogeneous_time'] != homogeneous_time:
                    del self[ds]

        return self

    def document(self, what=['coordinates', 'data_type', 'documentation', 'units']):
        """
        RST documentation of the ODs content

        :param what: fields to be included in the documentation
                     if None, all fields are included

        :return: string with RST documentation
        """
        pp = {}
        for item in self.pretty_paths():
            tt = l2u(p2l(item))
            pp[tt] = omas_info_node(tt)

        txt = []
        for item in sorted(pp.keys()):
            txt += ['', item]
            txt += ['-' * len(item)]
            for elms in pp[item]:
                if what is not None and elms not in what:
                    continue
                value = str(pp[item][elms])
                txt += [f'* {elms}: {value}']

        return '\n'.join(txt)

    def to_odx(self, homogeneous=None):
        '''
        Generate a ODX from current ODS

        :param homogeneous: * False: flat representation of the ODS
                                      (data is not collected across arrays of structures)
                            * 'time': collect arrays of structures only along the time dimension
                                      (always valid for homogeneous_time=True)
                            * 'full': collect arrays of structures along all dimensions
                                      (may be valid in many situations, especially related to
                                       simulation data with homogeneous_time=True and where
                                       for example number of ions, sources, etc. do not vary)
                            * None: smart setting, uses homogeneous='time' if homogeneous_time=True else False

        :return: ODX
        '''
        return ods_2_odx(self, homogeneous=homogeneous)

    def info(self, location):
        '''
        return node info

        :param location: location of the node to return info of

        :return: dictionary with info
        '''
        return omas_info_node((self.location + '.' + location).lstrip('.'))

    def __enter__(self):
        if self.dynamic:
            return self.dynamic.__enter__()
        else:
            raise RuntimeError('Missing call to .open() ?')

    def __exit__(self, type, value, traceback):
        if self.dynamic:
            return self.dynamic.__exit__(type, value, traceback)
        else:
            raise RuntimeError('Missing call to .open() ?')


omas_dictstate = dir(ODS)
omas_dictstate.extend(['omas_data'] + omas_ods_attrs)
omas_dictstate = sorted(list(set(omas_dictstate)))


class ODC(ODS):
    """
    OMAS Data Collection class
    """

    def __init__(self, *args, **kw):
        ODS.__init__(self, *args, **kw)
        self.omas_data = {}

    @property
    def consistency_check(self):
        return False

    @consistency_check.setter
    def consistency_check(self, consistency_value):
        for item in self.keys(dynamic=0):
            self[item].consistency_check = consistency_value
        self._consistency_check = consistency_value

    def same_init_ods(self, cls=None):
        if cls is None:
            cls = ODS
        return ODS.same_init_ods(self, cls=cls)

    def keys(self, dynamic=True):
        keys = list(self.omas_data.keys())
        for k, item in enumerate(keys):
            try:
                keys[k] = c(item)
            except Exception:
                pass
        return keys

    def save(self, *args, **kw):
        # figure out format that was used
        if '/' not in args[0] and '.' not in os.path.split(args[0])[1]:
            ext = args[0]
            args = args[1:]
        else:
            ext = os.path.splitext(args[0])[-1].strip('.')
            if not ext:
                ext = 'pkl'

        if ext in ['pkl', 'nc', 'json', 'h5']:
            pass
        else:
            raise ValueError(f'Cannot save ODC to {ext} format')

        return ODS.save(self, *args, **kw)

    def load(self, *args, **kw):
        # figure out format that was used
        if '/' not in args[0] and '.' not in os.path.split(args[0])[1]:
            ext = args[0]
            args = args[1:]
        else:
            ext = os.path.splitext(args[0])[-1].strip('.')
            if not ext:
                ext = 'pkl'

        if ext == 'pkl':
            pass
        elif ext in ['h5', 'nc', 'json']:
            kw['cls'] = ODC
        else:
            raise ValueError(f'Cannot load ODC from {ext} format')

        # manage consistency_check logic
        if 'consistency_check' not in kw:
            kw['consistency_check'] = True

        return ODS.load(self, *args, **kw)


class dynamic_ODS:
    """
    Abstract base class that dynamic_omas_... classes inherit from
    """

    kw = {}

    active = False

    def __init__(self):
        raise NotImplementedError('Classes that subclass %s should have a __init__() method' % self.__class__)

    def open(self):
        raise NotImplementedError('Classes that subclass %s should have a open() method' % self.__class__)

    def close(self):
        raise NotImplementedError('Classes that subclass %s should have a close() method' % self.__class__)

    def __getstate__(self):
        return self.kw

    def __setstate__(self, kw):
        self.kw = kw
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class CodeParameters(dict):
    """
    Class used to interface with IMAS code-parameters XML files
    """

    def __init__(self, string=None):
        if isinstance(string, str):
            if os.path.exists(string):
                self.from_file(string)
            else:
                self.from_string(string)

    def from_string(self, code_params_string):
        """
        Load data from code.parameters XML string

        :param code_params_string: XML string

        :return: self
        """
        import xmltodict

        self.clear()
        if not code_params_string.strip().endswith('</parameters>'):
            code_params_string = '<parameters>' + code_params_string + '</parameters>'
        tmp = xmltodict.parse(code_params_string).get('parameters', '')
        if not isinstance(tmp, (list, dict)):
            import xml

            raise xml.parsers.expat.ExpatError('Not in XML format')
        elif tmp:
            recursive_interpreter(tmp, dict_cls=CodeParameters)
            self.update(tmp)
        return self

    def from_file(self, code_params_file):
        """
        Load data from code.parameters XML file

        :param code_params_file: XML file

        :return: self
        """
        with open(code_params_file, 'r') as f:
            return self.from_string(f.read())

    def to_string(self):
        """
        generate an XML string from this dictionary

        :return: XML string
        """
        import xmltodict

        tmp = {'parameters': CodeParameters()}
        tmp['parameters'].update(copy.deepcopy(self))
        recursive_encoder(tmp)
        return xmltodict.unparse(tmp, pretty=True)

    def __setitem__(self, key, value):
        key = p2l(key)

        if not len(key):
            return self

        # go deeper
        if len(key) > 1:
            if key[0] not in self or not isinstance(self[key[0]], CodeParameters):
                self.setraw(key[0], self.__class__())
            self.getraw(key[0])[key[1:]] = value
        # return leaf
        else:
            # convert ODSs to CodeParameters
            if isinstance(value, ODS):
                value = CodeParameters()
            self.setraw(key[0], value)

    def __getitem__(self, key):
        key = p2l(key)

        if not len(key):
            return self

        # go deeper
        if len(key) > 1:
            return self.getraw(key[0])[key[1:]]
        # return leaf
        else:
            return self.getraw(key[0])

    def getraw(self, key):
        """
        Method to access data to CodeParameters with no processing of the key.
        Effectively behaves like a pure Python dictionary/list __getitem__.
        This method is mostly meant to be used in the inner workings of the CodeParameters class.

        :param key: string or integer

        :return: value
        """
        return dict.__getitem__(self, key)

    def setraw(self, key, value):
        """
        Method to assign data to CodeParameters with no processing of the key.
        Effectively behaves like a pure Python dictionary/list __setitem__.
        This method is mostly meant to be used in the inner workings of the CodeParameters class.

        :param key: string or integer

        :param value: value to assign

        :return: value
        """
        return dict.__setitem__(self, key, value)

    def update(self, value):
        """
        Update CodeParameters
        NOTE: ODSs will be converted to CodeParameters classes

        :param value: dictionary structure

        :return: self
        """
        # convert ODSs to CodeParameters
        if isinstance(value, (ODS, CodeParameters)):
            for item in value.paths():
                self[item] = value[item]
        else:
            for item in value.keys():
                self[item] = value[item]
        return self

    def paths(self, **kw):
        """
        Traverse the code parameters and return paths that have data

        :return: list of paths that have data
        """
        paths = kw.setdefault('paths', [])
        path = kw.setdefault('path', [])
        for kid in self.keys():
            if isinstance(self.getraw(kid), CodeParameters):
                self.getraw(kid).paths(paths=paths, path=path + [kid])
            else:
                paths.append(path + [kid])
        return paths

    def keys(self):
        """
        :return: keys as list
        """
        return list(dict.keys(self))

    def values(self):
        """
        :return: values as list
        """
        return list(dict.values(self))

    def items(self):
        """
        :return: key-value pairs as list
        """
        return list(dict.items(self))

    def flat(self, **kw):
        r"""
        Flat dictionary representation of the data

        :param \**kw: extra keywords passed to the path() method

        :return: OrderedDict with flat representation of the data
        """
        tmp = OrderedDict()
        for path in self.paths(**kw):
            tmp[l2o(path)] = self[path]
        return tmp

    def from_structure(self, structure, depth=0):
        """
        Generate CodeParamters starting from a hierarchical structure made of dictionaries and lists

        :param structure: input structure

        :return: self
        """
        if isinstance(structure, dict):
            keys = list(map(str, structure.keys()))
        elif isinstance(structure, list):
            keys = list(range(len(structure)))
        else:
            raise ValueError('from_structure must be fed either a structure made of dictionaries and lists')

        for item in keys:
            if isinstance(structure[item], dict):
                self[item].from_structure(structure[item], depth=depth + 1)
            elif isinstance(structure[item], list):
                # identify if this is a leaf
                if len(structure[item]) and not isinstance(structure[item][0], dict):
                    self[item] = numpy.array(structure[item])
                # or a node in the IMAS tree
                else:
                    self[item].from_structure(structure[item], depth=depth + 1)
            else:
                self[item] = copy.deepcopy(structure[item])

        return self


def codeparams_xml_save(f):
    """
    Decorator function to be used around the omas_save_XXX methods to enable
    saving of code.parameters as an XML string
    """

    def wrapper(ods, *args, **kwargs):
        with omas_environment(ods, xmlcodeparams=True):
            return f(ods, *args, **kwargs)

    return wrapper


def codeparams_xml_load(f):
    """
    Decorator function to be used around the omas_load_XXX methods to enable
    loading of code.parameters from an XML string
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        ods = f(*args, **kwargs)
        ods.codeparams2dict()
        return ods

    return wrapper


# --------------------------------------------
# import sample functions and add them as ODS methods
# --------------------------------------------
try:
    from . import omas_sample
    from .omas_sample import ods_sample

    __all__.append('omas_sample')
    for item in omas_sample.__ods__:
        setattr(ODS, 'sample_' + item, getattr(omas_sample, item))
except ImportError as _excp:
    printe('OMAS sample function are not available: ' + repr(_excp))
    raise

# --------------------------------------------
# import physics functions and add them as ODS methods
# --------------------------------------------
try:
    from . import omas_physics
    from .omas_physics import *

    __all__.append('omas_physics')
    __all__.extend(omas_physics.__all__)
    for item in omas_physics.__ods__:
        setattr(ODS, 'physics_' + item, getattr(omas_physics, item))
except ImportError as _excp:
    printe('OMAS physics function are not available: ' + repr(_excp))

# --------------------------------------------
# import plotting functions and add them as ODS methods
# --------------------------------------------
try:
    from . import omas_plot

    __all__.append('omas_plot')
    for item in omas_plot.__ods__:
        setattr(ODS, 'plot_' + item, getattr(omas_plot, item))
except ImportError as _excp:
    printe('OMAS plotting function are not available: ' + repr(_excp))


# --------------------------------------------
# save and load OMAS with Python pickle
# --------------------------------------------
def save_omas_pkl(ods, filename, **kw):
    """
    Save ODS to Python pickle

    :param ods: OMAS data set

    :param filename: filename to save to

    :param kw: keywords passed to pickle.dump function
    """
    printd('Saving to %s' % filename, topic='pkl')

    kw.setdefault('protocol', omas_rcparams['pickle_protocol'])

    with open(filename, 'wb') as f:
        pickle.dump(ods, f, **kw)


def load_omas_pkl(filename, consistency_check=None, imas_version=None):
    """
    Load ODS or ODC from Python pickle

    :param filename: filename to save to

    :param consistency_check: verify that data is consistent with IMAS schema (skip if None)

    :param imas_version: imas version to use for consistency check (leave original if None)

    :returns: ods OMAS data set
    """
    printd('Loading from %s' % filename, topic='pkl')

    with open(filename, 'rb') as f:
        try:
            tmp = pickle.load(f)
        except UnicodeDecodeError:
            # to support ODSs created with Python2
            tmp = pickle.load(f, encoding="latin1")
    if imas_version is not None:
        tmp.imas_version = imas_version
    if consistency_check is not None:
        tmp.consistency_check = consistency_check
    return tmp


def through_omas_pkl(ods):
    """
    Test save and load Python pickle

    :param ods: ods

    :return: ods
    """
    filename = omas_testdir(__file__) + '/test.pkl'
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
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
from .omas_hdc import *
from .omas_uda import *
from .omas_h5 import *
from .omas_ds import *
from .omas_ascii import *
from .omas_mongo import *
from .omas_symbols import *
from .omas_machine import *
from . import omas_structure
