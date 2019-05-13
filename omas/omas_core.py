'''ODS class and save/load from pickle routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *

__version__ = open(os.path.abspath(str(os.path.dirname(__file__)) + os.sep + 'version'), 'r').read().strip()

__all__ = [
    'ODS', 'ods_sample', 'different_ods',
    'save_omas_pkl', 'load_omas_pkl', 'through_omas_pkl',
    'save_omas_json', 'load_omas_json', 'through_omas_json',
    'save_omas_hdc', 'load_omas_hdc', 'through_omas_hdc',
    'load_omas_uda',
    'save_omas_nc', 'load_omas_nc', 'through_omas_nc',
    'save_omas_h5', 'load_omas_h5', 'through_omas_h5',
    'save_omas_ds', 'load_omas_ds', 'through_omas_ds',
    'load_omas_dx',
    'save_omas_imas', 'load_omas_imas', 'through_omas_imas', 'load_omas_iter_scenario', 'browse_imas',
    'save_omas_s3', 'load_omas_s3', 'through_omas_s3', 'list_omas_s3', 'del_omas_s3',
    'generate_xml_schemas', 'create_json_structure', 'create_html_documentation',
    'imas_json_dir', 'imas_versions', 'ids_cpo_mapper', 'omas_info', 'omas_info_node',
    'omas_rcparams', 'rcparams_environment', '__version__'
]

# List of functions that can be added by third-party Python
# packages for processing input data that will go in an ODS
# This is necessary because ODSs should only contain [int (arrays), floats (arrays), strings]
# It is used for example by OMFIT to process OMFITexpressions
input_data_process_functions = []

class ODS(MutableMapping):
    """
    OMAS Data Structure class
    """

    def __init__(self,
                 imas_version=omas_rcparams['default_imas_version'],
                 consistency_check=omas_rcparams['consistency_check'],
                 dynamic_path_creation=omas_rcparams['dynamic_path_creation'],
                 location='',
                 cocos=omas_rcparams['cocos'],
                 cocosio=omas_rcparams['cocosio'],
                 coordsio=omas_rcparams['coordsio'],
                 unitsio=omas_rcparams['unitsio'],
                 structure=None):
        """
        :param imas_version: IMAS version to use as a constrain for the nodes names

        :param consistency_check: whether to enforce consistency with IMAS schema

        :param dynamic_path_creation: whether to dynamically create the path when setting an item
                                      * False: raise an error when trying to access a structure element that does not exists
                                      * True (default): arrays of structures can be incrementally extended by accessing at the next element in the array
                                      * 'dynamic_array_structures': arrays of structures can be dynamically extended

        :param location: string with location of this object relative to IMAS schema in ODS path format

        :param cocos: internal COCOS representation (this can only be set when the object is created)

        :param cocosio: COCOS representation of the data that is read/written from/to the ODS

        :param coordsio: ODS with coordinates to use for the data that is read/written from/to the ODS

        :param unitsio: ODS will return data with units if True

        :param structure: IMAS schema to use
        """
        self.omas_data = None
        self._consistency_check = consistency_check
        self._dynamic_path_creation = dynamic_path_creation
        if consistency_check and imas_version not in imas_versions:
            raise ValueError("Unrecognized IMAS version `%s`. Possible options are:\n%s" % (imas_version, imas_versions.keys()))
        self.imas_version = imas_version
        self.location = location
        self._cocos = cocos
        self.cocosio = cocosio
        self.coordsio = coordsio
        self.unitsio = unitsio
        if structure is None:
            structure = {}
        self.structure = structure

    def homogeneous_time(self, key='', default=True):
        '''
        Dynamically evaluate whether time is homogeneous or not
        NOTE: this method does not read ods['ids_properties.homogeneous_time'] instead it uses the time info to figure it out

        :param default: what to return in case no time basis is defined

        :return: True/False or default value (True) if no time basis is defined
        '''
        if not len(self.location) and not len(key):
            raise ValueError('homogeneous_time() can not be called on a top-level ODS')
        extra_info = {}
        self.time(key=key, extra_info=extra_info)
        homogeneous_time = extra_info['homogeneous_time']
        if homogeneous_time is None:
            return default
        else:
            return homogeneous_time

    def time(self, key='', extra_info=None, skip=None):
        """
        Return the time information for a given ODS location

        :param key: ods location

        :param extra_info: dictionary that will be filled in place with extra information about time

        :return: time information for a given ODS location (scalar or array)
        """

        if extra_info is None:
            extra_info = {}

        # subselect on requested key
        subtree = p2l(self.location + '.' + key)

        # get time nodes from data structure definitions
        loc = p2l(subtree)
        flat_structure, dict_structure = load_structure(loc[0], imas_version=self.imas_version)
        times_ds = [i2o(k) for k in flat_structure.keys() if k.endswith('.time')]

        # traverse ODS upstream until time information is found
        time = {}
        for sub in [subtree[:k] for k in range(len(subtree), 0, -1)]:
            times_sub_ds = [k for k in times_ds if k.startswith(l2u(sub))]
            subtree = l2o(sub)

            # get time data from ods
            times = {}
            n = len(self.location)
            for item in times_sub_ds:
                try:
                    time = self.__getitem__(u2o(item, subtree)[n:], None)  # traverse ODS
                    if isinstance(time, numpy.ndarray):
                        if time.size == 0:
                            continue
                        elif len(time.shape) > 1:
                            time = numpy.atleast_1d(numpy.squeeze(time))
                    times[item] = time
                except IndexError:
                    pass
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
                time = None
                extra_info['homogeneous_time'] = None
            # if there is a single time entry, or there are multiple time entries that are all consistent with one another
            elif len(times) == 1 or all([times_values[0].shape == time.shape and numpy.allclose(times_values[0], time) for time in times_values[1:]]):
                time = times_values[0]
                extra_info['homogeneous_time'] = True
                return time
            # there are inconsistencies with different ways of specifying times in the IDS
            else:
                raise ValueError('Inconsistent time definitions in %s' % times.keys())

        return None

    def slice_at_time(self, time=None, time_index=None):
        '''
        method for selecting a time slice from an time-dependent ODS (NOTE: this method operates in place)

        :param time: time value to select

        :param time_index: time index to select (NOTE: time_index has precedence over time)

        :return: modified ODS
        '''

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
            if 'coordinates' in info and any([k.endswith('.time') for k in info['coordinates']]):

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
    def consistency_check(self):
        """
        property that sets whether consistency with IMAS schema is enabled or not

        :return: True/False
        """
        if not hasattr(self, '_consistency_check'):
            self._consistency_check = omas_rcparams['consistency_check']
        return self._consistency_check

    @consistency_check.setter
    def consistency_check(self, consistency_value):
        old_consistency_check = self._consistency_check
        try:
            # set .consistency_check for this ODS
            self._consistency_check = consistency_value
            # set .consistency_check and assign the .structure/.location attributes to the underlying ODSs
            for item in self.keys():
                if isinstance(self.getraw(item), ODS):
                    if consistency_value:
                        if not self.structure:
                            if not self.location:
                                # load the json structure file
                                structure = load_structure(item, imas_version=self.imas_version)[1][item]
                            else:
                                raise RuntimeError('When switching from False to True .consistency_check=True must be set at the top-level ODS')
                        else:
                            structure_key = item if not isinstance(item, int) else ':'
                            structure = self.structure[structure_key]
                        # assign structure and location information
                        self.getraw(item).structure = structure
                        self.getraw(item).location = l2o([self.location] + [item])
                    self.getraw(item).consistency_check = consistency_value

        except Exception as _excp:
            # restore existing consistency_check value in case of error
            if old_consistency_check != consistency_value:
                for item in self.keys():
                    if isinstance(self.getraw(item), ODS):
                        self.getraw(item).consistency_check = old_consistency_check
                self.consistency_check = old_consistency_check
            raise  # (LookupError('Consistency check failed: %s' % repr(_excp)))

    @property
    def cocos(self):
        """
        property that tells in what COCOS format the data is stored internally of the ODS
        (NOTE: this parameter can only be set when the object is created)

        :return: cocos value
        """
        if not hasattr(self, '_cocos'):
            self._cocos = omas_rcparams['cocos']
        return self._cocos

    @cocos.setter
    def cocos(self, cocos_value):
        raise AttributeError('cocos parameter is readonly!')

    @property
    def cocosio(self):
        """
        property that tells in what COCOS format the data will be input/output

        :return: cocosio value
        """
        if not hasattr(self, '_cocosio'):
            self._cocosio = omas_rcparams['cocosio']
        return self._cocosio

    @cocosio.setter
    def cocosio(self, cocosio_value):
        self._cocosio = cocosio_value
        for item in self.keys():
            if isinstance(self.getraw(item), ODS):
                self.getraw(item).cocosio = cocosio_value

    @property
    def unitsio(self):
        """
        property that if data should be returned with units or not

        :return: unitsio value
        """
        if not hasattr(self, '_unitsio'):
            self._unitsio = omas_rcparams['unitsio']
        return self._unitsio

    @unitsio.setter
    def unitsio(self, unitsio_value):
        self._unitsio = unitsio_value
        for item in self.keys():
            if isinstance(self.getraw(item), ODS):
                self.getraw(item).unitsio = unitsio_value

    @property
    def coordsio(self):
        """
        property that tells in what COCOS format the data will be input/output

        :return: coordsio value
        """
        if not hasattr(self, '_coordsio'):
            self._coordsio = (None, omas_rcparams['coordsio'])
        return self._coordsio

    @coordsio.setter
    def coordsio(self, coordsio_value):
        if not isinstance(coordsio_value, (list, tuple)):
            coordsio_value = (self, coordsio_value)
        self._coordsio = coordsio_value
        for item in self.keys():
            if isinstance(self.getraw(item), ODS):
                self.getraw(item).coordsio = coordsio_value

    @property
    def dynamic_path_creation(self):
        """
        property that sets whether dynamic path creation is enabled or not

        :return: True/False
        """
        if not hasattr(self, '_dynamic_path_creation'):
            self._dynamic_path_creation = True
        return self._dynamic_path_creation

    @property
    def ulocation(self):
        '''
        :return: string with location of this object in universal ODS path format
        '''
        return o2u(self.location)

    @dynamic_path_creation.setter
    def dynamic_path_creation(self, dynamic_path_value):
        self._dynamic_path_creation = dynamic_path_value
        for item in self.keys():
            if isinstance(self.getraw(item), ODS):
                self.getraw(item).dynamic_path_creation = dynamic_path_value

    def _validate(self, value, structure):
        """
        Validate that the value is consistent with the provided structure field

        :param value: sub-tree to be checked

        :param structure: reference structure
        """
        for key in value.keys():
            structure_key = o2u(key)
            if isinstance(value[key], ODS) and value[key].consistency_check:
                value._validate(value[key], structure[structure_key])
            else:
                structure[structure_key]

    def set_child_locations(self):
        '''
        traverse ODSs and set .location attribute
        '''
        for item in self.keys():
            if isinstance(self.getraw(item), ODS):
                self.getraw(item).location = l2o([self.location, item])
                self.getraw(item).set_child_locations()

    def __setitem__(self, key, value):
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
        elif key[0] == '+':
            if self.omas_data is None:
                key[0] = 0
            elif isinstance(self.omas_data, list):
                key[0] = len(self.omas_data)

        # if the user has entered path rather than a single key
        if len(key) > 1:
            pass_on_value = value
            value = ODS(imas_version=self.imas_version,
                        consistency_check=self.consistency_check,
                        dynamic_path_creation=self.dynamic_path_creation,
                        cocos=self.cocos, cocosio=self.cocosio, coordsio=self.coordsio)

        # full path where we want to place the data
        location = l2o([self.location, key[0]])

        if self.consistency_check:
            # perform consistency check with IMAS structure
            structure = {}
            structure_key = key[0] if not isinstance(key[0], int) else ':'
            try:
                if isinstance(value, ODS):
                    if not self.structure:
                        # load the json structure file
                        structure = load_structure(key[0], imas_version=self.imas_version)[1][key[0]]
                    else:
                        structure = self.structure[structure_key]
                        if not len(structure):
                            raise ValueError('`%s` has no data' % location)
                    # check that tha data will go in the right place
                    self._validate(value, structure)
                    # assign structure and location information
                    value.structure = structure
                    # determine if entry is a list of structures or just a structure
                    if value.omas_data is None:
                        if ':' in value.structure.keys():
                            value.omas_data = []
                        elif len(value.structure.keys()):
                            value.omas_data = {}
                    value.location = location
                else:
                    self.structure[structure_key]

            except (LookupError, TypeError):
                if self.consistency_check == 'warn':
                    printe('`%s` is not a valid IMAS %s location' % (location, self.imas_version))
                    if isinstance(value, ODS):
                        value.consistency_check = False
                elif self.consistency_check:
                    options = list(self.structure.keys())
                    if len(options) == 1 and options[0] == ':':
                        options = 'A numerical index is needed with n>=0'
                    else:
                        options = 'Did you mean: %s' % options
                    spaces = ' ' * len('LookupError') + '  ' + ' ' * (len(self.location) + 2)
                    raise LookupError('`%s` is not a valid IMAS %s location\n' % (location, self.imas_version) +
                                      spaces + '^' * len(structure_key) + '\n' + '%s' % options)

        # check what container type is required and if necessary switch it
        if isinstance(key[0], int) and not isinstance(self.omas_data, list):
            if not self.omas_data or not len(self.omas_data):
                self.omas_data = []
            else:
                raise Exception('Cannot convert from dict to list once ODS has data')
        if not isinstance(key[0], int) and not isinstance(self.omas_data, dict):
            if not self.omas_data or not len(self.omas_data):
                self.omas_data = {}
            else:
                raise Exception('Cannot convert from list to dict once ODS has data')

        # if the value is not an ODS strucutre
        if not isinstance(value, ODS):

            # now that all checks are completed we can assign the structure information
            if self.consistency_check:
                ulocation = o2u(location)

                # handle cocos transformations coming in
                if self.cocosio and self.cocosio != self.cocos and '.' in location and ulocation in omas_physics.cocos_signals and not isinstance(value, ODS):
                    transform = omas_physics.cocos_signals[ulocation]
                    if transform == '?':
                        if self.consistency_check == 'warn':
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
                    if 'units' in info and isinstance(value, pint.quantity._Quantity) or (isinstance(value, numpy.ndarray) and value.size and isinstance(numpy.atleast_1d(value).flat[0], pint.quantity._Quantity)):
                        value = value.to(info['units']).magnitude

                # coordinates interpolation
                ods_coordinates, input_coordinates = self.coordsio
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
                        if all([coord in input_coordinates and coord in ods_coordinates for coord in coordinates]):
                            # if there is any coordinate that does not match
                            if any([len(input_coordinates.__getitem__(coord, None)) != len(ods_coordinates.__getitem__(coord, None)) or
                                    (not numpy.allclose(input_coordinates.__getitem__(coord, False), ods_coordinates.__getitem__(coord, False))) for coord in coordinates]):

                                # for the time being omas interpolates only 1D quantities
                                if len(info['coordinates']) > 1:
                                    raise Exception('coordio does not support multi-dimentional interpolation just yet')

                                # if the (first) coordinate is in input_coordinates
                                coordinate = coordinates[0]
                                if len(input_coordinates.__getitem__(coordinate, None)) != len(value):
                                    raise Exception('coordsio %s.shape=%s does not match %s.shape=%s' % (coordinate, input_coordinates.__getitem__(coordinate, False).shape, location, value.shape))
                                printd('Adding %s interpolated to input %s coordinate' % (self.location, coordinate), topic='coordsio')
                                value = omas_interp1d(ods_coordinates.__getitem__(coordinate, None), input_coordinates.__getitem__(coordinate, None), value)
                            else:
                                printd('%s ods and coordsio match' % (coordinates), topic='coordsio')
                        else:
                            printd('Adding `%s` without knowing coordinates `%s`' % (self.location, all_coordinates), topic='coordsio')

                    elif ulocation in omas_coordinates(self.imas_version) and location in ods_coordinates:
                        value = ods_coordinates.__getitem__(location, None)

            # lists are saved as numpy arrays, and 0D numpy arrays as scalars
            for function in input_data_process_functions:
                value = function(value)
            if isinstance(value, list):
                value = numpy.array(value)
            if isinstance(value, xarray.DataArray):
                value = value.values
            if isinstance(value, numpy.ndarray) and not (len(value.shape)):
                value = value.item()
            if isinstance(value, (float, numpy.floating)):
                value = float(value)
            elif isinstance(value, (int, numpy.integer)):
                value = int(value)
            elif isinstance(value, basestring):
                pass
            elif isinstance(value, bytes):
                value = value.tolist().decode('utf-8', errors='ignore')

            if self.consistency_check:
                # check type
                if not (isinstance(value, (int, float, unicode, str, numpy.ndarray, uncertainties.core.Variable)) or value is None):
                    raise ValueError('Trying to write %s in %s\nSupported types are: string, float, int, array' % (type(value), location))

                # check consistency for scalar entries
                if 'data_type' in info and '_0D' in info['data_type'] and isinstance(value, numpy.ndarray):
                    printe('%s must be a scalar of type %s' % (location, info['data_type']))
                # check consistency for number of dimensions
                elif 'coordinates' in info and len(info['coordinates']) and (not isinstance(value, numpy.ndarray) or len(value.shape) != len(info['coordinates'])):
                    # may want to raise a ValueError in the future
                    printe('%s must be an array with dimensions: %s' % (location, info['coordinates']))
                elif 'lifecycle_status' in info and info['lifecycle_status'] in ['obsolescent']:
                    printe('%s is in %s state' % (location, info['lifecycle_status'].upper()))

        # if the user has entered a path rather than a single key
        if len(key) > 1:
            dynamically_created = False
            if key[0] not in self.keys():
                dynamically_created = True
                if isinstance(self.omas_data, dict):
                    self.omas_data[key[0]] = value
                else:
                    if key[0] >= len(self.omas_data) and self.dynamic_path_creation == 'dynamic_array_structures':
                        for item in range(len(self.omas_data), key[0]):
                            ods = ODS()
                            ods.copy_attrs_from(self)
                            self[item] = ods
                    if key[0] == len(self.omas_data):
                        self.omas_data.append(value)
                    else:
                        raise IndexError('`%s[%d]` but maximun index is %d' % (self.location, key[0], len(self.omas_data) - 1))
            try:
                self.getraw(key[0])[key[1:]] = pass_on_value
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
            if not len(self.omas_data):
                IndexError('`%s[%d]` but ods has no data' % (self.location, key[0]))
            else:
                raise IndexError('`%s[%d]` but maximun index is %d' % (self.location, key[0], len(self.omas_data) - 1))

        # if the value is an ODS strucutre
        if isinstance(value, ODS) and value.omas_data is not None and len(value):
            # make sure entries have the right location
            self.set_child_locations()

    def getraw(self, key):
        '''
        Method to access data stored in ODS with no processing of the key, and it is thus faster than the ODS.__getitem__(key)
        Effectively behaves like a pure Python dictionary/list __getitem__.
        This method is mostly meant to be used in the inner workings of the ODS class.
        NOTE: ODS.__getitem__(key, False) can be used to access items in the ODS with disabled cocos and coordinates processing but with support for different syntaxes to access data

        :param key: string or integer

        :return: ODS value
        '''

        return self.omas_data[key]

    def setraw(self, key, value):
        '''
        Method to assign data to an ODS with no processing of the key, and it is thus faster than the ODS.__setitem__(key, value)
        Effectively behaves like a pure Python dictionary/list __setitem__.
        This method is mostly meant to be used in the inner workings of the ODS class.

        :param key: string or integer

        :param value: value to assign

        :return: value
        '''

        self.omas_data[key] = value
        return value

    def __getitem__(self, key, cocos_and_coords=True):
        '''
        ODS getitem method allows support for different syntaxes to access data

        :param key: different syntaxes to access data, for example:
              * ods['equilibrium']['time_slice'][0]['profiles_2d'][0]['psi']   # standard Python dictionary syntax
              * ods['equilibrium.time_slice[0].profiles_2d[0].psi']            # IMAS hierarchical tree syntax
              * ods['equilibrium.time_slice.0.profiles_2d.0.psi']              # dot separated string syntax
              * ods[['equilibrium','time_slice',0,'profiles_2d',0,'psi']]      # list of nodes syntax

        :param cocos_and_coords: processing of cocos transforms and coordinates interpolations [True/False/None]
              * True: enabled COCOS and enabled interpolation
              * False: enabled COCOS and disabled interpolation
              * None: disabled COCOS and disabled interpolation

        :return: ODS value
        '''

        # handle pattern match
        if isinstance(key, basestring) and key.startswith('@'):
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

        dynamically_created = False

        # data slicing
        if key[0] == ':':
            data = []
            for k, item in enumerate(self.keys()):
                try:
                    data.append(self.__getitem__([item] + key[1:], cocos_and_coords))
                except ValueError:
                    data.append([])
            # handle missing data by filling out with NaNs
            valid = _empty = []
            for k, item in enumerate(data):
                if (isinstance(item, list) and not len(item)) or (isinstance(item, numpy.ndarray) and not item.size):
                    _empty.append(k)
                else:
                    valid = item
            if valid is not _empty and len(_empty):
                for k in _empty:
                    data[k] = valid * numpy.nan
            # force dtype to avoid obtaining arrays of objects in case
            # the shape of the concatenated arrays do not match
            return numpy.array(data, dtype=numpy.array(data[0]).dtype)

        # dynamic path creation
        elif key[0] not in self.keys():
            if self.dynamic_path_creation:
                dynamically_created = True
                self.__setitem__(key[0], ODS(imas_version=self.imas_version,
                                             consistency_check=self.consistency_check,
                                             dynamic_path_creation=self.dynamic_path_creation,
                                             cocos=self.cocos, cocosio=self.cocosio, coordsio=self.coordsio))
            else:
                location = l2o([self.location, key[0]])
                raise LookupError('Dynamic path creation is disabled, hence `%s` needs to be manually created' % location)

        value = self.omas_data[key[0]]
        if len(key) > 1:
            # if the user has entered path rather than a single key
            try:
                if isinstance(value, ODS):
                    return value.__getitem__(key[1:], cocos_and_coords)
                else:
                    return value[l2o(key[1:])]
            except ValueError:
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
                ods_coordinates, output_coordinates = self.coordsio
                if cocos_and_coords and output_coordinates:
                    all_coordinates = []
                    coordinates = []
                    if len(output_coordinates) and 'coordinates' in info:
                        all_coordinates = list(map(lambda x: u2o(x, self.location), info['coordinates']))
                        coordinates = list(filter(lambda coord: not coord.startswith('1...'), all_coordinates))
                    if len(coordinates):
                        # if all coordinates information is present
                        if all([coord in output_coordinates and coord in ods_coordinates for coord in coordinates]):
                            # if there is any coordinate that does not match
                            if any([len(output_coordinates.__getitem__(coord, None)) != len(ods_coordinates.__getitem__(coord, None)) or
                                    (not numpy.allclose(output_coordinates.__getitem__(coord, None), ods_coordinates.__getitem__(coord, None))) for coord in coordinates]):

                                # for the time being omas interpolates only 1D quantities
                                if len(info['coordinates']) > 1:
                                    raise Exception('coordio does not support multi-dimentional interpolation just yet')

                                # if the (first) coordinate is in output_coordinates
                                coordinate = coordinates[0]
                                if len(ods_coordinates.__getitem__(coordinate, None)) != len(value):
                                    raise Exception('coordsio %s.shape=%s does not match %s.shape=%s' % (coordinate, output_coordinates.__getitem__(coordinate, False).shape, location, value.shape))
                                printd('Returning %s interpolated to output %s coordinate' % (location, coordinate), topic='coordsio')
                                try:
                                    value = omas_interp1d(output_coordinates.__getitem__(coordinate, None), ods_coordinates.__getitem__(coordinate, None), value)
                                except TypeError:
                                    if numpy.atleast_1d(is_uncertain(value)).any():
                                        v = omas_interp1d(output_coordinates.__getitem__(coordinate, None), ods_coordinates.__getitem__(coordinate, None), nominal_values(value))
                                        s = omas_interp1d(output_coordinates.__getitem__(coordinate, None), ods_coordinates.__getitem__(coordinate, None), std_devs(value))
                                        value = unumpy.uarray(v, s)
                            else:
                                printd('%s ods and coordsio match' % (coordinates), topic='coordsio')
                        else:
                            printd('Getting `%s` without knowing some of the coordinates `%s`' % (self.location, all_coordinates), topic='coordsio')

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

    def pretty_paths(self):
        """
        Traverse the ods and return paths that have data formatted nicely

        :return: list of paths that have data formatted nicely
        """
        return list(map(l2i, self.paths()))

    def paths(self, **kw):
        """
        Traverse the ods and return paths that have data

        :return: list of paths that have data
        """
        paths = kw.setdefault('paths', [])
        path = kw.setdefault('path', [])
        for kid in self.keys():
            if isinstance(self.getraw(kid), ODS):
                self.getraw(kid).paths(paths=paths, path=path + [kid])
            else:
                paths.append(path + [kid])
        return paths

    def full_paths(self):
        """
        Traverse the ods and return paths from root of ODS that have data

        :return: list of paths that have data
        """
        location = p2l(self.location)
        return [location + path for path in self.paths()]

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
        return (False, True)

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
                h = h.__getitem__(k, False)
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
            return list(map(str, self.omas_data.keys()))
        elif isinstance(self.omas_data, list):
            return list(range(len(self.omas_data)))
        else:
            return []

    def values(self):
        return [self[item] for item in self.keys()]

    def __str__(self):
        return self.location

    def __repr__(self):
        return repr(self.omas_data)

    def __tree_repr__(self):
        '''
        OMFIT tree representation
        '''
        if not self.location:
            return self, []
        s = '--{%d}--' % len(self)
        if 'dataset_description.data_entry' in self:
            s += ' ' + ' '.join(['%s:%s' % (k, v) for k, v in self['dataset_description']['data_entry'].items() if v not in ['None', None]])
        if 'summary.ids_properties.comment' in self:
            s += ' ' + repr(self['summary.ids_properties.comment'])
        return s, []

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
        state = {}
        for item in omas_dictstate:
            if item in self.__dict__:
                state[item] = self.__dict__[item]
        return state

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
            self.omas_data[:] = []
        return self

    def copy_attrs_from(self, ods):
        '''
        copy omas_ods_attrs ['_consistency_check','_dynamic_path_creation','imas_version','location','structure','_cocos','_cocosio','_coordsio'] attributes from input ods

        :param ods: input ods

        :return: self
        '''
        for item in omas_ods_attrs:
            setattr(self, item, getattr(ods, item))
        return self

    def prune(self):
        '''
        Prune ODS branches that are leafless

        :return: number of branches that were pruned
        '''
        n = 0
        for item in self.keys():
            if isinstance(self.getraw(item), ODS):
                n += self.getraw(item).prune()
                if not len(self.getraw(item).keys()):
                    n += 1
                    del self[item]
        return n

    def set_time_array(self, key, time_index, value):
        '''
        Convenience function for setting time dependent arrays

        :param key: ODS location to edit

        :param time_index: time index of the value to set

        :param value: value to set

        :return: time dependent array
        '''

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
        '''
        Adds ods2's key-values pairs to the ods

        :param ods2: dictionary or ODS to be added into the ODS
        '''
        if isinstance(ods2, ODS):
            for item in ods2.paths():
                self[item] = ods2[item]
        else:
            try:
                bkp_dynamic_path_creation = self.dynamic_path_creation
                self.dynamic_path_creation = 'dynamic_array_structures'
                for item in ods2.keys():
                    self[item] = ods2[item]
            finally:
                self.dynamic_path_creation = bkp_dynamic_path_creation

    def list_coordinates(self):
        '''
        return dictionary with coordinates in a given ODS

        :return: dictionary with coordinates (keys are absolute location, values are relative locations)
        '''
        coords = {}

        n = len(self.location)
        for full_path in self.full_paths():
            if l2u(full_path) in omas_coordinates(self.imas_version):
                coords[l2o(full_path)] = self[l2o(full_path)[n:]]

        return coords

    def coordinates(self, key):
        '''
        return dictionary with coordinates of a given ODS location

        :param key: ODS location to return the coordinates of
                    Note: both the key location and coordinates must have data

        :return: OrderedDict with coordinates of a given ODS location
        '''
        coords = OrderedDict()

        self.__getitem__(key, False)  # raise an error if data is not there
        ulocation = l2u(p2l(key))
        info = omas_info_node(ulocation)
        if 'coordinates' not in info:
            raise ValueError('ODS location `%s` has no coordinates information' % ulocation)
        coordinates = map(lambda x: u2o(x, key), info['coordinates'])
        for coord in coordinates:
            coords[coord] = self[coord]  # this will raise an error if the coordinates data is not there

        return coords

    def search_paths(self, search_pattern, n=None, regular_expression_startswith=''):
        '''
        Find ODS locations that match a pattern

        :param search_pattern: regular expression ODS location string

        :param n: raise an error if a numbe of occurrences different from n is found

        :param regular_expression_startswith: indicates that use of regular expressions
               in the search_pattern is preceeded by certain characters.
               This is used internally by some methods of the ODS to force users
               to use '@' to indicate access to a path by regular expression.

        :return: list of ODS locations matching search_pattern pattern
        '''
        if not isinstance(search_pattern, basestring):
            return [search_pattern]

        elif regular_expression_startswith:
            if not search_pattern.startswith(regular_expression_startswith):
                return [search_pattern]
            else:
                search_pattern = search_pattern[len(regular_expression_startswith):]

        search = re.compile(search_pattern)
        matches = []
        for path in map(l2o, self.full_paths()):
            if re.match(search, path):
                matches.append(path)
        if n is not None and len(matches) != n:
            raise ValueError('Found %d matches of `%s` instead of the %d requested\n%s' % (len(matches), search_pattern, n, '\n'.join(matches)))
        return matches

    def xarray(self, key):
        '''
        Returns data of an ODS location and correspondnig coordinates as an xarray dataset
        Note that the Dataset and the DataArrays have their attributes set with the ODSs structure info

        :param key: ODS location

        :return: xarray dataset
        '''
        key = self.search_paths(key, 1, '@')[0]

        import xarray
        key = p2l(key)

        info = omas_info_node(l2u(key))
        coords = self.coordinates(key)

        short_coords = OrderedDict()
        for coord in coords:
            short_coords[p2l(coord)[-1]] = coords[coord]

        ds = xarray.Dataset()
        ds[key[-1]] = xarray.DataArray(self[key], coords=short_coords, dims=short_coords.keys(), attrs=info)
        ds.attrs['y'] = key[-1]
        ds.attrs['y_full'] = l2o(key)

        ds.attrs['x'] = []
        ds.attrs['x_full'] = []
        for coord in coords:
            info = omas_info_node(o2u(coord))
            ds[p2l(coord)[-1]] = xarray.DataArray(coords[coord], dims=p2l(coord)[-1], attrs=info)
            ds.attrs['x'].append(p2l(coord)[-1])
            ds.attrs['x_full'].append(coord)
        return ds

    def dataset(self, homogeneous=[False, 'time', 'full', None][-1]):
        '''
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
        '''
        import xarray

        if not self.location:
            DS = xarray.Dataset()
            for ds in self:
                DS.update(self[ds].dataset(homogeneous=homogeneous))
            return DS

        def arraystruct_indexnames(key):
            '''
            return list of strings with a name for each of the arrays of structures indexes

            :param key: ods location

            :return: list of strings
            '''
            base = key.split('.')[0]
            coordinates = []
            for c in [':'.join(key.split(':')[:k + 1]).strip('.') for k, struct in enumerate(key.split(':'))]:
                info = omas_info_node(o2u(c))
                if 'coordinates' in info:
                    for infoc in info['coordinates']:
                        if infoc == '1...N':
                            infoc = c
                        coordinates.append('_'.join([base, infoc.split('.')[-1]] + [str(k) for k in p2l(key) if isinstance(k, int) or k == ':'] + ['index']))
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
            raise ValueError("OMAS dataset homogeneous attribute can only be [False, None, 'time', 'full']")
        upaths = fupaths
        if self.location:
            n = len(self.location)
            upaths = list(map(lambda key: key[n + 1:], fupaths))

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
                printe('WARNING: %s is not part of IMAS' % o2i(fukey))
                continue
            data = self[ukey]  # OMAS data slicing at work
            for k, c in enumerate(coordinates[fukey]):
                if c not in DS:
                    DS[c] = xarray.DataArray(numpy.arange(data.shape[k]), dims=c)
            DS[fukey] = xarray.DataArray(data, dims=coordinates[fukey])
        return DS

    def satisfy_imas_requirements(self, attempt_fix=True, raise_errors=True):
        """
        Assign .time and .ids_properties.homogeneous_time info for top-level structures
        since these are required for writing an IDS to IMAS

        :param attempt_fix: fix dataset_description and wall IDS to have 0 times if none is set

        :param raise_errors: raise errors if could not satisfy IMAS requirements

        :return: `True` if all is good, `False` if requirements are not satisfied, `None` if fixes were applied
        """

        # if called at top level, loop over all data structures
        if not len(self.location):
            out = []
            for ds in self:
                out.append(self.getraw(ds).satisfy_imas_requirements(attempt_fix=attempt_fix, raise_errors=raise_errors))
            if any([k is False for k in out]):
                return False
            elif any([k is None for k in out]):
                return None
            else:
                return True

        ds = p2l(self.location)[0]

        if ds in add_datastructures:
            return True

        extra_info = {}
        time = self.time(extra_info=extra_info)
        if extra_info['homogeneous_time'] is False:
            self['ids_properties']['homogeneous_time'] = extra_info['homogeneous_time']
        elif time is not None and len(time):
            self['time'] = time
            self['ids_properties']['homogeneous_time'] = extra_info['homogeneous_time']
        elif attempt_fix and ds in ['dataset_description', 'wall']:
            self['time'] = [0.0]
            self['ids_properties']['homogeneous_time'] = extra_info['homogeneous_time']
            return None
        elif raise_errors:
            raise ValueError(self.location + '.time cannot be automatically filled! Missing time information in the data structure.')
        else:
            return False
        return True


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
try:
    # --------------------------------------------
    from . import omas_plot

    __all__.append('omas_plot')
    for item in omas_plot.__ods__:
        setattr(ODS, 'plot_' + item, getattr(omas_plot, item))
except ImportError as _excp:
    printe('OMAS plotting function are not available: ' + repr(_excp))

omas_ods_attrs = ['_consistency_check', '_dynamic_path_creation', 'imas_version', 'location', 'structure', '_cocos', '_cocosio', '_coordsio']
omas_dictstate = dir(ODS)
omas_dictstate.extend(['omas_data'] + omas_ods_attrs)
omas_dictstate = sorted(list(set(omas_dictstate)))


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
    Test save and load Python pickle

    :param ods: ods

    :return: ods
    """
    if not os.path.exists(tempfile.gettempdir() + '/OMAS_TESTS/'):
        os.makedirs(tempfile.gettempdir() + '/OMAS_TESTS/')
    filename = tempfile.gettempdir() + '/OMAS_TESTS/test.pkl'
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
from .omas_hdc import *
from .omas_uda import *
from .omas_h5 import *
from .omas_ds import *


# --------------------------------------------
# backward compatibility
# --------------------------------------------
def save_omas_itm(*args, **kw):
    '''
    Dummy function to avoid older versions of OMFIT not to start with latest OMAS
    '''
    raise NotImplementedError('omas itm functions are deprecated')


load_omas_itm = through_omas_itm = save_omas_itm
__all__.extend(['save_omas_itm', 'load_omas_itm', 'through_omas_itm'])
