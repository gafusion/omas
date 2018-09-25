from __future__ import print_function, division, unicode_literals

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
    'imas_json_dir', 'imas_versions', 'ids_cpo_mapper', 'omas_info', 'omas_info_node',
    'cocos_transform', 'define_cocos', 'transform_current',
    'omas_environment', 'cocos_environment', 'coords_environment',
    'omas_rcparams', 'rcparams_environment', '__version__'
]

class ODS(MutableMapping):
    """
    OMAS class
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
        self.imas_version = imas_version
        self.location = location
        self._cocos = cocos
        self.cocosio = cocosio
        self.coordsio = coordsio
        self.unitsio = unitsio
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
                extra_info['location'] = self.location + '.time'
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
            if 'location' in extra_info:
                extra_info['location']=o2u(extra_info['location'])
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
                    'time info is defined both in %s as well as upstream' % (self.location + '.time')))

        # loop over items
        for item in self.keys():
            # time (if present) is treated last
            if item == 'time':
                continue

            # identify time-dependent data
            info = omas_info_node(self.ulocation + '.' + item)
            if 'coordinates' in info and any([k.endswith('.time') for k in info['coordinates']]):

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
    def consistency_check(self, consistency_value):
        self._consistency_check = consistency_value
        for item in self.keys():
            if isinstance(self[item], ODS):
                if consistency_value:
                    structure_key = item if not isinstance(item, int) else ':'
                    if not self.structure:
                        # load the json structure file
                        structure = load_structure(item, imas_version=self.imas_version)[1][item]
                    else:
                        structure = self.structure[structure_key]
                    # assign structure and location information
                    self[item].structure = structure
                    self[item].location=l2o([self.location]+[item])

                self[item].consistency_check = consistency_value

    @property
    def cocos(self):
        """
        property that tells in what COCOS format the data is stored internally of the ODS
        (NOTE: this parameter can only be set when the object is created)

        :return: cocos value
        """
        if not hasattr(self,'_cocos'):
            self._cocos=omas_rcparams['cocos']
        return self._cocos

    @cocos.setter
    def cocos(self, cocos_value):
        raise(AttributeError('cocos parameter is readonly!'))

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
            if isinstance(self[item], ODS):
                self[item].cocosio = cocosio_value

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
            if isinstance(self[item], ODS):
                self[item].unitsio = unitsio_value

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
            if isinstance(self[item], ODS):
                self[item].coordsio = coordsio_value

    @property
    def dynamic_path_creation(self):
        """
        property that sets whether dynamic path creation is enabled or not

        :return: True/False
        """
        if not hasattr(self,'_dynamic_path_creation'):
            self._dynamic_path_creation=True
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
            if isinstance(self[item], ODS):
                self[item].dynamic_path_creation = dynamic_path_value

    def _validate(self, value, structure):
        """
        validate that the value is consistent with the provided structure field

        :param value: sub-tree to be checked

        :param structure: reference structure
        """
        for key in value.keys():
            structure_key = o2u(key)
            if isinstance(value[key], ODS) and value[key].consistency_check:
                value._validate(value[key], structure[structure_key])
            else:
                structure[structure_key]

    def __setitem__(self, key, value):
        # handle individual keys as well as full paths
        key = p2l(key)

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
            structure_key = key[0] if not isinstance(key[0],int) else ':'
            try:
                if isinstance(value, ODS):
                    if not self.structure:
                        # load the json structure file
                        structure = load_structure(key[0], imas_version=self.imas_version)[1][key[0]]
                    else:
                        structure = self.structure[structure_key]
                        if not len(structure):
                            raise(ValueError('`%s` has no data'%location))
                    # check that tha data will go in the right place
                    self._validate(value, structure)
                    # assign structure and location information
                    value.structure = structure
                    value.location = location
                else:
                    self.structure[structure_key]

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
                        spaces + '^' * len(structure_key) + '\n' + '%s' % options)

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
        if self.consistency_check and not isinstance(value, ODS):
            ulocation = o2u(location)
            # handle cocos transformations coming in
            if self.cocosio and self.cocosio != self.cocos and '.' in location and ulocation in omas_physics.cocos_signals and not isinstance(value, ODS):
                value = value * omas_physics.cocos_transform(self.cocosio, self.cocos)[omas_physics.cocos_signals[ulocation]]

            # get node information
            info = omas_info_node(ulocation)

            # handle units (Python pint package)
            if pint is not None:
                if 'units' in info and isinstance(value,pint.quantity._Quantity) or (isinstance(value,numpy.ndarray) and len(value.shape) and isinstance(value.flatten()[0],pint.quantity._Quantity)):
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
                            ods_coordinates[coordinate] = input_coordinates.__getitem__(coordinate,False)

                    # if all coordinates information is present
                    if all([coord in input_coordinates and coord in ods_coordinates for coord in coordinates]):
                        # if there is any coordinate that does not match
                        if any([len(input_coordinates.__getitem__(coord,None)) != len(ods_coordinates.__getitem__(coord,None)) or
                                (not numpy.allclose(input_coordinates.__getitem__(coord,False), ods_coordinates.__getitem__(coord,False))) for coord in coordinates]):

                            # for the time being omas interpolates only 1D quantities
                            if len(info['coordinates']) > 1:
                                raise (Exception('coordio does not support multi-dimentional interpolation just yet'))

                            # if the (first) coordinate is in input_coordinates
                            coordinate = coordinates[0]
                            if len(input_coordinates.__getitem__(coordinate,None)) != len(value):
                                raise (Exception('coordsio %s.shape=%d does not match %s.shape=%d' % (coordinate, input_coordinates.__getitem__(coordinate,False).shape, location, value.shape)))
                            printd('Adding %s interpolated to input %s coordinate'%(self.location, coordinate), topic='coordsio')
                            value = numpy.interp(ods_coordinates.__getitem__(coordinate,None),input_coordinates.__getitem__(coordinate,None), value)
                        else:
                            printd('%s ods and coordsio match'%(coordinates), topic='coordsio')
                    else:
                        printd('Adding `%s` without knowing coordinates `%s`' % (self.location, all_coordinates), topic='coordsio')

                elif ulocation in omas_coordinates(self.imas_version) and location in ods_coordinates:
                    value = ods_coordinates.__getitem__(location, None)

        # lists are saved as numpy arrays, and 0D numpy arrays as scalars
        if isinstance(value, list):
            value = numpy.array(value)
        elif isinstance(value, numpy.ndarray) and not (len(value.shape)):
            value = numpy.asscalar(value)
        elif isinstance(value, float):
            value = float(value)
        elif isinstance(value, int):
            value = int(value)

        if self.consistency_check and not isinstance(value, ODS):
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
                        raise (IndexError('%s[:] index is at %d' % (self.location, len(self) - 1)))
            try:
                self[key[0]][key[1:]] = pass_on_value
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

    def __getitem__(self, key, consistency_check=True):
        # handle individual keys as well as full paths
        key = p2l(key)

        if not len(key):
            return self

        dynamically_created = False

        # data slicing
        if key[0] == ':':
            data = []
            for k in self.keys():
                data.append(self[[k] + key[1:]])
            return numpy.array(data)

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
                raise(LookupError('Dynamic path creation is disabled, hence `%s` needs to be manually created'%location))

        value = self.omas_data[key[0]]
        if len(key) > 1:
            # if the user has entered path rather than a single key
            try:
                if isinstance(value,ODS):
                    return value.__getitem__(key[1:],consistency_check)
                else:
                    return value[l2o(key[1:])]
            except ValueError:
                if dynamically_created:
                    del self[key[0]]
                raise
        else:

            if consistency_check is not None and self.consistency_check and not isinstance(value, ODS):

                location = l2o([self.location, key[0]])
                ulocation = o2u(location)

                # handle cocos transformations going out
                if self.cocosio and self.cocosio != self.cocos and '.' in location and ulocation in omas_physics.cocos_signals:
                    value = value * omas_physics.cocos_transform(self.cocos, self.cocosio)[omas_physics.cocos_signals[ulocation]]

                # get node information
                info = omas_info_node(ulocation)

                # coordinates interpolation
                ods_coordinates, output_coordinates = self.coordsio
                if consistency_check and output_coordinates:
                    all_coordinates = []
                    coordinates = []
                    if len(output_coordinates) and 'coordinates' in info:
                        all_coordinates = list(map(lambda x: u2o(x, self.location), info['coordinates']))
                        coordinates = list(filter(lambda coord: not coord.startswith('1...'), all_coordinates))
                    if len(coordinates):
                        # if all coordinates information is present
                        if all([coord in output_coordinates and coord in ods_coordinates for coord in coordinates]):
                            # if there is any coordinate that does not match
                            if any([len(output_coordinates.__getitem__(coord,None)) != len(ods_coordinates.__getitem__(coord,None)) or
                                    (not numpy.allclose(output_coordinates.__getitem__(coord,None), ods_coordinates.__getitem__(coord,None))) for coord in coordinates]):

                                # for the time being omas interpolates only 1D quantities
                                if len(info['coordinates']) > 1:
                                    raise (Exception('coordio does not support multi-dimentional interpolation just yet'))

                                # if the (first) coordinate is in output_coordinates
                                coordinate = coordinates[0]
                                if len(ods_coordinates.__getitem__(coordinate,None)) != len(value):
                                    raise (Exception('coordsio %s.shape=%s does not match %s.shape=%s' % (coordinate, output_coordinates.__getitem__(coordinate,False).shape, location, value.shape)))
                                printd('Returning %s interpolated to output %s coordinate'%(location, coordinate), topic='coordsio')
                                value = numpy.interp(output_coordinates.__getitem__(coordinate,None), ods_coordinates.__getitem__(coordinate,None), value)
                            else:
                                printd('%s ods and coordsio match'%(coordinates), topic='coordsio')
                        else:
                            printd('Getting `%s` without knowing some of the coordinates `%s`' % (self.location, all_coordinates), topic='coordsio')

                    elif ulocation in omas_coordinates(self.imas_version) and location in output_coordinates:
                        value = output_coordinates.__getitem__(location, False)

                # handle units (Python pint package)
                if pint is not None and 'units' in info and self.unitsio:
                    value = value * getattr(ureg, info['units'])

            return value

    def __delitem__(self, key):
        # handle individual keys as well as full paths
        key = p2l(key)
        if len(key) > 1:
            # if the user has entered path rather than a single key
            del self[key[0]][key[1:]]
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

    def full_paths(self, **kw):
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
                h = h.__getitem__(k,False)
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
        copy omas_ods_attrs ['_consistency_check','_dynamic_path_creation','imas_version','location','structure','_cocos','_cocosio','_coordsio'] attributes from input ods

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
            raise (IndexError('%s has length %d and time_index %d is bejond current range' % (l2o(key), len(orig_value), time_index)))

        self[key] = numpy.atleast_1d(orig_value)
        return orig_value

    def coordinates(self):
        '''
        return dictionary with coordinates in a given ODS

        NOTE: this needs to be a dictionary and not an ODS since a given coordinates may be
        present only at certain indexes of an arrays of strucutures and an ODS cannot represent that.

        :return: dictionary with coordinates
        '''
        n = len(self.location)
        coords = {}
        for full_path in self.full_paths():
            if l2u(full_path) in omas_coordinates(self.imas_version):
                coords[l2o(full_path)] = self[l2o(full_path)[n:]]
        return coords

    def update(self, ods2):
        '''
        Adds dictionary ods2's key-values pairs in to the ods

        :param ods2: This is the dictionary to be added into the ods
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
    from .omas_physics import cocos_transform, define_cocos, transform_current
    from .omas_physics import omas_environment, cocos_environment, coords_environment
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

omas_ods_attrs=['_consistency_check','_dynamic_path_creation','imas_version','location','structure','_cocos','_cocosio','_coordsio']
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
