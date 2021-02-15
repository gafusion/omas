'''save/load from xarray dataset routines

-------
'''

from .omas_utils import *
from .omas_core import ODS, omas_environment
import itertools


class ODX(MutableMapping):
    """
    OMAS data xarray class
    """

    def __init__(self, DS=None):
        self.omas_data = DS
        self.ucache = {}
        if DS is not None:
            for k in self.omas_data.data_vars:
                self.ucache.setdefault(o2u(k), []).append(k)

    def __delitem__(self, key):
        return self.omas_data.__delitem__(key)

    def __getitem__(self, key):
        # return data if key is exactly in dataset
        if key in self.omas_data:
            return self.omas_data[key].values

        # identify which element in DS has the requested data
        ukey = None
        for k in self.ucache[o2u(key)]:
            if u2o(key, k) == k:
                ukey = k
                break

        # slice the data as requested
        tmp = self.omas_data[ukey].values
        for uk, k in zip(p2l(ukey), p2l(key)):
            if uk == ':' and isinstance(k, int):
                tmp = tmp[k]

        # return data
        return tmp

    def __getattr__(self, attr):
        # avoid picking up deepcopy and pickling methods from dataset
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            raise AttributeError('bad attribute `%s`' % attr)
        return getattr(self.omas_data, attr)

    def __iter__(self):
        return self.omas_data.__iter__()

    def __len__(self):
        return self.omas_data.__len__()

    def __setitem__(self, key, value):
        if key in self.omas_data.data_vars:
            self.omas_data[key].values[:] = value

    def save(self, *args, **kw):
        return save_omas_dx(self, *args, **kw)

    def load(self, *args, **kw):
        ods = load_omas_dx(*args, **kw)
        self.omas_data = ods.omas_data
        return self

    def to_ods(self, consistency_check=True):
        '''
        Generate a ODS from current ODX

        :param consistency_check: use consistency_check flag in ODS

        :return: ODS
        '''
        return odx_2_ods(self, consistency_check=consistency_check)


def save_omas_ds(ods, filename):
    """
    Save an ODS to xarray dataset

    :param ods: OMAS data set

    :param filename: filename or file descriptor to save to
    """
    DS = ods.dataset()
    return DS.to_netcdf(filename)


def load_omas_dx(filename, consistency_check=True):
    """
    Load ODX from xarray dataset

    :param filename: filename or file descriptor to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :return: OMAS data xarray
    """
    import xarray

    DS = xarray.open_dataset(filename)
    DS.load()
    DS.close()
    return ODX(DS)


def save_omas_dx(odx, filename):
    """
    Save an ODX to xarray dataset

    :param odx: OMAS data xarray

    :param filename: filename or file descriptor to save to
    """
    return odx.omas_data.to_netcdf(filename)


def ods_2_odx(ods, homogeneous=None):
    """
    Map ODS to an ODX

    :param ods: OMAS data set

    :param homogeneous: * False: flat representation of the ODS
                                  (data is not collected across arrays of structures)
                        * 'time': collect arrays of structures only along the time dimension
                                  (always valid for homogeneous_time=True)
                        * 'full': collect arrays of structures along all dimensions
                                  (may be valid in many situations, especially related to
                                   simulation data with homogeneous_time=True and where
                                   for example number of ions, sources, etc. do not vary)
                        * None: smart setting, uses homogeneous='time' if homogeneous_time=True else False

    :return: OMAS data xarray
    """
    return ODX(ods.dataset(homogeneous=homogeneous))


def odx_2_ods(odx, consistency_check=True):
    """
    Map ODX to ODS

    :param odx: OMAS data xarray

    :param consistency_check: verify that data is consistent with IMAS schema

    :return: OMAS data set
    """
    DS = odx.omas_data
    ods = ODS(consistency_check=False)
    with omas_environment(ods, dynamic_path_creation='dynamic_array_structures'):
        for uitem in DS.data_vars:
            depth = uitem.count(':')
            value = DS[uitem].values
            if not depth:
                ods[uitem] = value
                continue
            # unroll
            for kkk in itertools.product(*map(range, DS[uitem].shape[:depth])):
                item = uitem.replace(':', '{}').format(*kkk)
                tmp = value
                for k in kkk:
                    tmp = tmp[k]
                ods[item] = tmp
    ods.consistency_check = consistency_check
    return ods


def load_omas_ds(filename, consistency_check=True):
    """
    Load ODS from xarray dataset

    :param filename: filename or file descriptor to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :return: OMAS data set
    """
    import xarray

    DS = xarray.open_dataset(filename)
    DS.load()
    DS.close()
    odx = ODX(DS)
    ods = odx_2_ods(odx, consistency_check=consistency_check)
    return ods


def through_omas_ds(ods, method=['function', 'class_method'][1]):
    """
    Test save and load ODS via xarray file format

    :param ods: OMAS data set

    :return: OMAS data set
    """
    filename = omas_testdir(__file__) + '/test.ds'
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
    if method == 'function':
        save_omas_ds(ods, filename)
        ods1 = load_omas_ds(filename)
    else:
        ods.save(filename)
        ods1 = ODS().load(filename)
    return ods1


def through_omas_dx(odx, method=['function', 'class_method'][1]):
    """
    Test save and load OMAS data xarray via xarray file format

    :param ods: OMAS data xarray

    :return: OMAS data xarray
    """
    filename = omas_testdir(__file__) + '/test.dx'
    odx = copy.deepcopy(odx)  # make a copy to make sure save does not alter entering ODX
    if method == 'function':
        save_omas_dx(odx, filename)
        odx1 = load_omas_dx(filename)
    else:
        odx.save(filename)
        odx1 = ODX().load(filename)
    return odx1
