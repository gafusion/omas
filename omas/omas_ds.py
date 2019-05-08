'''save/load from xarray dataset routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS
import itertools


class ODX(MutableMapping):
    """
    OMAX Data Structure class
    """
    def __init__(self, DS):
        self.omas_data = DS
        self.ucache = {}
        for k in self.omas_data.data_vars:
            self.ucache.setdefault(o2u(k),[]).append(k)

    def __delitem__(self, key):
        pass

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

    def __iter__(self):
        pass

    def __len__(self):
        return None

    def __setitem__(self, key, value):
        if key in self.omas_data.data_vars:
            self.omas_data[key].values[:] = value


def save_omas_ds(ods, filename):
    """
    Save an OMAS data set to xarray dataset

    :param ods: OMAS data set

    :param filename: filename or file descriptor to save to
    """
    DS = ods.dataset()
    return DS.to_netcdf(filename)


def load_omas_dx(filename, consistency_check=True):
    """
    Load OMAS data set from xarray dataset

    :param filename: filename or file descriptor to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :return: OMAS data set
    """
    DS = xarray.open_dataset(filename)
    DS.load()
    DS.close()
    return ODX(DS)


def load_omas_ds(filename, consistency_check=True):
    """
    Load OMAS data set from xarray dataset

    :param filename: filename or file descriptor to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :return: OMAS data set
    """
    DS = xarray.open_dataset(filename)
    DS.load()
    DS.close()

    # map xarray dataset to ODS
    ods = ODS(consistency_check=False)
    ods.dynamic_path_creation = 'dynamic_array_structures'
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
    ods.dynamic_path_creation = True
    return ods


def through_omas_ds(ods):
    """
    Test save and load OMAS HDF5

    :param ods: ods

    :return: ods
    """
    if not os.path.exists(tempfile.gettempdir() + '/OMAS_TESTS/'):
        os.makedirs(tempfile.gettempdir() + '/OMAS_TESTS/')
    filename = tempfile.gettempdir() + '/OMAS_TESTS/test.xr'
    save_omas_ds(ods, filename)
    ods1 = load_omas_ds(filename)
    return ods1
