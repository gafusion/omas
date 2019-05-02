'''save/load from xarray dataset routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS
import itertools


def save_omas_ds(ods, filename):
    """
    Save an OMAS data set to xarray dataset

    :param ods: OMAS data set

    :param filename: filename or file descriptor to save to
    """
    DS = ods.dataset()
    return DS.to_netcdf(filename)


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
