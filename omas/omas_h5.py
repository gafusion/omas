'''save/load from HDF5 routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS


def dict2hdf5(filename, dictin, groupname='', recursive=True, lists_as_dicts=False, compression=None):
    '''
    Utility function to save hierarchy of dictionaries containing numpy-compatible objects to hdf5 file

    :param filename: hdf5 file to save to

    :param dictin: input dictionary

    :param groupname: group to save the data in

    :param recursive: traverse the dictionary

    :param lists_as_dicts: convert lists to dictionaries with integer strings

    :param compression: gzip compression level
    '''
    import h5py

    if isinstance(filename, basestring):
        with h5py.File(filename, 'w') as g:
            dict2hdf5(g, dictin, recursive=recursive, lists_as_dicts=lists_as_dicts, compression=compression)
        return
    else:
        parent = filename

    if isinstance(dictin, ODS):
        dictin = dictin.omas_data

    if groupname:
        g = parent.create_group(groupname)
    else:
        g = parent

    for key, item in list(dictin.items()):

        if isinstance(item, ODS):
            item = item.omas_data

        if isinstance(item, dict):
            if recursive:
                dict2hdf5(g, item, key, recursive=recursive, lists_as_dicts=lists_as_dicts, compression=compression)

        elif lists_as_dicts and isinstance(item, (list, tuple)) and not isinstance(item, numpy.ndarray):
            item = {'%d' % k: v for k, v in enumerate(item)}
            dict2hdf5(g, item, key, recursive=recursive, lists_as_dicts=lists_as_dicts, compression=compression)

        else:
            if item is None:
                item = '_None'
            tmp = numpy.array(item)
            if tmp.dtype.name.lower().startswith('u') or tmp.dtype.name.lower().startswith('s'):
                tmp = tmp.astype('S')
            elif tmp.dtype.name.lower().startswith('o'):
                if numpy.atleast_1d(is_uncertain(tmp)).any():
                    g.create_dataset(key + '_error_upper', std_devs(tmp).shape, dtype=std_devs(tmp).dtype, compression=compression)[...] = std_devs(tmp)
                    tmp = nominal_values(tmp)
                else:
                    continue
            if tmp.shape == ():
                g.create_dataset(key, tmp.shape, dtype=tmp.dtype)[...] = tmp
            else:
                g.create_dataset(key, tmp.shape, dtype=tmp.dtype, compression=compression)[...] = tmp

    return g


def save_omas_h5(ods, filename):
    """
    Save an OMAS data set to HDF5

    :param ods: OMAS data set

    :param filename: filename or file descriptor to save to
    """
    return dict2hdf5(filename, ods, lists_as_dicts=True)


def convertDataset(ods, data):
    '''
    Recursive utility function to map HDF5 structure to ODS

    :param ods: input ODS to be populated

    :param data: HDF5 dataset of group
    '''
    import h5py
    keys = data.keys()
    try:
        keys = sorted(list(map(int, keys)))
    except ValueError:
        pass
    for oitem in keys:
        item = str(oitem)
        if item.endswith('_error_upper'):
            continue
        if isinstance(data[item], h5py.Dataset):
            ods[item] = data[item][()]
            if item + '_error_upper' in data:
                ods[item] = uarray(ods[item], data[item + '_error_upper'][()])
        elif isinstance(data[item], h5py.Group):
            convertDataset(ods[oitem], data[item])


def load_omas_h5(filename, consistency_check=True):
    """
    Load OMAS data set from HDF5

    :param filename: filename or file descriptor to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :return: OMAS data set
    """
    import h5py
    ods = ODS(consistency_check=consistency_check)
    with h5py.File(filename, 'r') as data:
        convertDataset(ods, data)
    return ods


def through_omas_h5(ods):
    """
    Test save and load OMAS HDF5

    :param ods: ods

    :return: ods
    """
    if not os.path.exists(tempfile.gettempdir() + '/OMAS_TESTS/'):
        os.makedirs(tempfile.gettempdir() + '/OMAS_TESTS/')
    filename = tempfile.gettempdir() + '/OMAS_TESTS/test.h5'
    save_omas_h5(ods, filename)
    ods1 = load_omas_h5(filename)
    return ods1
