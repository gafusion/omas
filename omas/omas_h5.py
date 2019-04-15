'''save/load from HDF5 routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS


def dict2hdf5(filename, dictin, groupname='', recursive=True, lists_as_dicts=False, compression=None):
    '''
    Save hierarchy of dictionaries containing numpy-compatible objects to hdf5 file

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

        print(key)

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
            if tmp.dtype.name.lower().startswith('u'):
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


def save_omas_h5(ods, filename, **kw):
    return dict2hdf5(filename, ods, lists_as_dicts=True)
