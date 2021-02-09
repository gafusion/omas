'''save/load from NC routines

-------
'''

from .omas_utils import *
from .omas_core import ODS, ODC, dynamic_ODS


# --------------------------------------------
# save and load OMAS with NetCDF
# --------------------------------------------
def save_omas_nc(ods, filename, **kw):
    """
    Save an ODS to NetCDF file

    :param ods: OMAS data set

    :param filename: filename to save to

    :param kw: arguments passed to the netCDF4 Dataset function
    """
    printd('Saving to %s' % filename, topic='nc')

    from netCDF4 import Dataset

    odsf = ods.flat()
    with Dataset(filename, 'w', **kw) as dataset:
        for item in odsf:
            dims = []
            data = numpy.asarray(odsf[item])
            std = None
            if is_uncertain(odsf[item]):
                std = std_devs(data)
                data = nominal_values(data)
            for k in range(len(numpy.asarray(odsf[item]).shape)):
                dims.append('dim_%d' % (data.shape[k]))
                if dims[-1] not in dataset.dimensions:
                    dataset.createDimension(dims[-1], data.shape[k])
            if std is None:
                dataset.createVariable(item, data.dtype, dims)[:] = data
            else:
                dataset.createVariable(item, data.dtype, dims)[:] = data
                dataset.createVariable(item + '_error_upper', data.dtype, dims)[:] = std


def get_ds_item(dataset, item):
    """
    Convenience function for loading OMAS data stored in a NC file variable
    Handles arrays, scalars, strings, and uncertain quantities

    :param dataset: nc dataset

    :param item: variable name

    :return: data
    """
    if dataset.variables[item].shape:
        # arrays
        if item + '_error_upper' in dataset.variables.keys():
            tmp = uarray(numpy.array(dataset.variables[item]), numpy.array(dataset.variables[item + '_error_upper']))
        else:
            tmp = numpy.array(dataset.variables[item])
    else:
        # uncertain scalars
        if item + '_error_upper' in dataset.variables.keys():
            tmp = ufloat(dataset.variables[item][0].item(), dataset.variables[item + '_error_upper'][0].item())
        else:
            try:
                # scalars
                tmp = dataset.variables[item][0].item()
            except AttributeError:
                # strings
                tmp = dataset.variables[item][0]
    return tmp


def load_omas_nc(filename, consistency_check=True, imas_version=omas_rcparams['default_imas_version'], cls=ODS):
    """
    Load ODS or ODC from NetCDF file

    :param filename: filename to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :param imas_version: imas version to use for consistency check

    :param cls: class to use for loading the data

    :return: OMAS data set
    """
    printd('Loading from %s' % filename, topic='nc')

    from netCDF4 import Dataset

    ods = cls(imas_version=imas_version, consistency_check=False)
    with Dataset(filename, 'r') as dataset:
        for item in dataset.variables.keys():
            if item.endswith('_error_upper'):
                continue
            ods.setraw(p2l(item), get_ds_item(dataset, item))
    ods.consistency_check = consistency_check
    return ods


class dynamic_omas_nc(dynamic_ODS):
    """
    Class that provides dynamic data loading from NC file
    This class is not to be used by itself, but via the ODS.open() method.
    """

    def __init__(self, filename):
        self.kw = {'filename': filename}
        self.dataset = None
        self.active = False

    def open(self):
        printd('Dynamic open  %s' % self.kw, topic='dynamic')
        from netCDF4 import Dataset

        self.dataset = Dataset(self.kw['filename'], 'r')
        self.active = True
        return self

    def close(self):
        printd('Dynamic close %s' % self.kw, topic='dynamic')
        self.dataset.close()
        self.dataset = None
        self.active = False
        return self

    def __getitem__(self, key):
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        printd('Dynamic read  %s: %s' % (self.kw['filename'], key), topic='dynamic')
        return get_ds_item(self.dataset, key)

    def __contains__(self, key):
        if not self.active:
            raise RuntimeError('Dynamic link broken: %s' % self.kw)
        return key in self.dataset.variables

    def keys(self, location):
        return numpy.unique(
            [convert_int(k[len(location) :].lstrip('.').split('.')[0]) for k in self.dataset.variables.keys() if k.startswith(location)]
        )


def through_omas_nc(ods, method=['function', 'class_method'][1]):
    """
    Test save and load NetCDF

    :param ods: ods

    :return: ods
    """
    filename = omas_testdir(__file__) + '/test.nc'
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
    if method == 'function':
        save_omas_json(ods, filename)
        ods1 = load_omas_json(filename)
    else:
        ods.save(filename)
        ods1 = ODS().load(filename)
    return ods1
