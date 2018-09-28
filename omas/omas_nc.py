'''save/load from NC routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS


# --------------------------------------------
# save and load OMAS with NetCDF
# --------------------------------------------
def save_omas_nc(ods, filename, **kw):
    """
    Save an OMAS data set to on Amazon S3 server

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
            std  = None
            tmp=is_uncertain(odsf[item])
            if numpy.any(numpy.atleast_1d(tmp)):
                std=std_devs(data)
                data=nominal_values(data)
            for k in range(len(numpy.asarray(odsf[item]).shape)):
                dims.append('dim_%d' % (data.shape[k]))
                if dims[-1] not in dataset.dimensions:
                    dataset.createDimension(dims[-1], data.shape[k])
            if std is None:
                dataset.createVariable(item, data.dtype, dims)[:] = data
            else:
                dataset.createVariable(item, data.dtype, dims)[:] = data
                dataset.createVariable(item+'_error_upper', data.dtype, dims)[:] = std


def load_omas_nc(filename):
    """
    Load an OMAS data set from Amazon S3 server

    :param filename: filename to load from

    :return: OMAS data set
    """
    printd('Loading from %s' % filename, topic='nc')

    from netCDF4 import Dataset
    ods = ODS()
    with Dataset(filename, 'r') as dataset:
        for item in dataset.variables.keys():
            if item.endswith('_error_upper'):
                continue
            if dataset.variables[item].shape:
                # arrays
                if item+'_error_upper' in dataset.variables.keys():
                    ods[item] = uarray(numpy.array(dataset.variables[item]),
                                       numpy.array(dataset.variables[item+'_error_upper']))
                else:
                    ods[item] = numpy.array(dataset.variables[item])
            else:
                # uncertain scalars
                if item+'_error_upper' in dataset.variables.keys():
                    ods[item] = ufloat(numpy.asscalar(dataset.variables[item][0]),
                                       numpy.asscalar(dataset.variables[item+'_error_upper'][0]))
                else:
                    try:
                        # scalars
                        ods[item] = numpy.asscalar(dataset.variables[item][0])
                    except AttributeError:
                        # strings
                        ods[item] = dataset.variables[item][0]
    return ods


def through_omas_nc(ods):
    """
    test save and load NetCDF

    :param ods: ods

    :return: ods
    """
    if not os.path.exists(tempfile.gettempdir()+'/OMAS_TESTS/'):
        os.makedirs(tempfile.gettempdir()+'/OMAS_TESTS/')
    filename = tempfile.gettempdir()+'/OMAS_TESTS/test.nc'
    save_omas_nc(ods, filename)
    ods1 = load_omas_nc(filename)
    return ods1
