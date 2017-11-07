from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import omas, save_omas_pkl, load_omas_pkl

#--------------------------------------------
# save and load OMAS with NetCDF
#--------------------------------------------
def save_omas_nc(ods, filename, **kw):
    '''
    Save an OMAS data set to on Amazon S3 server

    :param ods: OMAS data set

    :param filename: filename to save to

    :param kw: arguments passed to the netCDF4 Dataset function
    '''
    printd('Saving to %s'%(filename),topic='nc')

    from netCDF4 import Dataset
    odsf=ods.flat()
    with Dataset(filename,'w',**kw) as dataset:
        for item in odsf:
            dims=[]
            for k in range(len(numpy.asarray(odsf[item]).shape)):
                dims.append('%s__dim_%d'%(item,k))
                dataset.createDimension(dims[-1],numpy.asarray(odsf[item]).shape[k])
            tmp=dataset.createVariable(item,numpy.asarray(odsf[item]).dtype,dims)
            tmp[:]=numpy.asarray(odsf[item])

def load_omas_nc(filename):
    '''
    Load an OMAS data set from Amazon S3 server

    :param filename: filename to load from

    :return: OMAS data set
    '''
    printd('Loading from %s'%(filename),topic='nc')

    from netCDF4 import Dataset
    ods=omas()
    with Dataset(filename,'r') as dataset:
        for item in dataset.variables:
            if dataset.variables[item].shape:
                #arrays
                ods[item]=numpy.array(dataset.variables[item])
            else:
                try:
                    #scalars
                    ods[item]=numpy.asscalar(dataset.variables[item][0])
                except AttributeError:
                    #strings
                    ods[item]=dataset.variables[item][0]
    return ods

def test_omas_nc(ods):
    '''
    test save and load NetCDF

    :param ods: ods

    :return: ods
    '''
    filename='test.nc'
    save_omas_nc(ods,filename)
    ods1=load_omas_nc(filename)
    return ods1
