'''save/load from HDC routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_core import ODS
from collections import Sequence
import numpy

try:
    _pyhdc_import_excp = None
    from pyhdc import HDC
except ImportError as _excp:
    _pyhdc_import_excp = _excp


    # replace HDC class by a simple exception throwing class
    class HDC(object):
        """Import error HDC class"""

        def __init__(self, *args, **kwargs):
            raise _pyhdc_import_excp


def save_omas_hdc(ods):
    """Convert OMAS data structure to HDC

    :param ods: input data structure

    :return: HDC container
    """
    # recurrent function - check types
    if isinstance(ods, ODS):
        if isinstance(ods.keys(), Sequence):
            # list type
            # TODO implement a better check
            hdc = HDC()
            for value in ods:
                hdc.append(save_omas_hdc(value))
        else:
            # mapping type
            hdc = HDC()
            for key, value in ods.items():
                hdc[key] = save_omas_hdc(value)
    else:
        # primitive type
        hdc = HDC(ods)

    return hdc


def load_omas_hdc(hdc, consistency_check=True):
    """Convert HDC data structure to OMAS

    :param hdc: input data structure

    :param consistency_check: verify that data is consistent with IMAS schema

    :return: populated ODS
    """
    # recurrent function - check types
    if not isinstance(hdc, HDC):
        raise ValueError('expecting HDC type')
    else:
        if hdc.get_type_str() == 'list':
            # list type
            ods = ODS(consistency_check=False)
            for i in range(hdc.shape[0]):
                ods[i] = load_omas_hdc(hdc[i], consistency_check=None)
        elif hdc.get_type_str() == 'struct':
            # mapping type
            ods = ODS(consistency_check=False)
            for key in hdc.keys():
                ods[key] = load_omas_hdc(hdc[key], consistency_check=None)
        elif hdc.get_type_str() == 'null':
            # null type
            ods = ODS(consistency_check=False)
        elif hdc.get_type_str() == 'string':
            # string type
            ods = str(hdc)
        else:
            # numeric type
            ods = numpy.asarray(hdc)
            if numpy.isscalar(ods) or ods.size == 1:
                ods = ods.item()
    if consistency_check is not None:
        ods.consistency_check = consistency_check
    return ods


def through_omas_hdc(ods):
    '''
    Test save and load HDC

    :param ods: ods

    :return: ods
    '''
    hdc = save_omas_hdc(ods)
    ods1 = load_omas_hdc(hdc)
    return ods1
