#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
Test all save/load systems
==========================
Save and load a sample ODS through the different OMAS save formats and check that information does not get lost or corrupted.
A final report summarizes if some storage systems combinations have errors.
"""

import os
import copy
import numpy
from pprint import pprint
from omas import *
from omas.omas_utils import printd

_tests = ['pkl', 'json', 'nc', 's3', 'h5', 'imas', 'hdc']


def through_omas_suite(ods=None, test_type=None, do_raise=False):
    """
    :param ods: omas structure to test. If None this is set to ods_sample

    :param test_type: None tests all suite, otherwise choose among %s

    :param do_raise: raise error if something goes wrong
    """

    if ods is None:
        ods = ODS().sample()
    ods = copy.deepcopy(ods)  # make a copy to make sure throuhs do not alter original ODS

    if test_type in _tests:
        os.environ['OMAS_DEBUG_TOPIC'] = test_type
        ods1 = globals()['through_omas_' + test_type](ods)
        difference = ods.diff(ods1)
        if not chedifferenceck:
            print('OMAS data got saved and loaded correctly')
        else:
            pprint(difference)

    else:
        os.environ['OMAS_DEBUG_TOPIC'] = '*'
        printd('OMAS is using IMAS data structure version `%s` as default' % omas_rcparams['default_imas_version'], topic='*')

        print('=' * 20)

        results = numpy.zeros((len(_tests), len(_tests)))

        for k1, t1 in enumerate(_tests):
            failed1 = False
            try:
                ods1 = globals()['through_omas_' + t1](ods)
            except Exception as _excp:
                failed1 = _excp
                if do_raise:
                    raise
            for k2, t2 in enumerate(_tests):
                try:
                    if failed1:
                        raise failed1
                    ods2 = globals()['through_omas_' + t2](ods1)

                    different = ods1.diff(ods2)
                    if not different:
                        print('FROM %s TO %s : OK' % (t1.center(5), t2.center(5)))
                        results[k1, k2] = 1.0
                    else:
                        print('FROM %s TO %s : NO --> %s' % (t1.center(5), t2.center(5), different))
                        results[k1, k2] = -1.0

                except Exception as _excp:
                    print('FROM %s TO %s : NO --> %s' % (t1.center(5), t2.center(5), repr(_excp)))
                    if do_raise:
                        raise
        print('=' * 20)
        print(results.astype(int))
        print('=' * 20)


through_omas_suite(ods=None, test_type=None, do_raise=False)

###################################
# In the above example `consistency_check = True` result in the following error::
#
#     LookupError: Not a valid IMAS 3.18.0 location: `equilibrium.time_slice.:.does_not_exist`
#                                                                              ^^^^^^^^^^^^^^
#     Did you mean: ['profiles_2d', 'ggd', 'boundary', 'profiles_1d', 'constraints', 'global_quantities', 'coordinate_system', 'boundary_separatrix', 'time', 'convergence']

#################################################
# If successful, the test should output:
#
# .. code-block:: none
#
#     ====================
#     FROM  pkl  TO  pkl  : OK
#     FROM  pkl  TO  json : OK
#     FROM  pkl  TO   nc  : OK
#     FROM  pkl  TO   s3  : OK
#     FROM  pkl  TO  imas : OK
#     FROM  json TO  pkl  : OK
#     FROM  json TO  json : OK
#     FROM  json TO   nc  : OK
#     FROM  json TO   s3  : OK
#     FROM  json TO  imas : OK
#     FROM   nc  TO  pkl  : OK
#     FROM   nc  TO  json : OK
#     FROM   nc  TO   nc  : OK
#     FROM   nc  TO   s3  : OK
#     FROM   nc  TO  imas : OK
#     FROM   s3  TO  pkl  : OK
#     FROM   s3  TO  json : OK
#     FROM   s3  TO   nc  : OK
#     FROM   s3  TO   s3  : OK
#     FROM   s3  TO  imas : OK
#     FROM  imas TO  pkl  : OK
#     FROM  imas TO  json : OK
#     FROM  imas TO   nc  : OK
#     FROM  imas TO   s3  : OK
#     FROM  imas TO  imas : OK
#     ====================
#     [[1 1 1 1 1]
#      [1 1 1 1 1]
#      [1 1 1 1 1]
#      [1 1 1 1 1]
#      [1 1 1 1 1]]
#     ====================
#
# If for example `imas` is not installed on the system, the test output will look like:
#
# .. code-block:: none
#
#     ====================
#     FROM  pkl  TO  pkl  : OK
#     FROM  pkl  TO  json : OK
#     FROM  pkl  TO   nc  : OK
#     FROM  pkl  TO   s3  : OK
#     FROM  pkl  TO  imas : NO --> ImportError('No module named imas',)
#     FROM  json TO  pkl  : OK
#     FROM  json TO  json : OK
#     FROM  json TO   nc  : OK
#     FROM  json TO   s3  : OK
#     FROM  json TO  imas : NO --> ImportError('No module named imas',)
#     FROM   nc  TO  pkl  : OK
#     FROM   nc  TO  json : OK
#     FROM   nc  TO   nc  : OK
#     FROM   nc  TO   s3  : OK
#     FROM   nc  TO  imas : NO --> ImportError('No module named imas',)
#     FROM   s3  TO  pkl  : OK
#     FROM   s3  TO  json : OK
#     FROM   s3  TO   nc  : OK
#     FROM   s3  TO   s3  : OK
#     FROM   s3  TO  imas : NO --> ImportError('No module named imas',)
#     FROM  imas TO  pkl  : NO --> ImportError('No module named imas',)
#     FROM  imas TO  json : NO --> ImportError('No module named imas',)
#     FROM  imas TO   nc  : NO --> ImportError('No module named imas',)
#     FROM  imas TO   s3  : NO --> ImportError('No module named imas',)
#     FROM  imas TO  imas : NO --> ImportError('No module named imas',)
#     ====================
#     [[1 1 1 1 0]
#      [1 1 1 1 0]
#      [1 1 1 1 0]
#      [1 1 1 1 0]
#      [0 0 0 0 0]]
#     ====================
