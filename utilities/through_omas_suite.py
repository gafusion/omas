#!/usr/bin/env python
# # -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals

import os
import numpy
from omas import *
from omas.omas_utils import printd

_tests = ['pkl', 'json', 'nc', 's3', 'imas', 'hdc']


def through_omas_suite(ods=None, test_type=None, do_raise=False):
    """
    :param ods: omas structure to test. If None this is set to ods_sample

    :param test_type: None tests all suite, otherwise choose among %s

    :param do_raise: raise error if something goes wrong
    """

    if ods is None:
        ods = ods_sample()

    if test_type in _tests:
        os.environ['OMAS_DEBUG_TOPIC'] = test_type
        ods1 = globals()['through_omas_' + test_type](ods)
        check = different_ods(ods, ods1)
        if not check:
            print('OMAS data got saved and loaded correctly')
        else:
            print(check)

    else:
        os.environ['OMAS_DEBUG_TOPIC'] = '*'
        printd('OMAS is using IMAS data structure version `%s` as default' % default_imas_version, topic='*')

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

                    different = different_ods(ods1, ods2)
                    if not different:
                        print('FROM %s TO %s : OK' % (t1.center(5), t2.center(5)))
                        results[k1, k2] = 1.0
                    else:
                        print('FROM %s TO %s : NO --> %s' %
                              (t1.center(5), t2.center(5), different))
                        results[k1, k2] = -1.0

                except Exception as _excp:
                    print('FROM %s TO %s : NO --> %s' %
                          (t1.center(5), t2.center(5), repr(_excp)))
                    if do_raise:
                        raise
        print('=' * 20)
        print(results.astype(int))
        print('=' * 20)


through_omas_suite.__doc__ = through_omas_suite.__doc__ % _tests

through_omas_suite(ods=None, test_type=None, do_raise=False)
