#!/usr/bin/env python
# # -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
import unittest

import os
import numpy
from omas import *

try:
    import imas

    failed_IMAS = False
except ImportError as _excp:
    failed_IMAS = _excp

try:
    import hdc

    failed_HDC = False
except ImportError as _excp:
    failed_HDC = _excp

try:
    import ual

    failed_ITM = False
except ImportError as _excp:
    failed_ITM = _excp

try:
    if not os.path.exists(os.environ.get('AWS_CONFIG_FILE', os.environ['HOME'] + '/.aws/config')):
        raise (RuntimeError('Missing AWS configuration file ~/.aws/config'))
    failed_S3 = False
except RuntimeError as _excp:
    failed_S3 = _excp

class TestOmasSuite(unittest.TestCase):

    def test_omas_pkl(self):
        ods = ods_sample()
        through_omas_pkl(ods)

    def test_omas_json(self):
        ods = ods_sample()
        through_omas_json(ods)

    def test_omas_nc(self):
        ods = ods_sample()
        through_omas_nc(ods)

    @unittest.skipUnless(not failed_S3, str(failed_S3))
    def test_omas_s3(self):
        ods = ods_sample()
        through_omas_s3(ods)

    @unittest.skipUnless(not failed_IMAS, str(failed_IMAS))
    def test_omas_imas(self):
        ods = ods_sample()
        through_omas_imas(ods)

    @unittest.skipUnless(not failed_HDC, str(failed_HDC))
    def test_omas_hdc(self):
        ods = ods_sample()
        through_omas_hdc(ods)

    @unittest.skipUnless(not failed_ITM, str(failed_ITM))
    def test_omas_itm(self):
        ods = ods_sample()
        through_omas_itm(ods)

    def test_omas_coordinates_intepolation(self):
        # if a coordinate exists, then that is the coordinate that it is used
        ods1 = ODS()
        ods1['equilibrium.time_slice[0].profiles_1d.psi'] = numpy.linspace(0, 1, 10)
        with coords_environment(ods1, {'equilibrium.time_slice[0].profiles_1d.psi': numpy.linspace(0, 1, 5)}):
            ods1['equilibrium.time_slice[0].profiles_1d.f'] = numpy.linspace(0, 1, 5)
        assert (len(ods1['equilibrium.time_slice[0].profiles_1d.f']) == 10)

        # if a does not exists, then that coordinate is set
        ods2 = ODS()
        with coords_environment(ods2, {'equilibrium.time_slice[0].profiles_1d.psi': numpy.linspace(0, 1, 5)}):
            ods2['equilibrium.time_slice[0].profiles_1d.pressure'] = numpy.linspace(0, 1, 5)
        assert (len(ods2['equilibrium.time_slice[0].profiles_1d.pressure']) == 5)

        # coordinates can be taken from existing ODSs
        ods3 = ODS()
        with coords_environment(ods3, ods1):
            ods3['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
        with coords_environment(ods3, ods2):
            ods3['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2[
                'equilibrium.time_slice[0].profiles_1d.pressure']
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10)
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.pressure']) == 10)

        # order matters
        ods4 = ODS()
        with coords_environment(ods4, ods2):
            ods4['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2[
                'equilibrium.time_slice[0].profiles_1d.pressure']
        with coords_environment(ods4, ods1):
            ods4['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
        assert (len(ods4['equilibrium.time_slice[0].profiles_1d.f']) == 5)
        assert (len(ods4['equilibrium.time_slice[0].profiles_1d.pressure']) == 5)

        # ods can be queried on different coordinates
        with coords_environment(ods4, ods1):
            assert(len(ods4['equilibrium.time_slice[0].profiles_1d.f'])==10)
        assert(len(ods4['equilibrium.time_slice[0].profiles_1d.f'])==5)

if __name__ == '__main__':
    unittest.main()
