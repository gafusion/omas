#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas saving/loading data in different formats

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_suite

-------
"""

import unittest
import os
import numpy

# OMAS imports
from omas import *
from omas.tests.failed_imports import *
from omas.tests import warning_setup


class TestOmasSuite(unittest.TestCase):

    def test_omas_pkl(self):
        ods = ods_sample()
        ods1 = through_omas_pkl(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('pkl through difference: %s' % diff)

    def test_omas_json(self):
        ods = ods_sample()
        ods1 = through_omas_json(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('json through difference: %s' % diff)

    def test_omas_nc(self):
        ods = ods_sample()
        ods1 = through_omas_nc(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('nc through difference: %s' % diff)

    def test_omas_h5(self):
        ods = ods_sample()
        ods1 = through_omas_h5(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('h5 through difference: %s' % diff)

    def test_omas_ds(self):
        ods = ods_sample()
        ods1 = through_omas_ds(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('ds through difference: %s' % diff)

    def test_omas_dx(self):
        ods = ods_sample()
        odx = ods_2_odx(ods)
        odx1 = through_omas_dx(odx)
        ods1 = odx_2_ods(odx1)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('dx through difference: %s' % diff)

    @unittest.skipIf(failed_MONGO, str(failed_MONGO))
    def test_omas_mongo(self):
        ods = ods_sample()
        ods1 = through_omas_mongo(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('mongo through difference: %s' % diff)

    @unittest.skipIf(failed_S3, str(failed_S3))
    def test_omas_s3(self):
        ods = ods_sample()
        ods1 = through_omas_s3(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('s3 through difference: %s' % diff)

    @unittest.skipIf(failed_IMAS, str(failed_IMAS))
    def test_omas_imas(self):
        ods = ods_sample()
        ods1 = through_omas_imas(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('hdc through difference: %s' % diff)

    @unittest.skipIf(failed_HDC, str(failed_HDC))
    def test_omas_hdc(self):
        ods = ods_sample()
        ods1 = through_omas_hdc(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('hdc through difference: %s' % diff)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasSuite)
    unittest.TextTestRunner(verbosity=2).run(suite)
