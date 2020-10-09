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
        ods = ODS().sample()
        ods1 = through_omas_pkl(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('pkl through difference')

    def test_omas_json(self):
        ods = ODS().sample()
        ods1 = through_omas_json(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('json through difference')

    def test_omas_nc(self):
        ods = ODS().sample()
        ods1 = through_omas_nc(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('nc through difference')

    def test_omas_h5(self):
        ods = ODS().sample()
        ods1 = through_omas_h5(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('h5 through difference')

    def test_omas_ds(self):
        ods = ODS().sample()
        ods1 = through_omas_ds(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('ds through difference')

    def test_omas_ascii(self):
        ods = ODS().sample()
        for one_or_many_files in ['one', 'many']:
            ods1 = through_omas_ascii(ods, one_or_many_files=one_or_many_files)
            diff = ods.diff(ods1)
            if diff:
                print('\n'.join(diff))
                raise AssertionError(f'ascii through difference for {one_or_many_files} file(s)')

    def test_omas_dx(self):
        ods = ODS().sample()
        odx = ods_2_odx(ods)
        odx1 = through_omas_dx(odx)
        ods1 = odx_2_ods(odx1)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('dx through difference')

    @unittest.skipIf(failed_MONGO, str(failed_MONGO))
    def test_omas_mongo(self):
        ods = ODS().sample()
        ods1 = through_omas_mongo(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('mongo through difference')

    @unittest.skipIf(failed_S3, str(failed_S3))
    def test_omas_s3(self):
        ods = ODS().sample()
        ods1 = through_omas_s3(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('s3 through difference')

    @unittest.skipIf(failed_IMAS, str(failed_IMAS))
    def test_omas_imas(self):
        ods = ODS().sample()
        ods1 = through_omas_imas(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('hdc through difference')

    @unittest.skipIf(failed_HDC, str(failed_HDC))
    def test_omas_hdc(self):
        ods = ODS().sample()
        ods1 = through_omas_hdc(ods)
        diff = ods.diff(ods1)
        if diff:
            print('\n'.join(diff))
            raise AssertionError('hdc through difference')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasSuite)
    unittest.TextTestRunner(verbosity=2).run(suite)
