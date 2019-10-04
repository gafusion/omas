#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas saving/loading data in different formats

.. code-block:: none

   python -m unittest omas/tests/test_omas_suite

-------
"""

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
    import boto3

    if not os.path.exists(os.environ.get('AWS_CONFIG_FILE', os.environ['HOME'] + '/.aws/config')):
        raise RuntimeError('Missing AWS configuration file ~/.aws/config')
    failed_S3 = False
except RuntimeError as _excp:
    failed_S3 = _excp

try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError
    from omas.omas_mongo import get_mongo_credentials

    up = get_mongo_credentials(server=omas_rcparams['default_mongo_server'])
    client = MongoClient(omas_rcparams['default_mongo_server'].format(**up), serverSelectionTimeoutMS=1000)
    client.server_info()
    failed_mongo = False
except ServerSelectionTimeoutError as _excp:
    failed_mongo = _excp


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

    @unittest.skipUnless(not failed_mongo, str(failed_mongo))
    def test_omas_mongo(self):
        ods = ods_sample()
        ods1 = through_omas_mongo(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('mongo through difference: %s' % diff)

    @unittest.skipUnless(not failed_S3, str(failed_S3))
    def test_omas_s3(self):
        ods = ods_sample()
        ods1 = through_omas_s3(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('s3 through difference: %s' % diff)

    @unittest.skipUnless(not failed_IMAS, str(failed_IMAS))
    def test_omas_imas(self):
        ods = ods_sample()
        ods1 = through_omas_imas(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('hdc through difference: %s' % diff)

    @unittest.skipUnless(not failed_HDC, str(failed_HDC))
    def test_omas_hdc(self):
        ods = ods_sample()
        ods1 = through_omas_hdc(ods)
        diff = ods.diff(ods1)
        if diff:
            raise AssertionError('hdc through difference: %s' % diff)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasSuite)
    unittest.TextTestRunner(verbosity=2).run(suite)
