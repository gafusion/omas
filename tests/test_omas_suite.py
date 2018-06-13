#!/usr/bin/env python
# # -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
import unittest

import os
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
    if not os.path.exists(os.environ.get('AWS_CONFIG_FILE',os.environ['HOME']+'/.aws/config')):
        raise(RuntimeError('Missing AWS configuration file ~/.aws/config'))
    failed_S3 = False
except RuntimeError as _excp:
    failed_S3 = _excp


ods = ods_sample()

class TestOmasSuite(unittest.TestCase):

    def test_omas_pkl(self):
        through_omas_pkl(ods)

    def test_omas_json(self):
        through_omas_json(ods)

    def test_omas_nc(self):
        through_omas_nc(ods)

    @unittest.skipUnless(not failed_S3, str(failed_S3))
    def test_omas_s3(self):
        through_omas_s3(ods)

    @unittest.skipUnless(not failed_IMAS, str(failed_IMAS))
    def test_omas_imas(self):
        through_omas_imas(ods)

    @unittest.skipUnless(not failed_HDC, str(failed_HDC))
    def test_omas_hdc(self):
        through_omas_hdc(ods)

    @unittest.skipUnless(not failed_ITM, str(failed_ITM))
    def test_omas_itm(self):
        through_omas_itm(ods)

if __name__ == '__main__':
    unittest.main()
