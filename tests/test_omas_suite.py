#!/usr/bin/env python
# # -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
import unittest

from omas import *

try:
    import imas
    HAVE_IMAS = True
except ImportError:
    HAVE_IMAS = False

try:
    import hdc
    HAVE_HDC = True
except ImportError:
    HAVE_HDC = False

try:
    import ual
    HAVE_ITM = True
except ImportError:
    HAVE_ITM = False

ods = ods_sample()

class TestOmasSuite(unittest.TestCase):

    def test_omas_pkl(self):
        through_omas_pkl(ods)

    def test_omas_json(self):
        through_omas_json(ods)

    def test_omas_nc(self):
        through_omas_nc(ods)

    def test_omas_s3(self):
        through_omas_s3(ods)

    @unittest.skipUnless(HAVE_IMAS, "requires imas")
    def test_omas_imas(self):
        through_omas_imas(ods)

    @unittest.skipUnless(HAVE_HDC, "requires hdc")
    def test_omas_hdc(self):
        through_omas_hdc(ods)

    @unittest.skipUnless(HAVE_ITM, "requires ual")
    def test_omas_itm(self):
        through_omas_itm(ods)

if __name__ == '__main__':
    unittest.main()
