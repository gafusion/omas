#!/usr/bin/env python
# # -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
import unittest

from omas import *


class TestOmasSuite(unittest.TestCase):

    def test_omas(self):
        test_omas_suite(ods=None, test_type=None, do_raise=False)
        # do_raise = False will prevent exceptions! The test does nothing but look pretty.


if __name__ == '__main__':
    unittest.main()
