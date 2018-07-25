#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_core.py
python -m unittest test_omas_core (from omas/omas/tests)
python -m unittest discover omas (from omas top level; runs all tests)
"""

# Basic imports
from __future__ import print_function, division, unicode_literals
import unittest
from pprint import pprint

# OMAS imports
from omas import *

class TestOmasCore(unittest.TestCase):
    """
    Test suite for omas_physics.py
    """

    # Flags to edit while testing
    verbose = False  # Spammy, but occasionally useful for debugging a weird problem

    # Utilities for this test
    def printv(self, *arg):
        """Utility for tests to use"""
        if self.verbose:
            print(*arg)

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))

    def test_coordinates(self):
        ods = ods_sample()
        assert (len(ods.coordinates()) > 0)
        assert (len(ods['equilibrium'].coordinates()) > 0)

if __name__ == '__main__':
    unittest.main()
