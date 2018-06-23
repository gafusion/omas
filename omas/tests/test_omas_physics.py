#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_physics.py
python -m unittest test_omas_physics (from omas/omas/tests)
python -m unittest discover omas (from omas top level; runs all tests)
"""

# Basic imports
from __future__ import print_function, division, unicode_literals
import os
import unittest
import numpy
import warnings
import copy

# Plot imports
import matplotlib as mpl
from matplotlib import pyplot as plt

# OMAS imports
from omas import *
from omas.omas_utils import *
from omas.omas_physics import *


class TestOmasPhysics(unittest.TestCase):
    """
    Test suite for omas_physics.py
    """

    # Flags to edit while testing
    verbose = False  # Spammy, but occasionally useful for debugging a weird problem

    # Sample data for use in tests
    ods = ODS()

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

    @unittest.expectedFailure  # TODO: fix core_profiles_pressures or revise this test
    def test_core_profiles_pressures(self):
        ods2 = copy.deepcopy(self.ods)
        ods2.sample_profiles(include_pressure=False)
        ods3 = copy.deepcopy(self.ods)
        updated_ods = core_profiles_pressures(ods2, update=False)
        updated_ods3 = core_profiles_pressures(ods3, update=True)

        assert updated_ods3 == ods3
        assert updated_ods != ods2
        assert updated_ods3 == updated_ods

    def test_define_cocos(self):
        cocos_none = define_cocos(None)
        cocos1 = define_cocos(1)
        cocos2 = define_cocos(2)
        cocos3 = define_cocos(3)
        cocos4 = define_cocos(4)
        cocos5 = define_cocos(5)
        cocos6 = define_cocos(6)
        cocos7 = define_cocos(7)
        cocos8 = define_cocos(8)
        cocos11 = define_cocos(11)
        for cocos in [cocos_none, cocos1, cocos2, cocos5, cocos6, cocos11]:
            assert cocos['sigma_Bp'] == 1
        for cocos in [cocos3, cocos4, cocos7, cocos8]:
            assert cocos['sigma_Bp'] == -1

    def test_cocos_transform(self):
        assert cocos_transform(None, None)['TOR'] == 1
        assert cocos_transform(1, 3)['POL'] == -1
        for cocos_ind in range(1, 9):
            assert cocos_transform(cocos_ind, cocos_ind + 10)['invPSI'] != 1
            for cocos_add in range(2):
                for thing in ['BT', 'TOR', 'POL', 'Q']:
                    assert cocos_transform(cocos_ind+cocos_add*10, cocos_ind+cocos_add*10)[thing] == 1


if __name__ == '__main__':
    unittest.main()
