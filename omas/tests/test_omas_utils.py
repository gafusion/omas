#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_utils.py
python -m unittest test_omas_utils (from omas/omas/tests)
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


class TestOmasUtils(unittest.TestCase):
    """
    Test suite for omas_utils.py
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

    def test_different_ods(self):
        assert different_ods(self.ods, self.ods) is False
        ods2 = ODS()
        ods2.sample_equilibrium()
        diff_eq = different_ods(self.ods, ods2)
        assert isinstance(diff_eq, basestring)
        assert 'equilibrium' in diff_eq
        ods3 = copy.deepcopy(ods2)
        assert different_ods(ods2, ods3) is False
        ods3.sample_profiles()
        diff_prof = different_ods(ods2, ods3)
        assert isinstance(diff_prof, basestring)
        assert isinstance(different_ods(ods3, ods2), basestring)
        assert 'core_profiles' in diff_prof
        ods2.sample_profiles(include_pressure=False)
        diff_prof2 = different_ods(ods3, ods2)
        assert isinstance(diff_prof2, basestring)
        assert 'core_profiles' in diff_prof2
        ods2.sample_profiles()
        ods2['core_profiles.profiles_1d.0.electrons.density'][0] = 1.5212
        diff_prof3 = different_ods(ods2, ods3)
        assert isinstance(diff_prof3, basestring)
        assert 'value' in diff_prof3
        ods2.sample_profiles()
        ods2['core_profiles.profiles_1d.0.ion.0.element.0.a'] = 9.
        diff_prof4 = different_ods(ods2, ods3)
        assert isinstance(diff_prof4, basestring)
        assert 'value' in diff_prof4

    def test_printe(self):
        printe('blah blah printe test')
        printw('blah blah printw test')

    def test_is_numeric(self):
        assert is_numeric(5) is True
        assert is_numeric(numpy.array([5])) is True
        assert is_numeric('blah') is False
        assert is_numeric({'blah': 'blah'}) is False
        assert is_numeric([]) is False
        assert is_numeric(None) is False

    def test_remove_parentheses(self):
        assert remove_parentheses('zoom(blah)what', replace_with='|') == 'zoom|what'


if __name__ == '__main__':
    unittest.main()
