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
    specific_test_version = '3.18.0'

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
        self.printv('  diff_eq = {}'.format(diff_eq))
        assert isinstance(diff_eq, basestring)
        assert ('equilibrium' in diff_eq) or ('wall' in diff_eq)
        ods3 = copy.deepcopy(ods2)
        assert different_ods(ods2, ods3) is False
        ods3.sample_profiles()
        diff_prof = different_ods(ods2, ods3)
        self.printv('  diff_prof = {}'.format(diff_prof))
        assert isinstance(diff_prof, basestring)
        assert isinstance(different_ods(ods3, ods2), basestring)
        assert 'core_profiles' in diff_prof
        ods2.sample_profiles(include_pressure=False)
        diff_prof2 = different_ods(ods3, ods2)
        self.printv('  diff_prof2 = {}'.format(diff_prof2))
        assert isinstance(diff_prof2, basestring)
        assert 'core_profiles' in diff_prof2
        ods2.sample_profiles()
        ods2['core_profiles.profiles_1d.0.electrons.density'][0] = 1.5212
        diff_prof3 = different_ods(ods2, ods3)
        self.printv('  diff_prof3 = {}'.format(diff_prof3))
        assert isinstance(diff_prof3, basestring)
        assert 'value' in diff_prof3
        ods2.sample_profiles()
        ods2['core_profiles.profiles_1d.0.ion.0.element.0.a'] = 9.
        diff_prof4 = different_ods(ods2, ods3)
        self.printv('  diff_prof4 = {}'.format(diff_prof4))
        assert isinstance(diff_prof4, basestring)
        assert 'value' in diff_prof4
        ods2.sample_profiles()
        ods2['core_profiles.code.name'] = 'fake name 1'
        ods3['core_profiles.code.name'] = 'fake name 2'
        diff_prof5 = different_ods(ods2, ods3)
        self.printv('  diff_prof5 = {}'.format(diff_prof5))
        assert isinstance(diff_prof5, basestring)
        assert 'name' in diff_prof5
        ods3['core_profiles.code.name'] = numpy.array([2, 3, 4])
        assert isinstance(different_ods(ods2, ods3), basestring)
        ods2['core_profiles.code.name'] = uarray(numpy.array([2, 3, 4]), numpy.array([1, 1, 1]))
        assert isinstance(different_ods(ods2, ods3), basestring)

    def test_printe(self):
        printe('printe_test,', end='')
        printw('printw_test', end='')

    def test_is_numeric(self):
        assert is_numeric(5) is True
        assert is_numeric(numpy.array([5])) is True
        assert is_numeric('blah') is False
        assert is_numeric({'blah': 'blah'}) is False
        assert is_numeric([]) is False
        assert is_numeric(None) is False

    def test_remove_parentheses(self):
        assert remove_parentheses('zoom(blah)what', replace_with='|') == 'zoom|what'

    def test_closest_index(self):
        # Basic tests
        assert closest_index([1, 2, 3, 4], 3) == 2  # Basic test
        assert closest_index([1, 2, 3], 1) == 0  # Special: The first element is the one sought
        assert closest_index(numpy.array([1, 2, 3, 4]), 4) == 3  # Special: The last element is the one sought
        assert closest_index([1, 2, 2, 3], 2) == 1  # Special: duplicated value: pick first instance
        assert closest_index([1, 2, 3, 4], 2.2) == 1  # Make sure it works for numbers in between
        assert closest_index([1, 2, 3, 4], 2.7) == 2
        # Exception handling and coping with problems
        self.assertRaises(TypeError, closest_index, 5, 5)  # First arg is not a list --> TypeError
        self.assertRaises(TypeError, closest_index, [1, 2, 3], 'string_not_number')
        assert closest_index([1, 2, 3], [3]) == 2  # Should use first element of second arg if it's not a scalar
        self.assertRaises(TypeError, closest_index, [1, 2, 3], [3, 2, 1])  # Can't call w/ list as 2nd arg unless len=1

    def test_list_structures(self):  # Also tests dict_structures
        struct_list = list_structures(default_imas_version)
        struct_list2 = list_structures(self.specific_test_version)
        assert isinstance(struct_list, list)
        assert isinstance(struct_list2, list)
        assert isinstance(struct_list[0], basestring)
        assert 'pf_active' in struct_list2
        struct_dict = dict_structures(default_imas_version)
        struct_dict2 = dict_structures(self.specific_test_version)
        assert isinstance(struct_dict, dict)
        assert isinstance(struct_dict2, dict)
        assert 'pf_active' in struct_dict2.keys()
        assert all([item in struct_dict.keys() for item in struct_list])
        assert all([item in struct_dict2.keys() for item in struct_list2])

    def test_omas_info(self):
        get_list = ['pf_active', 'thomson_scattering', 'charge_exchange']
        ods_info_pfa = omas_info(get_list[0])
        ods_info_pfa2 = omas_info(get_list[0], self.specific_test_version)
        assert get_list[0] in ods_info_pfa.keys()
        assert get_list[0] in ods_info_pfa2.keys()
        if get_list[0] == 'pf_active':
            assert isinstance(ods_info_pfa['pf_active.circuit.0.connections.documentation'], basestring)
        ods_info_list = omas_info(get_list)
        assert all([item in ods_info_list for item in get_list])

    def test_o2u(self):
        assert o2u('equilibrium.time_slice.0.global_quantities.ip')=='equilibrium.time_slice.:.global_quantities.ip'
        assert o2u('equilibrium.time_slice.:.global_quantities.ip')=='equilibrium.time_slice.:.global_quantities.ip'
        assert o2u('2')==':'
        assert o2u('equilibrium')=='equilibrium'
        assert o2u('equilibrium.2')=='equilibrium.:'

    def test_set_time_array(self):
        ods=ODS()
        ods.set_time_array('equilibrium.vacuum_toroidal_field.b0',0,0.1)
        ods.set_time_array('equilibrium.vacuum_toroidal_field.b0',1,0.2)

if __name__ == '__main__':
    unittest.main()
