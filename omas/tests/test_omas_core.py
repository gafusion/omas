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
import numpy
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

    def test_time(self):
        # test generation of a sample ods
        ods = ODS()
        ods['equilibrium.time_slice'][0]['time'] = 100
        ods['equilibrium.time_slice.0.global_quantities.ip'] = 0.0
        ods['equilibrium.time_slice'][1]['time'] = 200
        ods['equilibrium.time_slice.1.global_quantities.ip'] = 1.0
        ods['equilibrium.time_slice'][2]['time'] = 300
        ods['equilibrium.time_slice.2.global_quantities.ip'] = 2.0

        # get time information from children
        extra_info = {}
        assert numpy.allclose(ods.time('equilibrium', extra_info=extra_info),[100, 200, 300])
        assert extra_info['location']=='equilibrium.time_slice.:.time'
        assert extra_info['homogeneous_time'] is True

        # time arrays can be set using `set_time_array` function
        # this simplifies the logic in the code since one does not
        # have to check if the array was already there or not
        ods.set_time_array('equilibrium.time',0,101)
        ods.set_time_array('equilibrium.time',1,201)
        ods.set_time_array('equilibrium.time',2,302)

        # the make the timeslices consistent
        ods['equilibrium.time_slice'][0]['time'] = 101
        ods['equilibrium.time_slice'][1]['time'] = 201
        ods['equilibrium.time_slice'][2]['time'] = 302

        # get time information from explicitly set time array
        extra_info = {}
        assert numpy.allclose(ods.time('equilibrium', extra_info=extra_info),[101, 201, 302])
        assert extra_info['homogeneous_time'] is False

        # get time value from item in array of structures
        extra_info = {}
        assert ods['equilibrium.time_slice'][0].time(extra_info=extra_info)==101
        assert extra_info['homogeneous_time'] is None

        # get time array from array of structures
        extra_info = {}
        assert numpy.allclose(ods['equilibrium.time_slice'].time(extra_info=extra_info),[101, 201, 302])
        assert extra_info['homogeneous_time'] is False

        # get time from parent
        extra_info = {}
        assert ods.time('equilibrium.time_slice.0.global_quantities.ip', extra_info=extra_info)==101
        assert extra_info['homogeneous_time'] is None

        # slice at time
        ods1 = ods['equilibrium'].slice_at_time(101)
        numpy.allclose(ods.time('equilibrium'),[101])

if __name__ == '__main__':
    unittest.main()
