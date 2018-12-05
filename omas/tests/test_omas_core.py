#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_core.py

.. code-block:: none

   python -m unittest omas/tests/test_omas_core

-------
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
        assert (len(ods.list_coordinates()) > 0)
        assert (len(ods['equilibrium'].list_coordinates()) > 0)

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

    def test_address_structures(self):
        ods = ODS()

        # make sure data structure is of the right type
        assert isinstance(ods['core_transport'].omas_data,dict)
        assert isinstance(ods['core_transport.model'].omas_data,list)

        # append elements by using `+`
        for k in range(10):
            ods['equilibrium.time_slice.+.global_quantities.ip'] = k
        assert len(ods['equilibrium.time_slice']) == 10
        assert (ods['equilibrium.time_slice'][9]['global_quantities.ip'] == 9)

        # access element by using negative indices
        assert (ods['equilibrium.time_slice'][-1]['global_quantities.ip'] == 9)
        assert (ods['equilibrium.time_slice.-10.global_quantities.ip'] == 0)

        # set element by using negative indices
        ods['equilibrium.time_slice.-1.global_quantities.ip'] = -99
        ods['equilibrium.time_slice'][-10]['global_quantities.ip'] = -100
        assert (ods['equilibrium.time_slice'][-1]['global_quantities.ip'] == -99)
        assert (ods['equilibrium.time_slice'][-10]['global_quantities.ip'] == -100)

        # access by pattern
        assert (ods['@eq.*1.*.ip'] == 1)

    def test_version(self):
        ods = ODS(imas_version='3.20.0')
        ods['ec_antennas.antenna.0.power'] = 1.0

        try:
            ods = ODS(imas_version='3.21.0')
            ods['ec_antennas.antenna.0.power'] = 1.0
            raise AssertionError('3.21.0 should not have `ec_antennas.antenna.0.power`')
        except LookupError:
            pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasCore)
    unittest.TextTestRunner(verbosity=2).run(suite)
