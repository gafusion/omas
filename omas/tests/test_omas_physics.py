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

try:
    import pint
    failed_PINT = False
except ImportError as _excp:
    failed_PINT = _excp

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

    def test_omas_coordinates_intepolation(self):
        # if a coordinate exists, then that is the coordinate that it is used
        ods1 = ODS()
        ods1['equilibrium.time_slice[0].profiles_1d.psi'] = numpy.linspace(0, 1, 10)
        with omas_environment(ods1, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': numpy.linspace(0, 1, 5)}):
            ods1['equilibrium.time_slice[0].profiles_1d.f'] = numpy.linspace(0, 1, 5)
        assert (len(ods1['equilibrium.time_slice[0].profiles_1d.f']) == 10)

        # if a does not exists, then that coordinate is set
        ods2 = ODS()
        with omas_environment(ods2, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': numpy.linspace(0, 1, 5)}):
            ods2['equilibrium.time_slice[0].profiles_1d.pressure'] = numpy.linspace(0, 1, 5)
        assert (len(ods2['equilibrium.time_slice[0].profiles_1d.pressure']) == 5)

        # coordinates can be taken from existing ODSs
        ods3 = ODS()
        with omas_environment(ods3, coordsio=ods1):
            ods3['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
        with omas_environment(ods3, coordsio=ods2):
            ods3['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2[
                'equilibrium.time_slice[0].profiles_1d.pressure']
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10)
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.pressure']) == 10)

        # order matters
        ods4 = ODS()
        with omas_environment(ods4, coordsio=ods2):
            ods4['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2[
                'equilibrium.time_slice[0].profiles_1d.pressure']
        with omas_environment(ods4, coordsio=ods1):
            ods4['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
        assert (len(ods4['equilibrium.time_slice[0].profiles_1d.f']) == 5)
        assert (len(ods4['equilibrium.time_slice[0].profiles_1d.pressure']) == 5)

        # ods can be queried on different coordinates
        with omas_environment(ods4, coordsio=ods1):
            assert(len(ods4['equilibrium.time_slice[0].profiles_1d.f'])==10)
        assert(len(ods4['equilibrium.time_slice[0].profiles_1d.f'])==5)

    @unittest.skipUnless(not failed_PINT, str(failed_PINT))
    def test_handle_units(self):
        ods=ODS()

        ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement'] = 8.0 * ureg.milliseconds
        assert(ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement']==0.008)

        with omas_environment(ods, unitsio=True):
            tmp=ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement']
            assert(tmp.magnitude==0.008)
            assert(tmp.units=='second')

if __name__ == '__main__':
    unittest.main()
