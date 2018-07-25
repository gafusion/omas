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

    def test_core_profiles_pressures(self):
        ods = ODS()
        ods.sample_profiles(include_pressure=False)
        ods2 = core_profiles_pressures(ods, update=True)

        diff = different_ods(ods, ods2)
        assert (not diff)

        ods = ODS()
        ods.sample_profiles(include_pressure=False)
        ods2 = core_profiles_pressures(ods, update=False)
        ods.update(ods2)

        diff = different_ods(ods, ods2)
        assert (diff)

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

    def test_coordsio(self):
        data5 = numpy.linspace(0, 1, 5)
        data10 = numpy.linspace(0, 1, 10)

        if self.verbose:
            os.environ['OMAS_DEBUG_TOPIC'] = 'coordsio'

        # data can be entered without any coordinate checks
        ods0 = ODS()
        ods0['equilibrium.time_slice[0].profiles_1d.psi'] = data10
        ods0['equilibrium.time_slice[0].profiles_1d.f'] = data5
        assert (len(ods0['equilibrium.time_slice[0].profiles_1d.psi']) != len(ods0['equilibrium.time_slice[0].profiles_1d.f']))

        # if a coordinate exists, then that is the coordinate that it is used
        ods1 = ODS()
        ods1['equilibrium.time_slice[0].profiles_1d.psi'] = data10
        with omas_environment(ods1, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': data5}):
            ods1['equilibrium.time_slice[0].profiles_1d.f'] = data5
        assert (len(ods1['equilibrium.time_slice[0].profiles_1d.f']) == 10)

        # if a coordinate does not exists, then it is added
        ods2 = ODS()
        with omas_environment(ods2, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': data5}):
            ods2['equilibrium.time_slice[0].profiles_1d.pressure'] = data5
        assert (len(ods2['equilibrium.time_slice[0].profiles_1d.pressure']) == 5)

        # coordinates can be easily copied over from existing ODSs with .coordinates() method
        ods3 = ODS()
        ods3.update(ods1.coordinates())
        with omas_environment(ods3, coordsio=ods1):
            ods3['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
        with omas_environment(ods3, coordsio=ods2):
            ods3['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2['equilibrium.time_slice[0].profiles_1d.pressure']
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10)
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.pressure']) == 10)

        # ods can be queried on different coordinates than they were originally filled in (ods example)
        with omas_environment(ods3, coordsio=ods2):
            assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 5)
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10)

        # ods can be queried on different coordinates than they were originally filled in (ods example)
        with omas_environment(ods3, coordsio=ods3):
            assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10)

        # ods can be queried on different coordinates than they were originally filled in (dictionary example)
        with omas_environment(ods3, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': data5}):
            assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 5)
        assert (len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10)

        # this case is different because the coordinate and the data do not share the same parent
        ods5 = ODS()
        ods5['core_profiles.profiles_1d[0].grid.rho_tor_norm'] = data5
        with omas_environment(ods5, coordsio={'core_profiles.profiles_1d[0].grid.rho_tor_norm': data10}):
            ods5['core_profiles.profiles_1d[0].electrons.density_thermal'] = data10
        assert (len(ods5['core_profiles.profiles_1d[0].grid.rho_tor_norm']) == 5)
        assert (len(ods5['core_profiles.profiles_1d[0].electrons.density_thermal']) == 5)

        ods6 = ODS()
        ods6['core_profiles.profiles_1d[0].grid.rho_tor_norm'] = data5

    def test_cocosio(self):
        x = numpy.linspace(.1, 1, 10)

        ods = ODS(cocosio=11, cocos=11)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))

        ods = ODS(cocosio=11, cocos=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))

        ods = ODS(cocosio=2, cocos=11)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))

        ods = ODS(cocosio=2, cocos=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))

        # reassign the same value
        ods = ODS(cocosio=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = ods['equilibrium.time_slice.0.profiles_1d.psi']
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))

        # use omas_environment
        ods = ODS(cocosio=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        with omas_environment(ods, cocosio=11):
            assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x*(2*numpy.pi)))

        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))

    def test_coordsio_cocosio(self):
        x = numpy.linspace(0.1, 1, 11)
        y = numpy.linspace(-1, 1, 11)

        xh = numpy.linspace(0.1, 1, 21)
        yh = numpy.linspace(-1, 1, 21)

        ods = ODS()
        with omas_environment(ods, cocosio=2):
            ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
            assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x))
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x*2*numpy.pi))

        with omas_environment(ods, cocosio=2, coordsio={'equilibrium.time_slice.0.profiles_1d.psi':xh}):
            ods['equilibrium.time_slice.0.profiles_1d.phi'] = yh
            assert (len(ods['equilibrium.time_slice.0.profiles_1d.phi']) == len(yh))
            assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], xh))
        assert (numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x*2*numpy.pi))
        assert (len(ods['equilibrium.time_slice.0.profiles_1d.phi']) == len(y))

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