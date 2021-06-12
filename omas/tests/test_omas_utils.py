#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_utils.py

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_utils

-------
"""

import os
import numpy
import warnings
import copy

# OMAS imports
from omas import *
from omas.omas_utils import *
from omas.tests import warning_setup


class TestOmasUtils(UnittestCaseOmas):
    """
    Test suite for omas_utils.py
    """

    # Sample data for use in tests
    specific_test_version = '3.18.0'

    def test_different_ods(self):
        ods = ODS()
        ods2 = ODS()
        ods2.sample_equilibrium()
        diff_eq = different_ods(ods, ods2)
        assert isinstance(diff_eq, list)
        assert ('equilibrium' in ' '.join(diff_eq)) or ('wall' in ' '.join(diff_eq))
        ods3 = copy.deepcopy(ods2)
        assert different_ods(ods2, ods3) is False
        ods3.sample_core_profiles()
        diff_prof = ods2.diff(ods3)
        assert isinstance(diff_prof, list)
        assert isinstance(different_ods(ods3, ods2), list)
        assert 'core_profiles' in ' '.join(diff_prof)
        ods2.sample_core_profiles(include_pressure=False)
        diff_prof2 = different_ods(ods3, ods2)
        assert isinstance(diff_prof2, list)
        assert 'core_profiles' in ' '.join(diff_prof2)
        ods2.sample_core_profiles()
        ods2['core_profiles.profiles_1d.0.electrons.density_thermal'][0] = 1.5212
        diff_prof3 = ods2.diff(ods3)
        assert isinstance(diff_prof3, list)
        assert 'value' in ' '.join(diff_prof3)
        ods2.sample_core_profiles()
        ods2['core_profiles.profiles_1d.0.ion.0.element.0.a'] = 2
        diff_prof4 = different_ods(ods2, ods3)
        assert not diff_prof4
        ods2.sample_core_profiles()
        ods2['core_profiles.code.name'] = 'fake name 1'
        ods3['core_profiles.code.name'] = 'fake name 2'
        diff_prof5 = different_ods(ods2, ods3)
        assert isinstance(diff_prof5, list)
        assert 'name' in ' '.join(diff_prof5)

    def test_printe(self):
        printe('printe_test,', end='')

    def test_is_numeric(self):
        assert is_numeric(5) is True
        assert is_numeric(numpy.array([5])) is True
        assert is_numeric('blah') is False
        assert is_numeric({'blah': 'blah'}) is False
        assert is_numeric([]) is False
        assert is_numeric(None) is False

    def test_remove_parentheses(self):
        assert remove_parentheses('zoom(b(la)h)what', replace_with='|') == 'zoom|what'

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
        self.assertRaises(TypeError, closest_index, [1, 2, 3], [3, 2, 1])  # Can't call w/ list as 2nd arg unless len=1

    def test_list_structures(self):  # Also tests structures_filenames
        struct_list = list_structures(omas_rcparams['default_imas_version'])
        struct_list2 = list_structures(self.specific_test_version)
        assert isinstance(struct_list, list)
        assert isinstance(struct_list2, list)
        assert isinstance(struct_list[0], str)
        assert 'pf_active' in struct_list2
        struct_dict = structures_filenames(omas_rcparams['default_imas_version'])
        struct_dict2 = structures_filenames(self.specific_test_version)
        assert isinstance(struct_dict, dict)
        assert isinstance(struct_dict2, dict)
        assert 'pf_active' in struct_dict2.keys()
        assert all(item in struct_dict.keys() for item in struct_list)
        assert all(item in struct_dict2.keys() for item in struct_list2)

    def test_omas_info(self):
        get_list = ['pf_active', 'thomson_scattering', 'charge_exchange']
        ods_info_pfa = omas_info(get_list[0])
        ods_info_pfa2 = omas_info(get_list[0], self.specific_test_version)
        assert get_list[0] in ods_info_pfa.keys()
        assert get_list[0] in ods_info_pfa2.keys()
        if get_list[0] == 'pf_active':
            assert isinstance(ods_info_pfa['pf_active.circuit.0.connections.documentation'], str)
        ods_info_list = omas_info(get_list)
        assert all(item in ods_info_list for item in get_list)

    def test_p2l(self):
        assert p2l('0') == [0]
        assert p2l('equilibrium') == ['equilibrium']
        assert p2l('equilibrium.time_slice.0.global_quantities.ip') == ['equilibrium', 'time_slice', 0, 'global_quantities', 'ip']

    def test_o2u(self):
        assert o2u('equilibrium.time_slice.0.global_quantities.ip') == 'equilibrium.time_slice.:.global_quantities.ip'
        assert o2u('equilibrium.time_slice.:.global_quantities.ip') == 'equilibrium.time_slice.:.global_quantities.ip'
        assert o2u('2') == ':'
        assert o2u('equilibrium') == 'equilibrium'
        assert o2u('equilibrium.2') == 'equilibrium.:'

    def test_set_time_array(self):
        ods = ODS()
        ods.set_time_array('equilibrium.vacuum_toroidal_field.b0', 0, 0.1)
        ods.set_time_array('equilibrium.vacuum_toroidal_field.b0', 1, 0.2)

    def test_info(self):
        omas_info('equilibrium')
        omas_info(None)

    def test_ouarray(self):
        import numpy
        import time
        import copy
        import pickle
        from omas import ODS, omas_environment
        from omas.omas_utils import is_uncertain
        from uncertainties import ufloat, unumpy
        from uncertainties.unumpy import uarray, nominal_values, std_devs
        from pprint import pprint

        n = 1001
        v = numpy.linspace(0, 1, n)
        u = v * 2
        l = v * 3

        # uncertainty package is cool but slow
        t0 = time.time()
        tmp = uarray(v, u)
        tu = time.time() - t0

        assert all(nominal_values(tmp) == v)
        assert all(std_devs(tmp) == u)

        ods = ODS()
        ods['equilibrium.time'] = [0.0]

        # float separate
        ods['equilibrium.time_slice[0].global_quantities.ip'] = 3
        ods['equilibrium.time_slice[0].global_quantities.ip_error_upper'] = 0.1
        # ufloat
        ods['equilibrium.time_slice[1].global_quantities.ip'] = ufloat(3, 0.1)
        assert 'equilibrium.time_slice[1].global_quantities.ip_error_upper' in ods
        assert (
            ods['equilibrium.time_slice[1].global_quantities.ip_error_upper']
            == ods['equilibrium.time_slice[0].global_quantities.ip_error_upper']
        )

        # array separate
        ods['thomson_scattering.channel[0].t_e.data'] = numpy.array([1.0, 2.0, 3.0])
        ods['thomson_scattering.channel[0].t_e.data_error_upper'] = numpy.array([0.1, 0.2, 0.3])

        # uarray
        ods['thomson_scattering.channel[1].t_e.data'] = unumpy.uarray([1, 2, 3], [0.1, 0.2, 0.3])
        assert 'thomson_scattering.channel[1].t_e.data_error_upper' in ods
        assert all(ods['thomson_scattering.channel[0].t_e.data_error_upper'] == ods['thomson_scattering.channel[01.t_e.data_error_upper'])

        with omas_environment(ods, uncertainio=True):
            assert is_uncertain(ods['thomson_scattering.channel[:].t_e.data'])
            assert is_uncertain(ods['equilibrium.time_slice[:].global_quantities.ip'])
            assert isinstance(ods['equilibrium.time_slice[0].global_quantities.ip'], uncertainties.core.AffineScalarFunc)
            assert isinstance(ods['equilibrium.time_slice[1].global_quantities.ip'], uncertainties.core.AffineScalarFunc)

    # End of TestOmasUtils class


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)
