#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_plot.py
python -m unittest test_omas_plot (from omas/omas/tests)
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


class TestOmasPlot(unittest.TestCase):
    """
    Test suite for omas_plot.py
    """

    # Flags to edit while testing
    show_plots = False  # This will get in the way of automatic testing
    keep_plots_open = False
    verbose = False  # Spammy, but occasionally useful for debugging a weird problem

    # Sample data for use in tests
    ods = ODS()
    ods.sample_equilibrium()
    ods.sample_core_profiles()

    x = numpy.linspace(0, 1.6, 25)
    xe = (x[1]-x[0])*0.75 + x * 0
    ux = unumpy.uarray(x, xe)

    y = 2*x**2
    e = 0.1 + y*0.01 + x*0.01
    u = unumpy.uarray(y, e)

    # Utilities for this test
    def printv(self, *arg):
        """Utility for tests to use"""
        if self.verbose:
            print(*arg)

    # Support functions, utilities, and general overlay tests
    def test_ch_count(self):
        nc = 10
        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering(nc=nc)
        nc_ts = ts_ods.plot_get_channel_count('thomson_scattering')
        assert nc_ts == nc

        empty_ods = ODS()
        nc_empty = empty_ods.plot_get_channel_count('thomson_scattering')
        assert nc_empty == 0

        nc_ts_check_pass = ts_ods.plot_get_channel_count(
            'thomson_scattering', check_loc='thomson_scattering.channel.0.position.r', test_checker='checker > 0')
        assert nc_ts_check_pass == nc

        nc_ts_check_fail = ts_ods.plot_get_channel_count(
            'thomson_scattering', check_loc='thomson_scattering.channel.0.position.r', test_checker='checker < 0')
        assert nc_ts_check_fail == 0

        nc_ts_check_fail2 = ts_ods.plot_get_channel_count(
            'thomson_scattering', check_loc='thomson_scattering.channel.0.n_e.data', test_checker='checker > 0')
        assert nc_ts_check_fail2 == 0

    def test_uband(self):
        from omas.omas_plot import uband
        ax = plt.gca()
        ub1 = uband(self.x, self.u, ax)
        ub2 = uband(self.x, -self.u, fill_kw=dict(alpha=0.15, color='k'), color='r')
        assert ub1 != ub2
        ub3 = uband(self.ux, self.u)
        ub4 = uband(self.ux, self.y)
        assert ub3 != ub4
        assert ub1 != ub3

    def test_all_overlays(self):
        ods2 = copy.deepcopy(self.ods)
        for hw_sys in list_structures(ods2.imas_version):
            try:
                sample_func = getattr(ODS, 'sample_{}'.format(hw_sys))
                ods2 = sample_func(ods2)
            except AttributeError:
                pass
        ods2.plot_overlay(debug_all_plots=True)

    def test_gas_arrow(self):
        from omas.omas_plot import gas_arrow
        self.printv('  gas_arrow ods.cocos = {}'.format(self.ods.cocos))
        # Explicitly test the direction keyword
        gas_arrow(self.ods, 1.5, 0.0, direction=0, color='k')
        gas_arrow(self.ods, 1.5, 0.0, direction=numpy.pi/2, color='gray')
        gas_arrow(self.ods, 1.5, 0.0, direction=-numpy.pi/4.5, color='m')

    def test_geo_type_lookup(self):
        from omas.omas_plot import geo_type_lookup
        # Basic tests
        assert geo_type_lookup(1, 'pf_active', imas_version='3.19.0', reverse=False) == 'outline'
        assert geo_type_lookup(1, 'pf_active', imas_version='3.18.0', reverse=False) == 'outline'
        assert geo_type_lookup(1, 'pf_active', reverse=False) == 'outline'
        assert geo_type_lookup('outline', 'pf_active', imas_version='3.19.0', reverse=True) == 1
        assert geo_type_lookup(2, 'pf_active', imas_version='3.19.0', reverse=False) == 'rectangle'
        assert geo_type_lookup(4, 'pf_active', imas_version='3.19.0', reverse=False) == 'arcs of circle'
        assert geo_type_lookup('rectangle', 'pf_active', imas_version='3.19.0', reverse=True) == 2
        # Test handling of problem cases
        assert geo_type_lookup(1, 'unrecognized_nonsense_fail', imas_version=None, reverse=False) is None
        assert geo_type_lookup(1, 'pf_active', imas_version='99.99.99', reverse=False) is None

    # Equilibrium plots
    def test_eqcx(self):
        self.ods.plot_equilibrium_CX()
        ods2 = ODS().sample_equilibrium(include_profiles=False, include_phi=True, include_wall=False)
        ods2.plot_equilibrium_CX()  # Should be vs. rho this time
        ods2.sample_equilibrium(time_index=1, include_profiles=False, include_phi=False, include_wall=False).plot_equilibrium_CX()  # Get wall from slice 0
        plt.figure('TestOmasPlot.test_eqcx missing wall')
        ODS().sample_equilibrium(include_profiles=True, include_phi=False, include_wall=False).plot_equilibrium_CX()  # No wall

    def test_eq_summary(self):
        ods2 = ODS().sample_equilibrium(include_phi=False)
        ods3 = ODS().sample_equilibrium(include_profiles=True, include_phi=False, include_wall=True)
        ods2.plot_equilibrium_summary(fig=plt.gcf(), label='label test')
        ods3.plot_equilibrium_summary(fig=plt.figure('TestOmasPlot.test_eq_summary with rho'))

    def test_core_profiles(self):
        ods2 = copy.deepcopy(self.ods)
        ods2.sample_core_profiles()
        ods2.plot_core_profiles_summary(fig=plt.gcf())
        ods2.plot_core_profiles_summary(
            fig=plt.figure('TestOmasPlot.test_core_profiles totals only'), show_thermal_fast_breakdown=False,
            show_total_density=True)
        ods2.plot_core_profiles_summary(
            fig=plt.figure('TestOmasPlot.test_core_profiles total and breakdown'), show_thermal_fast_breakdown=True,
            show_total_density=True)
        ods2.plot_core_profiles_summary(
            fig=plt.figure('TestOmasPlot.test_core_profiles no combine temp/dens'), combine_dens_temps=False)

    def test_core_pressure(self):
        ods2 = copy.deepcopy(self.ods)
        ods2.sample_core_profiles()
        ods2.plot_core_profiles_pressures()
        ods3 = copy.deepcopy(self.ods)
        ods3.sample_core_profiles(add_junk_ion=True)
        ods3.plot_core_profiles_pressures()

    # PF active overlay
    def test_pf_active_overlay(self):
        # Basic test
        pf_ods = copy.deepcopy(self.ods)
        pf_ods.sample_pf_active(nc_weird=1, nc_undefined=1)
        pf_ods.plot_overlay(thomson_scattering=False, pf_active=True)
        # Test keywords
        pf_ods.plot_overlay(thomson_scattering=False, pf_active=dict(facecolor='r', labelevery=1))
        # Test direct call
        pf_ods.plot_pf_active_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=True, pf_active=True)
        ODS().plot_pf_active_overlay()

    # Magnetics overlay
    def test_magnetics_overlay(self):
        # Basic test
        mag_ods = copy.deepcopy(self.ods)
        mag_ods.sample_magnetics()
        mag_ods.plot_overlay(thomson_scattering=False, magnetics=True)
        # Test keywords
        mag_ods.plot_overlay(
            thomson_scattering=False,
            magnetics=dict(bpol_probe_color='r', bpol_probe_marker='x', show_flux_loop=False, labelevery=1))
        mag_ods.plot_overlay(
            thomson_scattering=False,
            magnetics=dict(bpol_probe_color='m', flux_loop_marker='+', show_bpol_probe=False, notesize=9, labelevery=1))
        # Test direct call
        mag_ods.plot_magnetics_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=True, magnetics=True)
        ODS().plot_magnetics_overlay()

    # Thomson scattering overlay
    def test_ts_overlay(self):
        # Basic test
        ts_ods = copy.deepcopy(self.ods)
        ts_ods.sample_thomson_scattering()
        ts_ods.plot_overlay(thomson_scattering=True)
        # Test direct call
        ts_ods.plot_thomson_scattering_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=True)

    def test_ts_overlay_mask(self):
        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering()
        nc = ts_ods.plot_get_channel_count('thomson_scattering')
        mask0 = numpy.ones(nc, bool)
        markers = ['.', '^', '>', 'v', '<', 'o', 'd', '*', 's', '|', '_', 'x']
        markers *= int(numpy.ceil(float(nc)/len(markers)))
        for i in range(nc):
            mask = copy.copy(mask0)
            mask[i] = False
            ts_ods.plot_overlay(thomson_scattering=dict(mask=mask, marker=markers[i], mew=0.5, markersize=3*(nc-i)))

    def test_ts_overlay_labels(self):
        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering()
        for i, lab in enumerate([2, 3, 5, 7]):
            ts_ods.plot_overlay(thomson_scattering=dict(labelevery=lab, notesize=10+i*2+lab, color='k'))

    # Charge exchange overlay
    def test_cer_overlay(self):
        # Basic test
        cer_ods = copy.deepcopy(self.ods)
        cer_ods.sample_charge_exchange()
        cer_ods.plot_overlay(thomson_scattering=False, charge_exchange=True)
        # Keywords
        cer_ods.plot_overlay(thomson_scattering=False, charge_exchange=dict(which_pos='all'))
        cer_ods.plot_overlay(
            thomson_scattering=False,
            charge_exchange=dict(which_pos='closest', color_tangential='r', color_vertical='b', marker_tangential='h'))
        # Make a fresh copy with no EQ data so it will trigger the exception and fall back to which_pos='all'
        ODS().sample_charge_exchange().plot_overlay(thomson_scattering=False, charge_exchange=dict(which_pos='closest'))
        # Test direct call
        cer_ods.plot_charge_exchange_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, charge_exchange=True)

    # Inteferometer overlay
    def test_interferometer_overlay(self):
        # Basic test
        intf_ods = copy.deepcopy(self.ods)
        intf_ods.sample_interferometer()
        intf_ods.plot_overlay(thomson_scattering=False, interferometer=True)
        # Test direct call
        intf_ods.plot_interferometer_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, interferometer=True)

    # Bolometer overlay
    def test_bolo_overlay(self):
        # Basic test
        bolo_ods = copy.deepcopy(self.ods)
        bolo_ods.sample_bolometer()
        bolo_ods.plot_overlay(thomson_scattering=False, bolometer=True)
        # Keywords
        bolo_ods.plot_overlay(thomson_scattering=False, bolometer=dict(colors='rgb', reset_fan_color=True))
        # Test direct call
        bolo_ods.plot_bolometer_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, bolometer=True)

    def test_bolo_overlay_mask(self):
        bolo_ods = copy.deepcopy(self.ods)
        bolo_ods = bolo_ods.sample_bolometer()
        nc = bolo_ods.plot_get_channel_count('bolometer')
        mask0 = numpy.ones(nc, bool)
        markers = ['.', '^', '>', 'v', '<', 'o', 'd', '*', 's', '|', '_', 'x']
        markers *= int(numpy.ceil(float(nc) / len(markers)))
        for i in range(nc):
            mask = copy.copy(mask0)
            mask[i] = False
            bolo_ods.plot_overlay(
                thomson_scattering=False,
                bolometer=dict(mask=mask, marker=markers[i], mew=0.5, markersize=3*(nc-i), lw=0.5*(nc-i)))

    # Gas injection overlay
    def test_gas_overlay(self):
        # Basic test
        gas_ods = copy.deepcopy(self.ods)
        gas_ods = gas_ods.sample_gas_injection()
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=True)
        # Fancy keywords tests
        gas_ods.plot_overlay(
            thomson_scattering=False,
            gas_injection=dict(which_gas=['GASA', 'GASB'], draw_arrow=False))
        gas_ods.plot_overlay(
            thomson_scattering=False,
            gas_injection=dict(which_gas=['FAKE_GAS_A', 'FAKE_GAS_B'], draw_arrow=False))
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=dict(which_gas=['NON-EXISTENT GAS VALVE']))
        gas_ods.plot_overlay(
            thomson_scattering=False, gas_injection=dict(angle_not_in_pipe_name=True, simple_labels=True))

        # Test direct call
        gas_ods.plot_gas_injection_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, gas_injection=True)
        # Test without equilibrium data: can't use magnetic axis to help decide how to align labels
        ODS().sample_gas_injection().plot_overlay(thomson_scattering=False, gas_injection=True)

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        if test_name not in ['TestOmasPlot.test_ch_count']:
            self.fig = plt.figure(test_name)
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))
        if not self.keep_plots_open:
            plt.close()

    @classmethod
    def tearDownClass(cls):
        if cls.show_plots:
            plt.show()


if __name__ == '__main__':
    unittest.main()
