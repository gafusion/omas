#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_plot.py
python -m unittest test_omas_plot
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
    show_all_plots = False  # This will get in the way of automatic testing
    show_inspectable_plots = False  # Shows plots that a human could check sometimes. Also a problem for auto-testing.
    verbose = False

    # Sample data for use in tests
    ods = ODS()
    ods.sample_equilibrium()

    x = numpy.linspace(0, 1.6, 25)
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
        self.printv('TestOmasPlot.test_ch_count...')
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

        self.printv('  TestOmasPlot.test_ch_count done.')

    def test_uband(self):
        from omas.omas_plot import uband
        self.printv('TestOmasPlot.test_uband...')
        fig, ax = plt.subplots(1)
        ub1 = uband(self.x, self.u, ax)
        ub2 = uband(self.x, -self.u, fill_kw=dict(alpha=0.15, color='k'), color='r')
        assert ub1 != ub2
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_uband done.')

    def test_all_overlays(self):
        self.printv('TestOmasPlot.test_all_overlays...')
        plt.figure()
        ods2 = copy.deepcopy(self.ods)
        for hw_sys in list_structures(ods2.imas_version):
            try:
                sample_func = getattr(ODS, 'sample_{}'.format(hw_sys))
                ods2 = sample_func(ods2)
            except AttributeError:
                pass
        ods2.plot_overlay(debug_all_plots=True)
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_all_overlays done')

    # Equilibrium cross section plot
    def test_eqcx(self):
        self.printv('TestOmasPlot.test_eqcx...')
        self.ods.plot_equilibrium_CX()
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_eqcx done')

    # PF active overlay
    def test_pf_active_overlay(self):
        self.printv('TestOmasPlot.test_pf_active_overlay...')
        # Basic test
        pf_ods = copy.deepcopy(self.ods)
        pf_ods.sample_pf_active()
        pf_ods.plot_overlay(thomson_scattering=False, pf_active=True)
        # Test keywords
        pf_ods.plot_overlay(thomson_scattering=False, pf_active=dict(facecolor='r'))
        # Test direct call
        pf_ods.plot_pf_active_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=True)
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_pf_active_overlay done')

    # Thomson scattering overlay
    def test_ts_overlay(self):
        self.printv('TestOmasPlot.test_ts_overlay...')
        # Basic test
        ts_ods = copy.deepcopy(self.ods)
        ts_ods.sample_thomson_scattering()
        ts_ods.plot_overlay(thomson_scattering=True)
        # Test direct call
        ts_ods.plot_thomson_scattering_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=True)
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_ts_overlay done')

    def test_ts_overlay_mask(self):
        self.printv('TestOmasPlot.test_ts_overlay_mask...')
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
        if self.show_all_plots or self.show_inspectable_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_ts_overlay_mask done')

    def test_ts_overlay_labels(self):
        self.printv('TestOmasPlot.test_ts_overlay_labels...')
        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering()
        for i, lab in enumerate([2, 3, 5, 7]):
            ts_ods.plot_overlay(thomson_scattering=dict(labelevery=lab, notesize=10+i*2+lab, color='k'))
        if self.show_all_plots or self.show_inspectable_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_ts_overlay_labels done')

    # Charge exchange overlay
    def test_cer_overlay(self):
        self.printv('TestOmasPlot.test_cer_overlay...')
        # Basic test
        cer_ods = copy.deepcopy(self.ods)
        cer_ods.sample_charge_exchange()
        cer_ods.plot_overlay(thomson_scattering=False, charge_exchange=True)
        # Test direct call
        cer_ods.plot_charge_exchange_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, charge_exchange=True)
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_cer_overlay done')

    # Bolometer overlay
    def test_bolo_overlay(self):
        self.printv('TestOmasPlot.test_bolo_overlay...')
        # Basic test
        bolo_ods = copy.deepcopy(self.ods)
        bolo_ods.sample_bolometer()
        bolo_ods.plot_overlay(thomson_scattering=False, bolometer=True)
        # Test direct call
        bolo_ods.plot_bolometer_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, bolometer=True)
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_bolo_overlay done')

    def test_bolo_overlay_mask(self):
        self.printv('TestOmasPlot.test_bolo_overlay_mask...')
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
        if self.show_all_plots or self.show_inspectable_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_bolo_overlay_mask done')

    # Gas injection overlay
    def test_gas_overlay(self):
        self.printv('TestOmasPlot.test_gas_overlay...')
        # Basic test
        gas_ods = copy.deepcopy(self.ods)
        gas_ods = gas_ods.sample_gas_injection()
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=True)
        # Fancy keywords tests
        gas_ods.plot_overlay(
            thomson_scattering=False,
            gas_injection=dict(which_gas=['GASA', 'GASB'], simple_labels=True, draw_arrow=False))
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=dict(which_gas=['NON-EXISTENT GAS VALVE']))
        # Test direct call
        gas_ods.plot_gas_injection_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, gas_injection=True)
        # Test without equilibrium data: can't use magnetic axis to help decide how to align labels
        ODS().sample_gas_injection().plot_overlay(thomson_scattering=False, gas_injection=True)
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_gas_overlay done')


if __name__ == '__main__':
    unittest.main()
