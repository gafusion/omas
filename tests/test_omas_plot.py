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


class TestOmasPlot(unittest.TestCase):
    """
    Test suite for omas_plot.py
    """

    ods = ODS()
    ods.sample_equilibrium()

    show_all_plots = False  # This will get in the way of automatic testing
    show_inspectable_plots = False  # Shows plots that a human could check sometimes. Also a problem for auto-testing.
    verbose = False

    # Utilities for this test
    def printv(self, *arg):
        """Utility for tests to use"""
        if self.verbose:
            print(*arg)

    # Support functions
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

    # Equilibrium
    def test_eqcx(self):
        self.printv('TestOmasPlot.test_eqcx...')
        self.ods.plot_equilibrium_CX()
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_eqcx done')

    # Thomson scattering
    def test_ts_overlay(self):
        self.printv('TestOmasPlot.test_ts_overlay...')
        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering()
        ts_ods.plot_overlay(thomson_scattering=True)
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

    # Bolometers
    def test_bolo_overlay(self):
        self.printv('TestOmasPlot.test_bolo_overlay...')
        bolo_ods = copy.deepcopy(self.ods)
        bolo_ods = bolo_ods.sample_bolometer()
        bolo_ods.plot_overlay(thomson_scattering=False, bolometer=True)
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

    # Gas
    def test_gas_overlay(self):
        self.printv('TestOmasPlot.test_gas_overlay...')
        gas_ods = copy.deepcopy(self.ods)
        gas_ods = gas_ods.sample_gas_injection()
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=True)
        if self.show_all_plots:
            plt.show()
        self.printv('  TestOmasPlot.test_gas_overlay done')


if __name__ == '__main__':
    unittest.main()
