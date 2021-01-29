#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_plot.py

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_plot

-------
"""

# Basic imports
import os
import numpy
import warnings
import copy

# Use Agg backend to avoid opening up figures
import matplotlib

interactive = False

if not interactive:
    matplotlib.use('Agg')
from matplotlib import pyplot

# OMAS imports
from omas import *
from omas.omas_utils import *
from omas.tests import warning_setup


class TestOmasPlot(UnittestCaseOmas):
    """
    Test suite for omas_plot.py
    """

    # Sample data for use in tests
    ods = ODS().sample()

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        if test_name not in ['TestOmasPlot.test_ch_count']:
            self.fig = pyplot.figure(test_name)

    def tearDown(self):
        if interactive:
            pyplot.show()
        pyplot.close()

    def test_quantity(self):
        self.ods.plot_quantity('core_profiles.profiles_1d.0.electrons.density_thermal', '$n_e$', lw=2)
        self.ods.plot_quantity('@core.*elec.*dens', '$n_e$', lw=2)
        try:
            self.ods.plot_quantity('@core.*')
        except ValueError:
            pass
        try:
            self.ods.plot_quantity('core.*')
        except LookupError:
            pass
        omas_plot.quantity(self.ods, '@core.*ion.0.*dens.*th', '$n_D$')
        omas_plot.quantity(self.ods, '@core.*ion.1.*dens.*th', '$n_C$')

    # Support functions, utilities, and general overlay tests
    def test_ch_count(self):
        from omas.omas_plot import get_channel_count

        nc = 10
        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering(nc=nc)
        nc_ts = get_channel_count(ts_ods, 'thomson_scattering')
        assert nc_ts == nc

        empty_ods = ODS()
        nc_empty = get_channel_count(empty_ods, 'thomson_scattering')
        assert nc_empty == 0

        nc_ts_check_pass = get_channel_count(
            ts_ods, 'thomson_scattering', check_loc='thomson_scattering.channel.0.position.r', test_checker='checker > 0'
        )
        assert nc_ts_check_pass == nc

        nc_ts_check_fail = get_channel_count(
            ts_ods, 'thomson_scattering', check_loc='thomson_scattering.channel.0.position.r', test_checker='checker < 0'
        )
        assert nc_ts_check_fail == 0

        nc_ts_check_fail2 = get_channel_count(
            ts_ods, 'thomson_scattering', check_loc='thomson_scattering.channel.0.n_e.data', test_checker='checker > 0'
        )
        assert nc_ts_check_fail2 == 0

    def test_uband(self):
        from omas.omas_plot import uband

        x = numpy.linspace(0, 1.6, 25)
        xe = (x[1] - x[0]) * 0.75 + x * 0
        ux = unumpy.uarray(x, xe)

        y = 2 * x ** 2
        e = 0.1 + y * 0.01 + x * 0.01
        u = unumpy.uarray(y, e)

        ax = pyplot.gca()
        ub1 = uband(x, u, ax)
        ub2 = uband(x, -u, fill_kw=dict(alpha=0.15, color='k'), color='r')
        assert ub1 != ub2
        ub3 = uband(ux, u)
        ub4 = uband(ux, y)
        assert ub3 != ub4
        assert ub1 != ub3

    def test_all_overlays(self):
        ods2 = copy.deepcopy(self.ods)
        ods2.plot_overlay(debug_all_plots=True)

    def test_gas_arrow(self):
        from omas.omas_plot import gas_arrow

        # Explicitly test the direction keyword
        gas_arrow(self.ods, 1.5, 0.0, direction=0, color='k')
        gas_arrow(self.ods, 1.5, 0.0, direction=numpy.pi / 2, color='gray')
        gas_arrow(self.ods, 1.5, 0.0, direction=-numpy.pi / 4.5, color='m')

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
    def test_eqcx_basic(self):
        """Our basic ods comes with eq data, so try to just plot that thing"""
        self.ods.plot_equilibrium_CX()

    def test_eqcx_data_availability_variations(self):
        """Plot all the equilibrium contour quantity options with all the combinations of available data"""
        cq_options = ['rho_tor_norm', 'psi', 'phi', 'q']
        for iwall in [True, False]:
            for ipsi in [True, False]:
                for iphi in [True, False]:
                    for iprof in [True, False]:
                        for iq in [True, False]:
                            ods = ODS().sample_equilibrium(
                                include_profiles=iprof,
                                include_phi=iphi,
                                include_psi=ipsi,
                                include_wall=iwall,
                                include_q=iq,
                                include_xpoint=ipsi,  # This doesn't need an independent scan
                            )
                            if iprof and not ipsi:
                                # Try to trip up the X-point plotter by putting in an incomplete definition. This
                                # access attempt should cause the number of X-points to be interpreted as 1, even though
                                # there is no data and so the X-point is not defined.
                                ods['equilibrium.time_slice.0.boundary.x_point.0']

                            for cqo in cq_options:
                                ods.plot_equilibrium_CX(contour_quantity=cqo, allow_fallback=True)

        # Test for disallowed fallback
        ods = ODS().sample_equilibrium(include_psi=False, include_phi=True)
        with self.assertRaises(ValueError):
            # Fails because we prepared a sample with no psi, then asked for psi and did not allow fallback
            ods.plot_equilibrium_CX(contour_quantity='psi', allow_fallback=False)

        ods = ODS().sample_equilibrium(include_phi=False, include_psi=True)
        with self.assertRaises(ValueError):
            # Fails because we prepared a sample with no phi, then asked for phi and did not allow fallback
            ods.plot_equilibrium_CX(contour_quantity='phi', allow_fallback=False)

        ods = ODS().sample_equilibrium(include_phi=True, include_psi=False, include_q=True)
        with self.assertRaises(ValueError):
            # Fails because we prepped sample w/ no psi, then asked for q (uses psi for interp) & didn't allow fallback
            ods.plot_equilibrium_CX(contour_quantity='q', allow_fallback=False)

        ods = ODS().sample_equilibrium(include_phi=True, include_psi=True, include_q=False)
        with self.assertRaises(ValueError):
            # Fails because we prepped sample w/ no q, then asked for q & didn't allow fallback
            ods.plot_equilibrium_CX(contour_quantity='q', allow_fallback=False)

        ods = ODS().sample_equilibrium(include_phi=False, include_psi=False, include_q=False)
        with self.assertRaises(ValueError):
            # Fails because we prepared a sample with no 2D equilibrium data at all and did not allow fallback
            # Fallback in this case would allow an abort without raising an error
            ods.plot_equilibrium_CX(allow_fallback=False)

        ods = ODS().sample_equilibrium(include_phi=True, include_psi=True, include_q=True)
        with self.assertRaises(ValueError):
            # Fails because we ask for junk.
            ods.plot_equilibrium_CX(contour_quantity='__not_existing_quantity__', allow_fallback=False)

    def test_eqcx_slices(self):
        """Test dealing with different time indices, including getting wall from a different slice than the eq"""
        ods2 = ODS().sample_equilibrium(time_index=0, include_wall=True)
        ods2.sample_equilibrium(time_index=1, include_wall=False).plot_equilibrium_CX()  # Get wall from slice 0
        # Test for missing wall
        pyplot.figure('TestOmasPlot.test_eqcx missing wall')
        ODS().sample_equilibrium(include_profiles=True, include_phi=False, include_wall=False).plot_equilibrium_CX()

    def test_eqcx_resample(self):
        """Test the sf (scaling factor) option"""
        self.ods.plot_equilibrium_CX(sf=1)
        self.ods.plot_equilibrium_CX(sf=3)

    def test_eq_summary(self):
        ods2 = ODS().sample_equilibrium(include_phi=False)
        ods3 = ODS().sample_equilibrium(include_profiles=True, include_phi=False, include_wall=True)
        ods2.plot_equilibrium_summary(fig=pyplot.gcf(), label='label test')
        ods3.plot_equilibrium_summary(fig=pyplot.figure('TestOmasPlot.test_eq_summary with rho'))

    def test_core_profiles(self):
        ods2 = copy.deepcopy(self.ods)
        ods2.sample_core_profiles()
        ods2.plot_core_profiles_summary(fig=pyplot.figure())
        ods2.plot_core_profiles_summary(
            fig=pyplot.figure(), quantities=['temperature', 'density_thermal', 'j_tor', 'zeff'], ods_species=[-1, 0], lw=3, ls='--'
        )

    def test_core_sources(self):
        ods2 = copy.deepcopy(self.ods)
        ods2.sample_core_sources()
        ods2.plot_core_sources_summary(fig=pyplot.figure())

    def test_core_transport(self):
        ods_test = copy.deepcopy(self.ods)
        ods_test.sample_core_transport()
        ods_test['core_transport.ids_properties.comment'] = "TGRYO"
        ods_test.sample_core_profiles()
        ods_test.sample_equilibrium()
        ods_test.plot_core_transport_fluxes()
        ods_test.plot_core_transport_fluxes(show_total_density=False, label='test')
        ods_test.plot_core_transport_fluxes(plot_zeff=True, label="test")

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
        mag_ods.plot_overlay(thomson_scattering=False, magnetics=dict(show_flux_loop=False, labelevery=1))
        mag_ods.plot_overlay(thomson_scattering=False, magnetics=dict(show_bpol_probe=False, notesize=9, labelevery=1))
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
        from omas.omas_plot import get_channel_count

        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering()
        nc = get_channel_count(ts_ods, 'thomson_scattering')
        mask0 = numpy.ones(nc, bool)
        markers = ['.', '^', '>', 'v', '<', 'o', 'd', '*', 's', '|', '_', 'x']
        markers *= int(numpy.ceil(float(nc) / len(markers)))
        for i in range(nc):
            mask = copy.copy(mask0)
            mask[i] = False
            ts_ods.plot_overlay(thomson_scattering=dict(mask=mask, marker=markers[i], mew=0.5, markersize=3 * (nc - i)))

    def test_ts_overlay_labels(self):
        ts_ods = copy.deepcopy(self.ods)
        ts_ods = ts_ods.sample_thomson_scattering()
        for i, lab in enumerate([2, 3, 5, 7]):
            ts_ods.plot_overlay(thomson_scattering=dict(labelevery=lab, notesize=10 + i * 2 + lab, color='k'))
        ts_ods.plot_overlay(thomson_scattering=dict(labelevery=2, notesize=9, color='b', label_ha='right', label_va='top'))

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
            charge_exchange=dict(which_pos='closest', color_tangential='r', color_vertical='b', marker_tangential='h'),
        )
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
        from omas.omas_plot import get_channel_count

        bolo_ods = copy.deepcopy(self.ods)
        bolo_ods = bolo_ods.sample_bolometer()
        nc = get_channel_count(bolo_ods, 'bolometer')
        mask0 = numpy.ones(nc, bool)
        markers = ['.', '^', '>', 'v', '<', 'o', 'd', '*', 's', '|', '_', 'x']
        markers *= int(numpy.ceil(float(nc) / len(markers)))
        for i in range(nc):
            mask = copy.copy(mask0)
            mask[i] = False
            bolo_ods.plot_overlay(
                thomson_scattering=False, bolometer=dict(mask=mask, marker=markers[i], mew=0.5, markersize=3 * (nc - i), lw=0.5 * (nc - i))
            )

    # Gas injection overlay
    def test_gas_overlay(self):
        # Basic test
        gas_ods = copy.deepcopy(self.ods)
        gas_ods = gas_ods.sample_gas_injection()
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=True)
        # Fancy keywords tests
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=dict(which_gas=['GASA', 'GASB'], draw_arrow=False))
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=dict(which_gas=['FAKE_GAS_A', 'FAKE_GAS_B'], draw_arrow=False))
        gas_ods.plot_overlay(thomson_scattering=False, gas_injection=dict(which_gas=['NON-EXISTENT GAS VALVE']))
        gas_ods.plot_overlay(
            thomson_scattering=False, gas_injection=dict(angle_not_in_pipe_name=True, simple_labels=True, show_all_pipes_in_group=False)
        )

        # Test direct call
        gas_ods.plot_gas_injection_overlay()
        # Test empty one; make sure fail is graceful
        ODS().plot_overlay(thomson_scattering=False, gas_injection=True)
        # Test without equilibrium data: can't use magnetic axis to help decide how to align labels
        ODS().sample_gas_injection().plot_overlay(thomson_scattering=False, gas_injection=True)

    # Langmuir probes overlays
    def test_langmuir_probes_embedded_overlay(self):
        """Tests method for plotting overlay of embedded LPs"""
        # Add sample data
        lp_ods = ODS()
        lp_ods.sample_wall()  # The wall is used to decide label alignment
        lp_ods.sample_langmuir_probes()

        # Basic overlay
        lp_ods.plot_overlay(thomson_scattering=False, langmuir_probes=True)

        # Overlay with customizations
        lp_ods.plot_overlay(
            thomson_scattering=False, langmuir_probes=dict(colors='r', label_ha='left', label_va='top', embedded_probes=['donkey!', 'zzz'])
        )

        # Direct call
        lp_ods.plot_langmuir_probes_overlay()
        # Empty ODS / graceful failure
        ODS().plot_langmuir_probes_overlay()
        # No wall data for helping align labels
        ODS().sample_langmuir_probes().plot_overlay(thomson_scattering=False, langmuir_probes=True)

    def test_position_control_overlay(self):
        """Tests method for plotting overlay of position_control data"""

        pc_ods = ODS()

        # No data; have to abort
        pc_ods.plot_overlay(thomson_scattering=False, position_control=True)

        # Add sample data
        pc_ods.sample_pulse_schedule()

        # Basic test
        pc_ods.plot_overlay(thomson_scattering=False, position_control=True)

        # Multi-time
        pc_ods.plot_overlay(thomson_scattering=False, position_control=dict(t=[0.5, 2.3]))

        # Now with equilibrium data (magnetic axis position affects label alignment
        pc_ods.sample_equilibrium()
        pc_ods.plot_overlay(thomson_scattering=False, position_control=dict(t=2.3, show_measured_xpoint=True))
        pc_ods.sample_equilibrium(include_xpoint=True)
        pc_ods.plot_overlay(thomson_scattering=False, position_control=dict(t=2.3, show_measured_xpoint=True))
        pc_ods.sample_equilibrium(include_xpoint=True, time_index=1)
        pc_ods.plot_overlay(thomson_scattering=False, position_control=dict(t=2.3, show_measured_xpoint=True))

        # Call the method itself
        pc_ods.plot_position_control_overlay()

        # Call with mask
        nb = len(pc_ods['pulse_schedule']['position_control']['boundary_outline'])
        nx = len(pc_ods['pulse_schedule']['position_control']['x_point'])
        ns = len(pc_ods['pulse_schedule']['position_control']['strike_point'])
        nn = nb + nx + ns
        mask = [False] * nn
        mask[nn // 2] = True
        mask[0] = True
        mask[-1] = True
        pc_ods.plot_position_control_overlay(mask=mask)
        # Force the overlay function to extend the mask by giving it one that's too small
        small_mask = mask[:-2]
        small_mask[-1] = True
        pc_ods.plot_position_control_overlay(mask=small_mask)

    def test_pulse_schedule_overlay(self):
        """
        Tests method for plotting overlay of pulse_schedule data.
        The main item here is position_control, which has its own test. So, this will be short.
        """
        import time

        pc_ods = ODS()
        pc_ods.sample_pulse_schedule()
        pc_ods.plot_overlay(thomson_scattering=False, pulse_schedule=dict(timing_ref=time.time()))
        pc_ods.plot_pulse_schedule_overlay()

    def test_ec_launchers_overlay(self):
        """Tests several plotting methods for showing EC launchers data"""
        # Prepare sample data
        ods = ODS()
        ods.sample_equilibrium()
        ods.sample_ec_launchers()
        # Test plots with default/minimal options
        ods.plot_ec_launchers_CX()
        ods.plot_ec_launchers_CX_topview()

    def test_nbi(self):
        """Tests basic NBI plots"""
        # Prep sample data
        ods = ODS()
        ods.sample_nbi()
        # Test relevant plots with basic/default/minimal options
        ods.plot_nbi_summary()

    def test_ods_time_plot(self):
        # generate some time-dependent data
        ods = ODS().sample(2)

        # omas plot for pressures
        ods.plot_core_profiles_pressures()

        # omas plot for core profiles
        ods.plot_core_profiles_summary()

        # omas plot for equilibrium
        omas_plot.equilibrium_summary(ods, linewidth=1, label='my equilibrium')

        # omas plot for transport fluxes
        ods.plot_core_transport_fluxes()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasPlot)
    unittest.TextTestRunner(verbosity=2).run(suite)
