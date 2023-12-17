#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_physics.py

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_physics

-------
"""

import os
import numpy
import warnings
import copy
import itertools

# OMAS imports
from omas import *
from omas.omas_utils import *
from omas.omas_physics import *
from omas.tests import warning_setup

try:
    import pint

    failed_PINT = False
except ImportError as _excp:
    failed_PINT = _excp


class TestOmasPhysics(UnittestCaseOmas):
    """
    Test suite for omas_physics.py
    """

    def test_check_iter_scenario_requirements(self):
        from omas.omas_imas import iter_scenario_requirements

        ods = ODS()
        tmp = ods.physics_check_iter_scenario_requirements()
        assert tmp == iter_scenario_requirements

        ods.sample_equilibrium()
        ods.sample_core_profiles()
        ods.physics_summary_consistent_global_quantities()
        tmp = ods.physics_check_iter_scenario_requirements()
        assert tmp != iter_scenario_requirements

    def test_equilibrium_consistent(self):
        ods = ODS()
        ods.sample_equilibrium()

        assert "energy_mhd" not in ods['equilibrium.time_slice.0.global_quantities']
        ods.physics_equilibrium_consistent()
        assert ("energy_mhd" in ods['equilibrium.time_slice.0.global_quantities']) and (
            ods['equilibrium.time_slice.0.global_quantities.energy_mhd'] > 0
        )
        return

    def test_core_profiles_pressures(self):
        ods = ODS()
        ods.sample_core_profiles(include_pressure=False)
        ods2 = ods.physics_core_profiles_pressures(update=True)
        diff = ods.diff(ods2)
        assert not diff

        ods2 = ods.physics_core_profiles_pressures(update=False)
        assert all('press' in item for item in ods2.flat().keys() if not item.endswith('rho_tor_norm'))
        return

    def test_core_profiles_currents(self):

        rho = numpy.linspace(0.0, 1.0, 4)
        Jval = 1e5 * numpy.ones(4)
        jdef = {}
        Js = ['j_actuator', 'j_bootstrap', 'j_non_inductive', 'j_ohmic', 'j_total']
        for j in Js:
            jdef[j] = 'default'

        def CPC(ods, kw=jdef, should_RE=False, should_AE=False, warn=False):

            try:
                ods.physics_core_profiles_currents(0, rho, warn=warn, **kw)
            except RuntimeError as err:
                if should_RE:
                    pass
                else:
                    print(repr(kw))
                    raise err
            except AssertionError as err:
                if should_AE:
                    pass
                else:
                    print(repr(kw))
                    raise err
            else:
                if should_RE:
                    raise RuntimeError("Should have raised RuntimeError but didn't: " + repr(kw))
                elif should_AE:
                    raise AssertionError("Should have raised AssertionError but didn't: " + repr(kw))
            return

        # Try just setting one
        for i, J1 in enumerate(Js):
            kw = copy.deepcopy(jdef)
            kw[J1] = Jval
            CPC(ODS(), kw=kw, should_AE=(J1 == 'j_actuator'))

            # Now try setting two
            for J2 in Js[i + 1 :]:
                kw = copy.deepcopy(jdef)
                kw[J1] = Jval
                kw[J2] = Jval
                should_AE = (
                    (not isinstance(kw['j_actuator'], str) or kw['j_actuator'] != 'default')
                    and (isinstance(kw['j_bootstrap'], str) and kw['j_bootstrap'] == 'default')
                    and (isinstance(kw['j_non_inductive'], str) and kw['j_non_inductive'] == 'default')
                )
                CPC(ODS(), kw=kw, should_AE=should_AE)

        # Try setting three
        for keys in list(itertools.combinations(Js, 3)):
            kw = copy.deepcopy(jdef)
            for key in keys:
                kw[key] = Jval
            if ('j_actuator' in keys) and ('j_bootstrap' in keys) and ('j_non_inductive' in keys):
                CPC(ODS(), kw=kw, should_AE=True)
                kw['j_non_inductive'] = 2 * Jval
            elif ('j_non_inductive' in keys) and ('j_ohmic' in keys) and ('j_total' in keys):
                CPC(ODS(), kw=kw, should_AE=True)
                kw['j_total'] = 2 * Jval
            CPC(ODS(), kw=kw)

        # Try setting four
        for dkey, rkey, factor in [
            ('j_total', 'j_non_inductive', 2),
            ('j_ohmic', 'j_non_inductive', 2),
            ('j_non_inductive', 'j_total', 3),
            ('j_bootstrap', 'j_total', 2),
            ('j_actuator', 'j_total', 2),
        ]:
            kw = copy.deepcopy(jdef)
            for key in kw.keys():
                if key != dkey:
                    kw[key] = Jval
            CPC(ODS(), kw=kw, should_AE=True)
            kw[rkey] = factor * Jval
            CPC(ODS(), kw=kw)

        # Try all 5
        kw = copy.deepcopy(jdef)
        for key in Js:
            kw[key] = Jval
        CPC(ODS(), kw=kw, should_AE=True)
        kw['j_non_inductive'] = 2 * Jval
        CPC(ODS(), kw=kw, should_AE=True)
        kw['j_total'] = 3 * Jval
        CPC(ODS(), kw=kw, warn=True)  # just to cover the warn sections

        # Now test with equilibrium and existing quantities
        ods = ODS().sample_equilibrium()
        kw = {'j_actuator': Jval, 'j_bootstrap': Jval}
        CPC(ods, kw=kw)  # j_ni = 2
        kw = {'j_bootstrap': 2 * Jval}
        CPC(ods, kw=kw, should_AE=True)
        kw = {'j_bootstrap': 2 * Jval, 'j_non_inductive': None}
        CPC(ods, kw=kw)  # j_ni = 3
        kw = {'j_actuator': 1.5 * Jval, 'j_bootstrap': 1.5 * Jval}
        CPC(ods, kw=kw)
        kw = {'j_bootstrap': 2 * Jval, 'j_actuator': None}
        CPC(ods, kw=kw)

        kw = {'j_ohmic': Jval}
        CPC(ods, kw=kw)  # j_total is 4
        kw = {'j_ohmic': 2 * Jval}
        CPC(ods, kw=kw, should_AE=True)
        kw = {'j_ohmic': 2 * Jval, 'j_total': None}
        CPC(ods, kw=kw)  # j_total is 5
        kw = {'j_ohmic': Jval, 'j_non_inductive': None}
        CPC(ods, kw=kw, should_AE=True)
        kw = {'j_ohmic': Jval, 'j_non_inductive': None, 'j_actuator': None}
        CPC(ods, kw=kw)  # j_ni is 4
        return

    def test_sumary_global_quantities(self):
        ods = ODS().sample()
        ods.physics_summary_global_quantities()
        assert ods['summary']['global_quantities']['tau_energy']['value'] is not None
        assert ods['summary.global_quantities.beta_tor.value'] is not None
        return

    def test_line_average_density(self):
        ods = ODS().sample()
        ods.physics_summary_lineaverage_density()
        assert ods['summary.line_average.n_e.value'] is not None
        return

    def test_current_from_eq(self):
        ods = ODS().sample_equilibrium()
        ods.physics_current_from_eq(0)
        return

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
        return

    def test_cocos_transform(self):
        assert cocos_transform(None, None)['TOR'] == 1
        assert cocos_transform(1, 3)['POL'] == -1
        for cocos_ind in range(1, 9):
            assert cocos_transform(cocos_ind, cocos_ind + 10)['invPSI'] != 1
            for cocos_add in range(2):
                for thing in ['BT', 'TOR', 'POL', 'Q']:
                    assert cocos_transform(cocos_ind + cocos_add * 10, cocos_ind + cocos_add * 10)[thing] == 1
        return

    def test_identify_cocos(self):
        tests = {}
        # Create test cases for COCOS 1, 3, 5, 7
        odds = {
            1: {
                "B0": 2.5,
                "Ip": 1e6,
                "psi": numpy.linspace(0, 2, 3),
                "q": numpy.linspace(0.5, 1.5, 3),
                "clockwise_phi": False,
                "a": numpy.linspace(0, 2, 3),
            },
            3: {
                "B0": 2.5,
                "Ip": 1e6,
                "psi": numpy.linspace(2, 0, 3),
                "q": numpy.linspace(-0.5, -1.5, 3),
                "clockwise_phi": False,
                "a": numpy.linspace(0, 2, 3),
            },
            5: {
                "B0": 2.5,
                "Ip": 1e6,
                "psi": numpy.linspace(0, 2, 3),
                "q": numpy.linspace(-0.5, -1.5, 3),
                "clockwise_phi": False,
                "a": numpy.linspace(0, 2, 3),
            },
            7: {
                "B0": 2.5,
                "Ip": 1e6,
                "psi": numpy.linspace(2, 0, 3),
                "q": numpy.linspace(0.5, 1.5, 3),
                "clockwise_phi": False,
                "a": numpy.linspace(0, 2, 3),
            },
        }
        tests.update(odds)
        # Set clockwise_phi to True in these to get COCOS 2, 4, 6, 8
        evens = {}
        for cocos, kwargs in odds.items():
            even_kwargs = kwargs.copy()
            even_kwargs["clockwise_phi"] = True
            evens[cocos + 1] = even_kwargs
        tests.update(evens)
        # Multiply by factor of 2*pi to get COCOS 11 -> 18
        tens = {}
        for cocos, kwargs in tests.items():
            tens_kwargs = kwargs.copy()
            # Note: can't use *= here, as some references are shared between tests
            tens_kwargs["psi"] = kwargs["psi"] * (2 * numpy.pi)
            tens_kwargs["q"] = kwargs["q"] * (2 * numpy.pi)
            tens[cocos + 10] = tens_kwargs
        tests.update(tens)
        # TODO include tests for negative/antiparallel B0 and Ip
        # Run each test
        for idx, (expected, kwargs) in enumerate(tests.items()):
            actual = identify_cocos(**kwargs)[0]
            err_msg = f"Expected COCOS {expected}, but found {actual} with: \n{kwargs}"
            with self.subTest(actual=actual, expected=expected, msg=err_msg):
                self.assertEqual(actual, expected)

    def test_coordsio(self):
        data5 = numpy.linspace(0, 1, 5)
        data10 = numpy.linspace(0, 1, 10)

        # data can be entered without any coordinate checks
        ods0 = ODS()
        ods0['equilibrium.time_slice[0].profiles_1d.psi'] = data10
        ods0['equilibrium.time_slice[0].profiles_1d.f'] = data5
        assert len(ods0['equilibrium.time_slice[0].profiles_1d.psi']) != len(ods0['equilibrium.time_slice[0].profiles_1d.f'])

        # if a coordinate exists, then that is the coordinate that it is used
        ods1 = ODS()
        ods1['equilibrium.time_slice[0].profiles_1d.psi'] = data10
        with omas_environment(ods1, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': data5}):
            ods1['equilibrium.time_slice[0].profiles_1d.f'] = data5
        assert len(ods1['equilibrium.time_slice[0].profiles_1d.f']) == 10

        # if a coordinate does not exists, then it is added
        ods2 = ODS()
        with omas_environment(ods2, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': data5}):
            ods2['equilibrium.time_slice[0].profiles_1d.pressure'] = data5
        assert len(ods2['equilibrium.time_slice[0].profiles_1d.pressure']) == 5

        # coordinates can be easily copied over from existing ODSs with .list_coordinates() method
        ods3 = ODS()
        ods3.update(ods1.list_coordinates())
        with omas_environment(ods3, coordsio=ods1):
            ods3['equilibrium.time_slice[0].profiles_1d.f'] = ods1['equilibrium.time_slice[0].profiles_1d.f']
        with omas_environment(ods3, coordsio=ods2):
            ods3['equilibrium.time_slice[0].profiles_1d.pressure'] = ods2['equilibrium.time_slice[0].profiles_1d.pressure']
        assert len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10
        assert len(ods3['equilibrium.time_slice[0].profiles_1d.pressure']) == 10

        # ods can be queried on different coordinates than they were originally filled in (ods example)
        with omas_environment(ods3, coordsio=ods2):
            assert len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 5
        assert len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10

        # ods can be queried on different coordinates than they were originally filled in (ods example)
        with omas_environment(ods3, coordsio=ods3):
            assert len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10

        # ods can be queried on different coordinates than they were originally filled in (dictionary example)
        with omas_environment(ods3, coordsio={'equilibrium.time_slice[0].profiles_1d.psi': data5}):
            assert len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 5
        assert len(ods3['equilibrium.time_slice[0].profiles_1d.f']) == 10

        # this case is different because the coordinate and the data do not share the same parent
        ods5 = ODS()
        ods5['core_profiles.profiles_1d[0].grid.rho_tor_norm'] = data5
        with omas_environment(ods5, coordsio={'core_profiles.profiles_1d[0].grid.rho_tor_norm': data10}):
            ods5['core_profiles.profiles_1d[0].electrons.density_thermal'] = data10
        assert len(ods5['core_profiles.profiles_1d[0].grid.rho_tor_norm']) == 5
        assert len(ods5['core_profiles.profiles_1d[0].electrons.density_thermal']) == 5

        ods6 = ODS()
        ods6['core_profiles.profiles_1d[0].grid.rho_tor_norm'] = data5

        return

    def test_cocosio(self):
        x = numpy.linspace(0.1, 1, 10)

        ods = ODS(cocosio=11, cocos=11)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x)

        ods = ODS(cocosio=11, cocos=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x)

        ods = ODS(cocosio=2, cocos=11)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x)

        ods = ODS(cocosio=2, cocos=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x)

        # reassign the same value
        ods = ODS(cocosio=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = ods['equilibrium.time_slice.0.profiles_1d.psi']
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x)

        # use omas_environment
        ods = ODS(cocosio=2)
        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        with omas_environment(ods, cocosio=11):
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x * (2 * numpy.pi))

        ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x)

        return

    def test_coordsio_cocosio(self):
        x = numpy.linspace(0.1, 1, 11)
        y = numpy.linspace(-1, 1, 11)

        xh = numpy.linspace(0.1, 1, 21)
        yh = numpy.linspace(-1, 1, 21)

        ods = ODS()
        with omas_environment(ods, cocosio=2):
            ods['equilibrium.time_slice.0.profiles_1d.psi'] = x
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], x)
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x * 2 * numpy.pi)

        with omas_environment(ods, cocosio=2, coordsio={'equilibrium.time_slice.0.profiles_1d.psi': xh}):
            ods['equilibrium.time_slice.0.profiles_1d.phi'] = yh
            assert len(ods['equilibrium.time_slice.0.profiles_1d.phi']) == len(yh)
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], xh)
        assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], -x * 2 * numpy.pi)
        assert len(ods['equilibrium.time_slice.0.profiles_1d.phi']) == len(y)

        ods = ODS()

        p = 1 - numpy.linspace(0, 1, 10)
        psi2 = numpy.linspace(0, 1, 10) ** 2
        psi11 = -psi2 * (2 * numpy.pi)

        psi2_ = numpy.linspace(0, 1, 10)
        psi11_ = -psi2_ * (2 * numpy.pi)

        with omas_environment(ods, cocosio=2):
            ods['equilibrium.time_slice.0.profiles_1d.psi'] = psi2
            assert all(ods['equilibrium.time_slice.0.profiles_1d.psi'] == psi2)
        assert all(ods['equilibrium.time_slice.0.profiles_1d.psi'] == psi11)

        with omas_environment(ods, cocosio=2):
            assert all(ods['equilibrium.time_slice.0.profiles_1d.psi'] == psi2)
        assert all(ods['equilibrium.time_slice.0.profiles_1d.psi'] == psi11)

        with omas_environment(ods, cocosio=2, coordsio={'equilibrium.time_slice.0.profiles_1d.psi': psi2}):
            assert all(ods['equilibrium.time_slice.0.profiles_1d.psi'] == psi2), '2'
        assert all(ods['equilibrium.time_slice.0.profiles_1d.psi'] == psi11), '11'

        with omas_environment(ods, cocosio=2, coordsio={'equilibrium.time_slice.0.profiles_1d.psi': psi2}):
            ods['equilibrium.time_slice.0.profiles_1d.pressure'] = p
            assert all(ods['equilibrium.time_slice.0.profiles_1d.pressure'] == p), 'p2'
        assert all(ods['equilibrium.time_slice.0.profiles_1d.pressure'] == p), 'p11'

        psi2_ = numpy.linspace(0, 1, 10)
        psi11_ = -psi2_ * (2 * numpy.pi)

        with omas_environment(ods, cocosio=2, coordsio={'equilibrium.time_slice.0.profiles_1d.psi': psi2_}):
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.psi'], psi2_)
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.pressure'], numpy.interp(psi2_, psi2, p))
            index = numpy.argsort(psi11_)
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.pressure'], numpy.interp(psi11_, psi11[index], p[index]))

        psi2__ = numpy.linspace(0, 1, 100)
        psi11__ = -psi2__ * (2 * numpy.pi)

        with omas_environment(ods, cocosio=2, coordsio={'equilibrium.time_slice.0.profiles_1d.psi': psi2__}):
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.pressure'], numpy.interp(psi2__, psi2, p))
            index = numpy.argsort(psi11)
            assert numpy.allclose(ods['equilibrium.time_slice.0.profiles_1d.pressure'], numpy.interp(psi11__, psi11[index], p[index]))
        return

    @unittest.skipIf(failed_PINT, str(failed_PINT))
    def test_handle_units(self):
        import pint

        ureg = pint.UnitRegistry()

        ods = ODS()
        ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement'] = 8.0 * ureg.milliseconds
        assert ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement'] == 0.008

        with omas_environment(ods, unitsio=True):
            tmp = ods['equilibrium.time_slice[0].constraints.diamagnetic_flux.time_measurement']
            assert tmp.magnitude == 0.008
            assert tmp.units == 'second'
        return

    def test_search_ion(self):
        ods = ODS()
        ods.sample_core_profiles(include_pressure=False)

        tmp = search_ion(ods["core_profiles.profiles_1d.0.ion"], 'D')
        assert repr(tmp) == '{0: [0]}'

        tmp = search_ion(ods["core_profiles.profiles_1d.0.ion"], multiple_matches_raise_error=False)
        assert repr(tmp) == '{0: [0], 1: [0]}'

        try:
            tmp = search_ion(ods["core_profiles.profiles_1d.0.ion"])
            raise AssertError('multiple_matches_raise_error')
        except IndexError:
            pass

        tmp = search_ion(ods["core_profiles.profiles_1d.0.ion"], A=12)
        assert repr(tmp) == '{1: [0]}'

        tmp = search_ion(ods["core_profiles.profiles_1d.0.ion"], 'W', no_matches_raise_error=False)
        assert repr(tmp) == '{}'

        try:
            tmp = search_ion(ods["core_profiles.profiles_1d.0.ion"], 'W')
            raise AssertError('no_matches_raise_error failed')
        except IndexError:
            pass
        return

    def test_search_in_array_structure(self):
        ods = ODS()

        tmp = search_in_array_structure(ods['core_transport.model'], {'identifier.name': 'omas_tgyro'}, no_matches_return=0)[0]
        assert tmp == 0

        ods['core_transport.model.+.identifier.name'] = 'omas_tgyro'
        ods['core_transport.model.+.identifier.name'] = 'test1'
        ods['core_transport.model.+.identifier.name'] = 'omas_tgyro'
        ods['core_transport.model.-1.identifier.description'] = 'bla bla'
        ods['core_transport.model.+.identifier.name'] = 'test2'

        try:
            search_in_array_structure(ods['core_transport.model'], {'identifier.name': 'omas_tgyro'})
            raise AssertionError('multiple_matches_raise_error failed')
        except IndexError:
            pass

        tmp = search_in_array_structure(ods['core_transport.model'], {'identifier.name': 'omas_tgyro'}, multiple_matches_raise_error=False)
        assert tmp[0] == 0 and tmp[1] == 2

        tmp = search_in_array_structure(ods['core_transport.model'], {'identifier.name': 'omas_tgyro', 'identifier.description': 'bla bla'})
        assert tmp[0] == 2
        return

    def test_latest_cocos(self):
        from omas.omas_utils import list_structures, omas_rcparams
        from omas.omas_physics import generate_cocos_signals

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*defining its COCOS transform.*")
            generate_cocos_signals(
                list_structures(imas_version=omas_rcparams['default_imas_version']), threshold=0, write=False, verbose=False
            )
        return

    # End of TestOmasPhysics class


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasPhysics)
    unittest.TextTestRunner(verbosity=2).run(suite)
