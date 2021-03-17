#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_core.py

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_core

-------
"""

import os
import numpy
from pprint import pprint
import xarray

# OMAS imports
from omas import *
from omas.omas_utils import *
from omas.tests import warning_setup
from omas.tests.failed_imports import *


class TestOmasCore(UnittestCaseOmas):
    """
    Test suite for omas_core.py
    """

    def test_misc(self):
        ods = ODS()
        # check effect of disabling dynamic path creation
        try:
            with omas_environment(ods, dynamic_path_creation=False):
                ods['dataset_description.data_entry.user']
        except LookupError:
            ods['dataset_description'] = ODS()
            ods['dataset_description.data_entry.user'] = os.environ.get('USER', 'dummy_user')
        else:
            raise Exception('OMAS error handling dynamic_path_creation=False')

        # check that accessing leaf that has not been set raises a ValueError, even with dynamic path creation turned on
        try:
            ods['dataset_description.data_entry.machine']
        except ValueError:
            pass
        else:
            raise Exception('querying leaf that has not been set should raise a ValueError')

        # info ODS is used for keeping track of IMAS metadata
        ods['dataset_description.data_entry.machine'] = 'ITER'
        ods['dataset_description.imas_version'] = omas_rcparams['default_imas_version']
        ods['dataset_description.data_entry.pulse'] = 1
        ods['dataset_description.data_entry.run'] = 0

        # check .get() method
        assert ods.get('dataset_description.data_entry.pulse') == ods['dataset_description.data_entry.pulse']
        assert ods.get('dataset_description.bad', None) is None

        # check that keys is an iterable
        keys = ods.keys()
        keys[0]

        # check that dynamic path creation during __getitem__ does not leave empty fields behind
        try:
            print(ods['wall.description_2d.0.limiter.unit.0.outline.r'])
        except ValueError:
            assert 'wall.description_2d.0.limiter.unit.0.outline' not in ods

        ods['equilibrium']['time_slice'][0]['time'] = 1000.0
        ods['equilibrium']['time_slice'][0]['global_quantities']['ip'] = 1.5

        ods2 = copy.deepcopy(ods)
        ods2['equilibrium']['time_slice'][1] = ods['equilibrium']['time_slice'][0]
        ods2['equilibrium.time_slice.1.time'] = 2000.0

        ods2['equilibrium']['time_slice'][2] = copy.deepcopy(ods['equilibrium']['time_slice'][0])
        ods2['equilibrium.time_slice[2].time'] = 3000.0

        assert (
            ods2['equilibrium']['time_slice'][0]['global_quantities'].ulocation
            == ods2['equilibrium']['time_slice'][2]['global_quantities'].ulocation
        )

        ods2['equilibrium.time_slice.1.global_quantities.ip'] = 2.0

        # check different ways of addressing data
        for item in [
            ods2['equilibrium.time_slice']['1.global_quantities'],
            ods2[['equilibrium', 'time_slice', 1, 'global_quantities']],
            ods2[('equilibrium', 'time_slice', 1, 'global_quantities')],
            ods2['equilibrium.time_slice.1.global_quantities'],
            ods2['equilibrium.time_slice[1].global_quantities'],
        ]:
            assert item.ulocation == 'equilibrium.time_slice.:.global_quantities'

        ods2['equilibrium.time_slice.0.profiles_1d.psi'] = numpy.linspace(0, 1, 10)

        # check data slicing
        assert numpy.all(ods2['equilibrium.time_slice[:].global_quantities.ip'] == numpy.array([1.5, 2.0, 1.5]))

        # uncertain scalar
        ods2['equilibrium.time_slice[2].global_quantities.ip'] = ufloat(3, 0.1)

        # uncertain array
        ods2['equilibrium.time_slice[2].profiles_1d.q'] = uarray([0.0, 1.0, 2.0, 3.0], [0, 0.1, 0.2, 0.3])

        ckbkp = ods.consistency_check
        tmp = pickle.dumps(ods2)
        ods2 = pickle.loads(tmp)
        if ods2.consistency_check != ckbkp:
            raise Exception('consistency_check attribute changed')

        # check flattening
        tmp = ods2.flat()

        # check deepcopy
        ods3 = ods2.copy()

    def test_xarray(self):
        ods = ODS().sample_equilibrium()
        abs = ods.xarray('equilibrium.time_slice.0.profiles_1d.q')
        rel = ods['equilibrium'].xarray('time_slice.0.profiles_1d.q')
        for k in [
            'cocos_label_transformation',
            'cocos_leaf_name_aos_indices',
            'cocos_transformation_expression',
            'coordinates',
            'data_type',
            'documentation',
            'full_path',
            'lifecycle_status',
            'type',
            'units',
        ]:
            assert k in abs['q'].attrs
            assert k in rel['q'].attrs
            assert abs['q'].attrs[k] == rel['q'].attrs[k]
        for k in ['y', 'y_rel', 'y_full', 'x', 'x_rel', 'x_full']:
            assert k in abs.attrs
            assert k in rel.attrs
            if '_rel' in k:
                assert abs.attrs[k] != rel.attrs[k]
            else:
                assert abs.attrs[k] == rel.attrs[k]

        # check setting of an xarray.DataArray
        with omas_environment(ods, dynamic_path_creation='dynamic_array_structures'):
            ods['equilibrium.time_slice[2].profiles_1d.q'] = xarray.DataArray(
                uarray([0.0, 1.0, 2.0, 3.0], [0, 0.1, 0.2, 0.3]), coords={'x': [1, 2, 3, 4]}, dims=['x']
            )

    def test_dynamic_location(self):
        ods = ODS()
        for k in range(5):
            ods[f'equilibrium.time_slice.{k}.global_quantities.ip'] = k
        tmp = ods[f'equilibrium.time_slice.-1']
        assert tmp.location == f'equilibrium.time_slice.{k}'
        del ods[f'equilibrium.time_slice.0']
        assert tmp.location == f'equilibrium.time_slice.{k - 1}'

    def test_auto_deepcopy_on_assignment(self):
        ods = ODS()
        ods[f'equilibrium.time_slice.0.global_quantities.ip'] = 0.0
        ods[f'equilibrium.time_slice.1'] = ods[f'equilibrium.time_slice.0']

        # test auto copy.deepcopy on assignment
        assert id(ods[f'equilibrium.time_slice.0']) != id(ods[f'equilibrium.time_slice.1'])

        # test no extra copy if the user does it for us
        tmp = copy.deepcopy(ods[f'equilibrium.time_slice.0'])
        ods[f'equilibrium.time_slice.2'] = tmp
        assert id(ods[f'equilibrium.time_slice.2']) == id(tmp)

        # test extra copy if multiple assignments
        ods[f'equilibrium.time_slice.3'] = tmp
        assert id(ods[f'equilibrium.time_slice.3']) != id(tmp)

    def test_data_slicing(self):
        ods = ODS()
        ods['langmuir_probes.embedded.0.name'] = '1'
        ods['langmuir_probes.embedded.1.name'] = '12'
        ods['langmuir_probes.embedded.2.name'] = '123'
        assert ods['langmuir_probes']['embedded.:.name'][2] == '123'

        ods = ODS()
        for k in range(3):
            ods[f'langmuir_probes.embedded.{k}.time'] = [float(k)] * (k + 1)
        assert numpy.allclose(
            ods['langmuir_probes']['embedded.:.time'],
            numpy.array([[0.0, numpy.nan, numpy.nan], [1.0, 1.0, numpy.nan], [2.0, 2.0, 2.0]]),
            equal_nan=True,
        )

    def test_uncertain_slicing(self):
        """Tests whether : slicing works properly with uncertain data"""
        from uncertainties import ufloat

        ods = ODS()
        ods['pulse_schedule']['position_control']['x_point'][0]['z']['reference']['data'] = [ufloat(1.019, 0.02), ufloat(1.019, 0.02)]
        result = ods['pulse_schedule.position_control.x_point.:.z.reference.data']
        # Trips a ValueError if the dtype of the uncertain array isn't handled properly.

    def test_dynamic_set_nonzero_array_index(self):
        ods = ODS()
        ods.consistency_check = False
        self.assertRaises(IndexError, ods.__setitem__, 'something[10]', 5)

    def test_coordinates(self):
        ods = ODS().sample_equilibrium()

        assert 'equilibrium.time_slice.0.profiles_1d.psi' in ods.coordinates('equilibrium.time_slice.0.profiles_1d.q')
        assert 'time_slice.0.profiles_1d.psi' in ods['equilibrium'].coordinates('time_slice.0.profiles_1d.q')

        assert 'equilibrium.time_slice.0.profiles_1d.psi' in ods.list_coordinates()
        assert 'equilibrium.time_slice.0.profiles_1d.psi' in ods['equilibrium'].list_coordinates()

        assert 'equilibrium.time_slice.0.profiles_1d.psi' in ods.list_coordinates(absolute_location=False)
        assert 'time_slice.0.profiles_1d.psi' in ods['equilibrium'].list_coordinates(absolute_location=False)

        assert 'equilibrium.time_slice.0.profiles_1d.psi' in ods.coordinates()
        assert 'time_slice.0.profiles_1d.psi' in ods['equilibrium'].coordinates()

    def test_dataset(self):
        ods = ODS()

        ods.sample_equilibrium(time_index=0)
        ods.sample_equilibrium(time_index=1)

        ods.sample_core_profiles(time_index=0)
        ods.sample_core_profiles(time_index=1)

        n = 1e10
        sizes = {}
        for homogeneous in [False, 'time', None, 'full']:
            DS = ods.dataset(homogeneous=homogeneous)
            print(homogeneous, len(DS.variables))
            sizes[homogeneous] = len(DS.variables)
            if homogeneous is not None:
                assert n >= sizes[homogeneous], 'homogeneity setting does not match structure reduction expectation'
                n = sizes[homogeneous]
        assert sizes[None] == sizes['time'], 'sample equilibrium and core_profiles should be homogeneous'

        ods.sample_pf_active()
        DS = ods.dataset(homogeneous='full')

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
        assert numpy.allclose(ods.time('equilibrium'), [100, 200, 300])
        assert ods['equilibrium'].homogeneous_time() is True

        # time arrays can be set using `set_time_array` function
        # this simplifies the logic in the code since one does not
        # have to check if the array was already there or not
        ods.set_time_array('equilibrium.time', 0, 101)
        ods.set_time_array('equilibrium.time', 1, 201)
        ods.set_time_array('equilibrium.time', 2, 302)

        # the make the timeslices consistent
        ods['equilibrium.time_slice'][0]['time'] = 101
        ods['equilibrium.time_slice'][1]['time'] = 201
        ods['equilibrium.time_slice'][2]['time'] = 302

        # get time information from explicitly set time array
        extra_info = {}
        assert numpy.allclose(ods.time('equilibrium'), [101, 201, 302])
        assert numpy.allclose(ods.time('equilibrium.time_slice'), [101, 201, 302])
        assert numpy.allclose(ods['equilibrium'].time('time_slice'), [101, 201, 302])
        assert ods['equilibrium'].homogeneous_time() is True

        # get time value from a single item in array of structures
        extra_info = {}
        assert ods['equilibrium.time_slice'][0].time() == 101
        assert ods['equilibrium'].time('time_slice.0') == 101
        assert ods['equilibrium.time_slice'][0].homogeneous_time() is True

        # sample pf_active data has non-homogeneous times
        ods.sample_pf_active()
        assert ods['pf_active'].homogeneous_time() is False, 'sample pf_active data should have non-homogeneous time'
        assert ods['pf_active.coil'][0]['current'].homogeneous_time() is True

        # sample ic_antennas data has non-homogeneous times
        ods.sample_ic_antennas()
        assert ods['ic_antennas'].homogeneous_time() is False, 'sample ic_antennas data should have non-homogeneous time'

        ods.sample_dataset_description()
        ods['dataset_description'].satisfy_imas_requirements()
        assert ods['dataset_description.ids_properties.homogeneous_time'] is not None

    def test_dynamic_set_existing_list_nonzero_array_index(self):
        ods = ODS()
        ods.consistency_check = False
        ods['something[0]'] = 5
        with omas_environment(ods, dynamic_path_creation='dynamic_array_structures'):
            ods['something[7]'] = 10
        assert ods['something[0]'] == 5
        assert ods['something[7]'] == 10

    def test_set_nonexisting_array_index(self):
        ods = ODS()
        ods.consistency_check = False
        with omas_environment(ods, dynamic_path_creation=False):
            self.assertRaises(IndexError, ods.__setitem__, 'something.[10]', 5)

    def test_force_type(self):
        ods = ODS()
        ods['core_profiles.profiles_1d'][0]['ion'][0]['z_ion'] = 1
        assert isinstance(ods['core_profiles.profiles_1d'][0]['ion'][0]['z_ion'], float)

    def test_address_structures(self):
        ods = ODS()

        # make sure data structure is of the right type
        assert isinstance(ods['core_transport'].omas_data, dict)
        assert isinstance(ods['core_transport.model'].omas_data, list)

        # append elements by using `+`
        for k in range(10):
            ods['equilibrium.time_slice.+.global_quantities.ip'] = k
        assert len(ods['equilibrium.time_slice']) == 10
        assert ods['equilibrium.time_slice'][9]['global_quantities.ip'] == 9

        # access element by using negative indices
        assert ods['equilibrium.time_slice'][-1]['global_quantities.ip'] == 9
        assert ods['equilibrium.time_slice.-10.global_quantities.ip'] == 0

        # set element by using negative indices
        ods['equilibrium.time_slice.-1.global_quantities.ip'] = -99
        ods['equilibrium.time_slice'][-10]['global_quantities.ip'] = -100
        assert ods['equilibrium.time_slice'][-1]['global_quantities.ip'] == -99
        assert ods['equilibrium.time_slice'][-10]['global_quantities.ip'] == -100

        # access by pattern
        assert ods['@eq.*1.*.ip'] == 1

    def test_version(self):
        ods = ODS(imas_version='3.20.0')
        ods['ec_antennas.antenna.0.power.data'] = [1.0]

        try:
            ods = ODS(imas_version='3.21.0')
            ods['ec_antennas.antenna.0.power.data'] = [1.0]
            raise AssertionError('3.21.0 should not have `ec_antennas.antenna.0.power`')
        except LookupError:
            pass

        # check support for development version is there
        ODS(imas_version='develop.3')

        try:
            tmp = ODS(imas_version='does_not_exist')
        except ValueError:
            pass

    def test_satisfy_imas_requirements(self):
        ods = ODS()
        ods['equilibrium.time_slice.0.global_quantities.ip'] = 0.0
        # check if data structures satisfy IMAS requirements (this should Fail)
        try:
            ods.satisfy_imas_requirements()
            raise ValueError('It is expected that not all the sample structures have the .time array set')
        except ValueError as _excp:
            pass

        ods = ODS().sample()

        # re-check if data structures satisfy IMAS requirements (this should pass)
        ods.satisfy_imas_requirements()

    def test_deepcopy(self):
        ods = ODS().sample()

        # inject non-consistent data
        ods.consistency_check = False
        ods['bla'] = ODS(consistency_check=False)
        ods['bla.tra'] = 1
        try:
            # this should fail
            ods.consistency_check = True
        except LookupError:
            assert not ods.consistency_check

        # deepcopy should not raise a consistency_check error
        # since we are directly manipulating the __dict__ attributes
        import copy

        ods1 = copy.deepcopy(ods)

        # make sure non-consistent data got also copied over
        assert ods1['bla'] == ods['bla']

        # make sure the deepcopy is not shallow
        ods1['equilibrium.vacuum_toroidal_field.r0'] += 1
        assert ods['equilibrium.vacuum_toroidal_field.r0'] + 1 == ods1['equilibrium.vacuum_toroidal_field.r0']

        # deepcopy using .copy() method
        ods2 = ods.copy()

        # make sure the deepcopy is not shallow
        ods2['equilibrium.vacuum_toroidal_field.r0'] += 1
        assert ods['equilibrium.vacuum_toroidal_field.r0'] + 1 == ods2['equilibrium.vacuum_toroidal_field.r0']

    def test_saveload(self):
        ods = ODS()
        ods.sample_equilibrium()
        ods.save('test.pkl')

    def test_input_data_process_functions(self):
        def robust_eval(string):
            import ast

            try:
                return ast.literal_eval(string)
            except:
                return string

        ods = ODS(consistency_check=False)
        with omas_environment(ods, input_data_process_functions=[robust_eval]):
            ods['int'] = '1'
            ods['float'] = '1.0'
            ods['str'] = 'bla'
            ods['complex'] = '2+1j'
        for item in ods:
            assert isinstance(ods[item], eval(item))

    def test_conversion_after_assignement(self):
        ods = ODS(consistency_check=False)
        ods['bla'] = 5
        try:
            ods[0] = 4
            raise AssertionError('Convertion of dict to list should not be allowed')
        except TypeError:
            pass

        del ods['bla']
        ods[0] = 4
        try:
            ods['bla'] = 5
            raise AssertionError('Convertion of list to dict should not be allowed')
        except TypeError:
            pass

    def test_codeparameters(self):
        ods = ODS()
        ods['equilibrium.code.parameters'] = CodeParameters(imas_json_dir + '/../samples/input_gray.xml')

        tmp = {}
        tmp.update(ods['equilibrium.code.parameters'])
        ods['equilibrium.code.parameters'] = tmp
        assert isinstance(ods['equilibrium.code.parameters'], CodeParameters)

        with omas_environment(ods, xmlcodeparams=True):
            assert isinstance(ods['equilibrium.code.parameters'], str)
        assert isinstance(ods['equilibrium.code.parameters'], CodeParameters)

        # test that dynamic creation of .code.parameters makes it a CodeParameters object
        ods = ODS()
        ods['equilibrium.code.parameters']['param1'] = 1
        assert isinstance(ods['equilibrium.code.parameters'], CodeParameters)

        # test saving of code_parameters in json format
        ods.save(tempfile.gettempdir() + '/ods_w_codeparams.json')

        # test that loading of data with code.parameters results in a CodeParameters object
        ods = ODS()
        ods.load(tempfile.gettempdir() + '/ods_w_codeparams.json')
        assert isinstance(ods['equilibrium.code.parameters'], CodeParameters)

        # test loading CodeParameters from a json
        ods = ODS().load(imas_json_dir + '/../samples/ods_w_code_parameters.json')
        # test traversing ODS and code parameters with OMAS syntax
        assert ods['ec_launchers.code.parameters.launcher.0.mharm'] == 2
        # test that sub-tree elements of code parameters are also of CodeParameters class
        assert isinstance(ods['ec_launchers.code.parameters.launcher'], CodeParameters)
        # test to_string and from_string methods
        with omas_environment(ods, xmlcodeparams=True):
            code_parameters_string = ods['ec_launchers.code.parameters']
            tmp = CodeParameters().from_string(code_parameters_string)
            assert isinstance(tmp['launcher'], CodeParameters)
            assert tmp['launcher.0.mharm'] == 2
        # test that CodeParameters are restored after xmlcodeparams=True environment
        assert isinstance(ods['ec_launchers.code.parameters'], CodeParameters)
        assert isinstance(ods['ec_launchers.code.parameters.launcher'], CodeParameters)
        assert isinstance(ods['ec_launchers.code.parameters.launcher.0'], CodeParameters)
        assert len(ods['ec_launchers.code.parameters.launcher.0']) == 2

    def test_latexit(self):
        assert latexit['somewhere.:.sublocation.n_e'] == '$n_e$'
        assert latexit['.n_e'] == '$n_e$'
        assert latexit['n_e'] == '$n_e$'

        assert latexit['core_profiles.profiles_1d[:].electrons.density_thermal'] == '$n_e$'
        assert latexit['barometry.gauge[:].pressure.data'] == '$P$'
        assert latexit['equilibrium.time_slice[0].ggd[1].b_field_tor[0].values'] == r'$B_{\phi}$'
        assert latexit['core_profiles.profiles_1d[4].ion[0].density'] == '$n_{i0}$'

        try:
            latexit['somewhere.does_not_exist']
            raise RuntimeError('latexit of missing variable should fail')
        except KeyError:
            pass

        assert latexit.get('somewhere.does_not_exist', 'somewhere.does_not_exist') == 'somewhere.does_not_exist'

    def test_odx(self):
        ods = ODS().sample_equilibrium()
        odx = ods.to_odx()
        ods1 = odx.to_ods()
        assert not ods.diff(ods)

    def test_odc(self):
        odc = ODC()
        for k in range(5):
            odc[f'133221.equilibrium.time_slice.{k}.global_quantities.ip'] = 1000.0 + k + 1
            odc[f'133229.equilibrium.time_slice.{k}.global_quantities.ip'] = 2000.0 + k + 1
            odc[f'133230.equilibrium.time_slice.{k}.global_quantities.ip'] = 3000.0 + k + 1
        assert odc.keys() == [133221, 133229, 133230]
        assert odc[':.equilibrium.time_slice.:.global_quantities.ip'].size == 15

        for ftype in ['h5', 'pkl', 'nc', 'json']:
            odc.save('test.' + ftype)
            odc1 = ODC().load('test.' + ftype)
            diff = odc1.diff(odc)
            assert not diff, f'save/load ODC to {ftype} failed:\r{repr(diff)}'

    def test_diff_attrs(self):
        ods = ODS(imas_version='3.30.0').sample_equilibrium()
        ods1 = ODS(imas_version='3.30.0').sample_equilibrium()
        assert not ods.diff_attrs(ods1)
        ods1 = ODS(imas_version='3.29.0').sample_equilibrium()
        ods1.consistency_check = False
        assert ods.diff_attrs(ods1, verbose=True)

    def test_top(self):
        ods = ODS()
        ods['equilibrium.time_slice.0.global_quantities.ip'] = 1.0
        assert ods['equilibrium.time_slice.0.global_quantities'].top is ods
        assert ods['equilibrium.time_slice.0.global_quantities'].top is not ods['equilibrium']

    def test_imas_version(self):
        ods = ODS()
        assert ods.imas_version == omas_rcparams['default_imas_version']
        assert ods['equilibrium.time_slice.0.global_quantities'].imas_version == omas_rcparams['default_imas_version']

        ods['equilibrium.time_slice.0.global_quantities'].imas_version = 'test'
        assert ods.imas_version == 'test'
        assert ods['equilibrium.time_slice.0.global_quantities'].imas_version == 'test'

    # End of TestOmasCore class


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasCore)
    unittest.TextTestRunner(verbosity=2).run(suite)
