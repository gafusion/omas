#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/examples/...

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_examples

-------
"""

# Basic imports
import os

# Use Agg backend to avoid opening up figures
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot

# OMAS imports
from omas import *
from omas.omas_utils import *
from omas.tests.failed_imports import *


class TestOmasExamples(UnittestCaseOmas):
    """
    Test suite for examples files
    """

    def tearDown(self):
        pyplot.close()

    def test_omas_units(self):
        from omas.examples import omas_units

    def test_omas_time(self):
        from omas.examples import omas_time

    def test_ods_sample(self):
        from omas.examples import ods_sample

    def test_omas_coordinates(self):
        from omas.examples import omas_coordinates

    def test_plot_quantity(self):
        from omas.examples import plot_quantity

    def test_parse_codeparameters(self):
        from omas.examples import parse_codeparameters

    @unittest.skipIf(failed_IMAS, str(failed_IMAS))
    @unittest.skipIf(not_running_on_iter_cluster, str(not_running_on_iter_cluster))
    def test_solps_imas(self):
        from omas.examples import solps_imas

    @unittest.skipIf((failed_IMAS or failed_OMFIT), str(failed_IMAS) + str(failed_OMFIT))
    def test_geqdsk_to_from_imas(self):
        from omas.examples import geqdsk_to_from_imas

    def test_showcase_paths(self):
        from omas.examples import showcase_paths

    def test_ods_process_input_data(self):
        from omas.examples import ods_process_input_data

    @unittest.skipIf(failed_MONGO, str(failed_MONGO))
    def test_omas_mongo_example(self):
        from omas.examples import omas_mongo_example

    @unittest.skipIf(failed_S3, str(failed_S3))
    def test_save_load_through(self):
        from omas.examples import save_load_through

    def test_connect_gkdb(self):
        from omas.examples import connect_gkdb

    @unittest.skipIf(failed_UDA, str(failed_UDA))
    def test_omas_uda_example(self):
        from omas.examples import omas_uda_example

    def test_save_load_all(self):
        from omas.examples import save_load_all

    @unittest.skipIf(failed_S3, str(failed_S3))
    def test_plot_omas(self):
        from omas.examples import plot_omas

    def test_omas_resample(self):
        from omas.examples import omas_resample

    def test_uncertain(self):
        from omas.examples import uncertain

    @unittest.skipIf(failed_OMFIT, str(failed_OMFIT))
    def test_plot_g_s_2_ip(self):
        from omas.examples import plot_g_s_2_ip

    def test_plot_saveload_scaling(self):
        from omas.examples import plot_saveload_scaling

    def test_across_ODSs(self):
        from omas.examples import across_ODSs

    def test_omas_cocos(self):
        from omas.examples import omas_cocos

    @unittest.skipIf(failed_IMAS, str(failed_IMAS))
    @unittest.skipIf(not_running_on_iter_cluster, str(not_running_on_iter_cluster))
    def test_iter_scenario(self):
        from omas.examples import iter_scenario

    @unittest.skipIf(failed_IMAS, str(failed_IMAS))
    def test_simple_imas(self):
        from omas.examples import simple_imas

    def test_consistency_check(self):
        from omas.examples import consistency_check

    @unittest.skipIf(failed_OMFIT, str(failed_OMFIT))
    def test_plot_omas_omfit(self):
        from omas.examples import plot_omas_omfit

    @unittest.skipIf(failed_S3, str(failed_S3))
    def test_plot_omas_overlays(self):
        from omas.examples import plot_omas_overlays

    def test_omas_info(self):
        from omas.examples import omas_info

    def test_omas_dynamic_nc(self):
        from omas.examples import omas_dynamic_nc

    @unittest.skipIf(failed_IMAS, str(failed_IMAS))
    def test_omas_dynamic_imas(self):
        from omas.examples import omas_dynamic_imas

    @unittest.skipIf(failed_D3D_MDS, str(failed_D3D_MDS))
    def test_omas_dynamic_machine(self):
        from omas.examples import omas_dynamic_machine

    def test_omas_collection(self):
        from omas.examples import omas_collection

    def test_extra_structures(self):
        from omas.examples import extra_structures

    @unittest.skipIf(failed_IMAS, str(failed_IMAS))
    @unittest.skipIf(not_running_on_cea_cluster, str(not_running_on_cea_cluster))
    def test_west_geqdsk(self):
        from omas.examples import west_geqdsk


# for filename in glob.glob(os.path.abspath(imas_json_dir+'/../examples/*.py')):
#     if '__init__' in filename:
#         continue
#     name = os.path.splitext(os.path.split(filename)[1])[0]
#     execstring = '''    def test_{name}(self):
#         from omas.examples import {name}
# '''.format(name=name)
#     print(execstring)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmasExamples)
    unittest.TextTestRunner(verbosity=2).run(suite)
