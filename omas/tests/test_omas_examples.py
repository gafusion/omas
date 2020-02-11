#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/examples/...

.. code-block:: none

   python -m unittest omas/tests/test_omas_examples

-------
"""

# Basic imports
from __future__ import print_function, division, unicode_literals
import os
import unittest
import numpy
import warnings
import copy

# OMAS imports
from omas import *
from omas.omas_utils import *
from failed_imports import *

# Use Agg backend to avoid opening up figures
import matplotlib
matplotlib.use('Agg')

class TestOmasExamples(unittest.TestCase):
    """
    Test suite for omas_utils.py
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

    @unittest.skipUnless(not failed_IMAS, str(failed_IMAS))
    def test_solps_imas(self):
        from omas.examples import solps_imas

    @unittest.skipUnless(not (failed_IMAS or failed_OMFIT), str(failed_IMAS) + str(failed_OMFIT))
    def test_geqdsk_to_from_imas(self):
        from omas.examples import geqdsk_to_from_imas

    def test_showcase_paths(self):
        from omas.examples import showcase_paths

    def test_ods_process_input_data(self):
        from omas.examples import ods_process_input_data

    @unittest.skipUnless(not failed_MONGO, str(failed_MONGO))
    def test_omas_mongo_example(self):
        from omas.examples import omas_mongo_example

    def test_save_load_through(self):
        from omas.examples import save_load_through

    def test_connect_gkdb(self):
        from omas.examples import connect_gkdb

    @unittest.skipUnless(not failed_UDA, str(failed_UDA))
    def test_omas_uda_example(self):
        from omas.examples import omas_uda_example

    def test_save_load_all(self):
        from omas.examples import save_load_all

    @unittest.skipUnless(not failed_S3, str(failed_S3))
    def test_plot_omas(self):
        from omas.examples import plot_omas

    def test_omas_resample(self):
        from omas.examples import omas_resample

    def test_uncertain(self):
        from omas.examples import uncertain

    @unittest.skipUnless(not failed_OMFIT, str(failed_OMFIT))
    def test_plot_g_s_2_ip(self):
        from omas.examples import plot_g_s_2_ip

    def test_plot_saveload_scaling(self):
        from omas.examples import plot_saveload_scaling

    def test_across_ODSs(self):
        from omas.examples import across_ODSs

    def test_omas_cocos(self):
        from omas.examples import omas_cocos

    @unittest.skipUnless(not failed_IMAS, str(failed_IMAS))
    def test_iter_scenario(self):
        from omas.examples import iter_scenario

    @unittest.skipUnless(not failed_IMAS, str(failed_IMAS))
    def test_simple_imas(self):
        from omas.examples import simple_imas

    def test_consistency_check(self):
        from omas.examples import consistency_check

    @unittest.skipUnless(not failed_OMFIT, str(failed_OMFIT))
    def test_plot_omas_omfit(self):
        from omas.examples import plot_omas_omfit

    @unittest.skipUnless(not failed_S3, str(failed_S3))
    def test_plot_omas_overlays(self):
        from omas.examples import plot_omas_overlays

    def test_omas_info(self):
        from omas.examples import omas_info

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
