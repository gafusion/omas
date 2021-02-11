#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/imas/__init__.py

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_machine

-------
"""

# OMAS imports
from omas import ODS
from omas import fakeimas as imas
from omas.omas_utils import *
from omas.tests import warning_setup
from omas.tests.failed_imports import *


class TestOmasFakeImas(UnittestCaseOmas):
    """
    Test suite for omas/imas/__init__.py
    """

    def test_fake_API(self):
        # ============================================
        # Use fake IMAS API in OMAS
        # ============================================
        eq = imas.equilibrium()
        eq.time_slice.resize(10)
        eq.time_slice[5].global_quantities.ip = 1.0
        print(eq.time_slice[5].global_quantities.ip)
        pprint(eq.pretty_paths())

        DB = imas.DBEntry('MDSPLUS_BACKEND', 'd3d', 133221, 0, os.environ['USER'], '3')
        DB.create()
        DB.put(eq)
        DB.close()

        DB = imas.DBEntry('MDSPLUS_BACKEND', 'd3d', 133221, 0, os.environ['USER'], '3')
        DB.open()
        eq1 = DB.get('equilibrium')
        DB.close()

        pprint(eq1.pretty_paths())
        print(type(eq.time_slice[5].global_quantities))
        print(eq.time_slice[5].global_quantities.ip)

    def test_fake_module(self):
        # ============================================
        # Load fake IMAS data with OMAS IMAS interface
        # ============================================
        # use imas.fake_environment
        ods = ODS()
        with imas.fake_environment():
            ods.load('imas', os.environ['USER'], 'd3d', 133221, 0, False, '3', 'HDF5')

        # make sure that outside of imas.fake_environment it would fail
        try:
            ods.load('imas', os.environ['USER'], 'd3d', 133221, 0, False, '3', 'HDF5')
            raise RuntimeError('Should not be here')
        except ModuleNotFoundError:
            pass

        # use nested imas.fake_environments
        ods = ODS()
        with imas.fake_environment():
            ods.load('imas', os.environ['USER'], 'd3d', 133221, 0, False, '3', 'HDF5')
            with imas.fake_environment():
                ods.load('imas', os.environ['USER'], 'd3d', 133221, 0, False, '3', 'HDF5')

        # make sure that outside of imas.fake_environment it would fail
        try:
            ods.load('imas', os.environ['USER'], 'd3d', 133221, 0, False, '3', 'HDF5')
            raise RuntimeError('Should not be here')
        except ModuleNotFoundError:
            pass

        # use imas.fake switch
        imas.fake_module(True)

        ods.load('imas', os.environ['USER'], 'd3d', 133221, 0, False, '3', 'HDF5')

        imas.fake_module(False)
        try:
            ods.load('imas', os.environ['USER'], 'd3d', 133221, 0, False, '3', 'HDF5')
            raise RuntimeError('Should not be here')
        except ModuleNotFoundError:
            pass

