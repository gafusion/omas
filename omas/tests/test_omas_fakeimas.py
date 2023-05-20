#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/imas/__init__.py

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_machine

-------
"""

# OMAS imports
from omas import ODS, fakeimas
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
        eq = fakeimas.equilibrium()
        eq.time_slice.resize(10)
        eq.time_slice[5].global_quantities.ip = 1.0
        print(eq.time_slice[5].global_quantities.ip)
        pprint(eq.pretty_paths())

        DB = fakeimas.DBEntry('MDSPLUS_BACKEND', 'd3d', 133221, 0, os.environ['USER'], '3')
        DB.create()
        DB.put(eq)
        DB.close()

        DB = fakeimas.DBEntry('MDSPLUS_BACKEND', 'd3d', 133221, 0, os.environ['USER'], '3')
        DB.open()
        eq1 = DB.get('equilibrium')
        DB.close()

        pprint(eq1.pretty_paths())
        print(type(eq.time_slice[5].global_quantities))
        print(eq.time_slice[5].global_quantities.ip)

    def test_fake_module(self):
        if 'imas' in sys.modules:
            del sys.modules['imas']

        # ============================================
        # Load fake IMAS data with OMAS IMAS interface
        # ============================================
        with fakeimas.fake_environment():
            pf = fakeimas.pf_active()
            pf.coil.resize(1)
            pf.coil[0].current.data = [1, 2, 3]
            pf.ids_properties.homogeneous_time = 1
            DB = fakeimas.DBEntry('MDSPLUS_BACKEND', 'd3d', 123456, 0, os.environ['USER'], '3')
            DB.create()
            DB.put(pf)
            DB.close()

        # use fakeimas.fake_environment
        ods = ODS()
        with fakeimas.fake_environment():
            ods.load('imas', os.environ['USER'], 'd3d', 123456, 0, backend='MDSPLUS')

        # make sure that outside of fakeimas.fake_environment it would fail
        try:
            ods.load('imas', os.environ['USER'], 'd3d', 123456, 0, backend='MDSPLUS')
            raise RuntimeError('Should not be here')
        except ModuleNotFoundError:
            pass

        # use nested fakeimas.fake_environments
        ods = ODS()
        with fakeimas.fake_environment():
            ods.load('imas', os.environ['USER'], 'd3d', 123456, 0, backend='MDSPLUS')
            with fakeimas.fake_environment():
                ods.load('imas', os.environ['USER'], 'd3d', 123456, 0, backend='MDSPLUS')

        # make sure that outside of fakeimas.fake_environment it would fail
        try:
            ods.load('imas', os.environ['USER'], 'd3d', 123456, 0, backend='MDSPLUS')
            raise RuntimeError('Should not be here')
        except ModuleNotFoundError:
            pass

        # use fakeimas.fake_module switch
        fakeimas.fake_module(True)
        ods.load('imas', os.environ['USER'], 'd3d', 123456, 0, backend='MDSPLUS')

        fakeimas.fake_module(False)
        # make sure that after fakeimas.fake_module(False) it would fail
        try:
            ods.load('imas', os.environ['USER'], 'd3d', 123456, 0, backend='MDSPLUS')
            raise RuntimeError('Should not be here')
        except ModuleNotFoundError:
            pass
