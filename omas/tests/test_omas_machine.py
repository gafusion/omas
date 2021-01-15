#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for omas/omas_machine.py

.. code-block:: none

   python3 -m unittest omas/tests/test_omas_machine

-------
"""

# OMAS imports
from omas import *
from omas.omas_utils import *
from omas.tests import warning_setup
from omas.tests.failed_imports import *
from omas.omas_machine import *

class TestOmasMachine(UnittestCaseOmas):
    """
    Test suite for omas_machine.py
    """

    machine = 'd3d'
    pulse = 168830

    @unittest.skipIf(failed_OMFIT, str(failed_OMFIT))
    def test_load_omas_machine(self):
        ods = load_omas_machine(self.machine, self.pulse)

    def test_machines(self):
        assert self.machine in machines()

    @unittest.skipIf(failed_OMFIT, str(failed_OMFIT))
    def test_user_mappings(self):
        user_machine_mappings = {'equilibrium.time_slice.:.global_quantities.beta_normal':
                       {'TDI': '\\{EFIT_tree}::TOP.RESULTS.AEQDSK.BETAN',
                        'treename': '{EFIT_tree}'}}
        location = 'equilibrium.time_slice.:.global_quantities.beta_normal'

        user_machine_mappings={
            "__options__": {'EFIT_tree': 'EFIT01', 'default_tree': 'D3D', 'machine':'d3d'},
            "dataset_description.data_entry.machine": {
                "VALUE": "{machine}"
            }
        }
        location='dataset_description.data_entry.machine'

        ods, _ = machine_to_omas(ODS(), self.machine, self.pulse, location, options={}, branch=None, user_machine_mappings=user_machine_mappings)
        print(ods[location])