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

    def test_machines_list(self):
        assert self.machine in machines()

    def test_machines(self):
        # access machine description that should fail
        for branch in [None, 'master', 'dummy']:
            try:
                machines('machine_that_does_not_exist', None)
                raise ValueError('error in machines()')
            except NotImplementedError:
                pass

        # local machine returns file
        assert os.path.abspath(imas_json_dir + '/..') in machines(self.machine, None)[0]

        # access machine description that should fail
        assert omas_rcparams['tmp_omas_dir'] in machines(self.machine, 'machine')[0]  # this test will fail when we delete this branch

    def test_user_mappings(self):
        location = 'dataset_description.data_entry.machine'
        for user_machine_mappings in [{}, {"dataset_description.data_entry.machine": {"VALUE": "{machine}123"}}]:
            ods, _ = machine_to_omas(ODS(), self.machine, self.pulse, location, user_machine_mappings=user_machine_mappings)
            if not user_machine_mappings:
                assert ods[location] == self.machine
            else:
                assert ods[location] == self.machine + '123'

    def test_value(self):
        location = 'dataset_description.data_entry.pulse'
        ods, data = machine_to_omas(ODS(), self.machine, self.pulse, location)
        print(ods[location])

    def test_python(self):
        location = 'interferometer.channel.:.identifier'
        ods, data = machine_to_omas(ODS(), self.machine, self.pulse, location)

    def test_tdi(self):
        # make sure all machines have a MDS+ server assigned
        for machine in machines():
            mds_machine_to_server_mapping(machine, None)
