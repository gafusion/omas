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
from omas.omas_machine import machine_to_omas


class TestOmasMachine(UnittestCaseOmas):
    """
    Test suite for omas_machine.py
    """

    machine = 'd3d'
    pulse = 168830

    def test_machines(self):
        # list local machines
        assert self.machine in machines()

        # access machine description that should fail
        for branch in ['', 'master', 'dummy']:
            try:
                machines('machine_that_does_not_exist', branch)
                raise ValueError('error in machines()')
            except NotImplementedError:
                pass

        # with branch=None return file in current repo
        assert os.path.abspath(imas_json_dir + '/..') in machines(self.machine, '')

        # with branch='master' return file in temp dir
        assert omas_rcparams['tmp_omas_dir'] in machines(self.machine, 'master')

    def test_remote_machine_mappings(self):
        # access machine description remotely
        location = 'dataset_description.data_entry.machine'

        # load local machine mapping
        machine_mappings(self.machine, '')
        # show that the data for the location is there
        ods, info = machine_to_omas(ODS(), self.machine, self.pulse, location)
        assert ods[location] == self.machine

        # now let's remove that mapping from the local machine mapping cache
        # so that we can force fallback on the remote master branch
        from omas.omas_machine import _machine_mappings

        try:
            tmp = copy.deepcopy(_machine_mappings[self.machine, ''])
            del _machine_mappings[self.machine, ''][location]

            # now let's access the same node again. The data should come from the `master` branch
            ods, info = machine_to_omas(ODS(), self.machine, self.pulse, location)
            assert ods[location] == self.machine
            assert info['branch'] == 'master'
        finally:
            _machine_mappings[self.machine, ''] = tmp

    def test_user_mappings(self):
        location = 'dataset_description.data_entry.machine'
        for user_machine_mappings in [{}, {"dataset_description.data_entry.machine": {"EVAL": "{machine!r}+'123'"}}]:
            ods, info = machine_to_omas(ODS(), self.machine, self.pulse, location, user_machine_mappings=user_machine_mappings)
            if not user_machine_mappings:
                assert ods[location] == self.machine
            else:
                assert ods[location] == self.machine + '123'

    def test_value(self):
        location = 'dataset_description.data_entry.pulse'
        ods, info = machine_to_omas(ODS(), self.machine, self.pulse, location)
        print(ods[location])

    def test_python(self):
        location = 'interferometer.channel.:.identifier'
        ods, info = machine_to_omas(ODS(), self.machine, self.pulse, location)

    def test_tdi(self):
        # make sure all machines have a MDS+ server assigned
        for machine in machines():
            machine_mappings(self.machine, '')['__mdsserver__']
