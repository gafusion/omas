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

import omas


class TestOmasW7XMachine(omas.omas_utils.UnittestCaseOmas):
    """
    Test suite for omas_physics.py
    """
    machine = "w7x"

    def setUp(self):
        import omas.machine_mappings.w7x

    def test_load(self):
        self.assertIn(self.machine, omas.omas_machine.machines())

    # def test_coils(self):
    #     ods = omas.ODS()
    #     ods, info = omas.omas_machine.machine_to_omas(ods, "w7x", -1, "pf_active")
