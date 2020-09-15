#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input data processing
=====================
OMAS supports user defined input data processing
"""

from omas import *


# user defined function that takes input string and converts it to python types
def robust_eval(string):
    import ast

    try:
        return ast.literal_eval(string)
    except:
        return string


# assign data and process it with user defined function
ods = ODS(consistency_check=False)
with omas_environment(ods, input_data_process_functions=[robust_eval]):
    ods['int'] = '1'
    ods['float'] = '1.0'
    ods['str'] = 'bla'
    ods['complex'] = '2+1j'

# test
for item in ods:
    print(ods[item], type(ods[item]))
    assert isinstance(ods[item], eval(item))
