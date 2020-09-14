'''
Utility to generate the omas/omas_cocos.py file
'''

import os, sys

sys.path.insert(0, os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])

from omas.omas_utils import list_structures, omas_rcparams
from omas.omas_physics import generate_cocos_signals

generate_cocos_signals(list_structures(imas_version=omas_rcparams['default_imas_version']), threshold=0, write=True)
