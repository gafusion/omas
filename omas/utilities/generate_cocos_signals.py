'''
Utility to generate the omas/omas_cocos.py file
'''

from omas.omas_utils import list_structures, default_imas_version
from omas.omas_physics import generate_cocos_signals

generate_cocos_signals(list_structures(imas_version=default_imas_version))
