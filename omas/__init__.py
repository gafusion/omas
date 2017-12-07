from __future__ import print_function, division, unicode_literals
from .omas_core import *
import os as _os

__version__ = open(_os.path.abspath(str(_os.path.dirname(__file__)) + _os.sep + 'version'), 'r').read().strip()
