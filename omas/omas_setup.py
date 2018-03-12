from __future__ import print_function, division, unicode_literals

# --------------------------------------------
# external imports
# --------------------------------------------
import os
import sys
import glob
import json
import copy
from collections import OrderedDict
import re
import numpy
from pprint import pprint
import tempfile

# Python3/2 import differences
if sys.version_info < (3, 0):
    import cPickle as pickle
    from collections import MutableMapping
else:
    basestring = str
    unicode = str
    import pickle
    from collections.abc import MutableMapping

# --------------------------------------------
# rcparams
# --------------------------------------------
omas_rcparams = {
    'consistency_check': bool(int(os.environ.get('OMAS_CONSISTENCY_CHECK', '1'))),
    'dynamic_path_creation': bool(int(os.environ.get('OMAS_DYNAMIC_PATH_CREATION', '1'))),
    'tmp_imas_dir': os.environ.get('OMAS_TMP_DIR',
                                    os.sep.join(
                                        [tempfile.gettempdir(), 'OMAS_TMP_DIR'])),
    'fake_imas_dir': os.environ.get('OMAS_FAKE_IMAS_DIR',
                                    os.sep.join(
                                        [os.environ.get('HOME', tempfile.gettempdir()), 'tmp', 'OMAS_FAKE_IMAS_DIR'])),
    'allow_fake_imas_fallback': bool(int(os.environ.get('OMAS_ALLOW_FAKE_IMAS_FALLBACK', '0')))
}

# --------------------------------------------
# configuration of directories and IMAS infos
# --------------------------------------------
imas_json_dir = os.path.abspath(str(os.path.dirname(__file__)) + '/imas_structures/')

separator = '.'

if 'IMAS_VERSION' in os.environ:
    default_imas_version = os.environ['IMAS_VERSION']
else:
    default_imas_version = re.sub('_', '.', os.path.split(sorted(glob.glob(imas_json_dir + os.sep + '*'))[-1])[-1])
