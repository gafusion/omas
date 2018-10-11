'''pypi setup file

-------
'''

from __future__ import print_function, division, unicode_literals

# --------------------------------------------
# external imports
# --------------------------------------------
import os
import sys
import glob
import json
import copy
from collections import MutableMapping, OrderedDict
import re
import numpy
from pprint import pprint
from io import StringIO
from contextlib import contextmanager
import tempfile
import warnings

formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    formatwarning_orig(message, category, filename, lineno, line='')

# pint
try:
    import pint
    from pint import UnitRegistry
    ureg = UnitRegistry()
except ImportError:
    pint = None
    ureg = None
    warnings.warn('pint Python library not found')

# uncertainties
import uncertainties
import uncertainties.unumpy as unumpy
from uncertainties.unumpy import nominal_values, std_devs, uarray
from uncertainties import ufloat

# xarrays
import xarray

# Python3/2 import differences
if sys.version_info < (3, 0):
    import cPickle as pickle
else:
    basestring = str
    unicode = str
    import pickle

    _orig_pickle_loads=pickle.loads
    def _pickle_loads_python2compatible(*args,**kw):
        kw.setdefault('encoding','latin1')
        return _orig_pickle_loads(*args,**kw)
    pickle.loads=_pickle_loads_python2compatible

    _orig_pickle_load=pickle.load
    def _pickle_load_python2compatible(*args,**kw):
        kw.setdefault('encoding','latin1')
        return _orig_pickle_load(*args,**kw)
    pickle.load=_pickle_load_python2compatible

# --------------------------------------------
# configuration of directories and IMAS infos
# --------------------------------------------
class IMAS_json_dir(unicode):
    '''
    directory where the JSON data structures for the different versions of IMAS are stored
    '''
    pass
imas_json_dir = IMAS_json_dir(os.path.abspath(str(os.path.dirname(__file__)) + '/imas_structures/'))

class IMAS_versions(OrderedDict):
    '''
    dictionary with list of IMAS version and their sub-folder name in the imas_json_dir
    '''
    pass
imas_versions = IMAS_versions()
# first `develop/3` and other branches
for _item in list(map(lambda x: os.path.basename(x), sorted(glob.glob(imas_json_dir + os.sep + '*')))):
    if not _item.startswith('3'):
        imas_versions[_item.replace('_', '.')] = _item
# next all tagged versions sorted by version number
for _item in list(map(lambda x: os.path.basename(x), sorted(glob.glob(imas_json_dir + os.sep + '*')))):
    if _item.startswith('3'):
        imas_versions[_item.replace('_', '.')] = _item

if 'OMAS_IMAS_VERSION' in os.environ:
    _default_imas_version = os.environ['OMAS_IMAS_VERSION']
else:
    try:
        _default_imas_version = list(imas_versions.keys())[-1]
    except IndexError:
        # IndexError will occur if `imas_json_dir` is empty: we must allow going forward, at least to build_json_structures
        _default_imas_version = ''

# --------------------------------------------
# configuration of directories and ITM infos
# --------------------------------------------
if 'OMAS_DATAVERSION_TAG' in os.environ:
    _default_itm_version = os.environ['OMAS_DATAVERSION_TAG']
else:
    _default_itm_version = '4.10b_rc'

# --------------------------------------------
# rcparams
# --------------------------------------------
class OMAS_rc_params(dict):
    '''
    dictionary of parameters that control how OMAS operates
    '''
    pass

omas_rcparams = OMAS_rc_params()
omas_rcparams.update({
    'cocos': 11,
    'cocosio': 11,
    'coordsio': {},
    'consistency_check': True,
    'dynamic_path_creation': True,
    'unitsio': False,
    'tmp_imas_dir': os.environ.get('OMAS_TMP_DIR',
                                   os.sep.join(
                                       [tempfile.gettempdir(), 'OMAS_TMP_DIR'])),
    'fake_imas_dir': os.environ.get('OMAS_FAKE_IMAS_DIR',
                                    os.sep.join(
                                        [os.environ.get('HOME', tempfile.gettempdir()), 'tmp', 'OMAS_FAKE_IMAS_DIR'])),
    'allow_fake_imas_fallback': bool(int(os.environ.get('OMAS_ALLOW_FAKE_IMAS_FALLBACK', '0'))),
    'fake_itm_dir': os.environ.get('OMAS_FAKE_ITM_DIR',
                                   os.sep.join(
                                       [os.environ.get('HOME', tempfile.gettempdir()), 'tmp', 'OMAS_FAKE_ITM_DIR'])),
    'allow_fake_itm_fallback': bool(int(os.environ.get('OMAS_ALLOW_FAKE_ITM_FALLBACK', '0'))),
    'default_imas_version': _default_imas_version,
    'default_itm_version': _default_itm_version
})

@contextmanager
def rcparams_environment(**kw):
    old_omas_rcparams=omas_rcparams.copy()
    omas_rcparams.update(kw)
    try:
        yield omas_rcparams
    finally:
        omas_rcparams.update(old_omas_rcparams)
