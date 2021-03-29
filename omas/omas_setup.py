'''pypi setup file

-------
'''

# --------------------------------------------
# external imports
# --------------------------------------------
import os
import sys

with open(os.path.abspath(str(os.path.dirname(__file__)) + os.sep + 'version'), 'r') as _f:
    __version__ = _f.read().strip()

# Add minor version revisions here
# This is done to keep track of changes between OMAS PYPI releases
# the if statements for these minor revisions can be deleted
# as the OMAS PYPI version increases
if __version__ == '0.66.0':
    __version__ += '.1'

if sys.version_info < (3, 5):
    raise Exception(
        '''
OMAS v%s only runs with Python 3.6+ and you are running Python %s
'''
        % (__version__, '.'.join(map(str, sys.version_info[:2])))
    )

import pwd
import glob
import json
import copy
from collections import OrderedDict
import re
import numpy
from pprint import pprint
from io import StringIO
from contextlib import contextmanager
import tempfile
import warnings
from functools import wraps
import ast
import base64
import traceback
import difflib
import weakref
import unittest
import itertools

try:
    import tqdm
except ImportError:
    tqdm = None

formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: formatwarning_orig(
    message, category, filename, lineno, line=''
)

# pint: avoid loading pint upfront since it can be slow and it is not always used
ureg = []
if False:
    import pint

    ureg.append(pint.UnitRegistry())
else:
    ureg.append(None)

# uncertainties
import uncertainties
import uncertainties.unumpy as unumpy
from uncertainties.unumpy import nominal_values, std_devs, uarray
from uncertainties import ufloat

# xarrays: avoid loading xarrays upfront since it can be slow and it is not always used
# import xarray

from collections.abc import MutableMapping
import pickle


def b2s(bytes):
    return bytes.decode("utf-8")


# --------------------------------------------
# configuration of directories and IMAS infos
# --------------------------------------------
class IMAS_json_dir(str):
    """
    directory where the JSON data structures for the different versions of IMAS are stored
    """

    pass


imas_json_dir = IMAS_json_dir(os.path.abspath(str(os.path.dirname(__file__)) + '/imas_structures/'))

omas_git_repo = False
if os.path.exists(imas_json_dir + '/../../.git') and os.access(imas_json_dir + '/../../.git', os.W_OK):
    omas_git_repo = True


class IMAS_versions(OrderedDict):
    """
    Dictionary with list of IMAS version and their sub-folder name in the imas_json_dir
    """

    def __init__(self, mode='all'):
        '''
        :param mode: `all`, `named`, `tagged`
        '''
        OrderedDict.__init__(self)
        if mode in ['all', 'named']:
            # first `develop/3` and other branches
            for _item in list(map(lambda x: os.path.basename(x), sorted(glob.glob(imas_json_dir + os.sep + '*')))):
                if not _item.startswith('3'):
                    self[_item.replace('_', '.')] = _item
        if mode in ['all', 'tagged']:
            # next all tagged versions sorted by version number
            for _item in list(map(lambda x: os.path.basename(x), sorted(glob.glob(imas_json_dir + os.sep + '*')))):
                if _item.startswith('3'):
                    self[_item.replace('_', '.')] = _item
        # do not include empty imas_structures directories (eg. needed to avoid issues wheen switching to old git branches)
        for item, value in list(self.items()):
            if not len(glob.glob(imas_json_dir + os.sep + value + os.sep + '*.json')):
                del self[item]


# imas versions
imas_versions = IMAS_versions()
if len(list(imas_versions.keys())):
    latest_imas_version = list(imas_versions.keys())[-1]
else:
    latest_imas_version = '0.0.0'
if 'OMAS_IMAS_VERSION' in os.environ:
    _default_imas_version = os.environ['OMAS_IMAS_VERSION']
else:
    if len(list(imas_versions.keys())):
        _default_imas_version = list(imas_versions.keys())[-1]
    else:
        _default_imas_version = '0.0.0'

# --------------------------------------------
# rcparams
# --------------------------------------------
class OMAS_rc_params(dict):
    """
    dictionary of parameters that control how OMAS operates
    """

    pass


omas_rcparams = OMAS_rc_params()
omas_rcparams.update(
    {
        'cocos': 11,
        'consistency_check': True,
        'dynamic_path_creation': True,
        'tmp_omas_dir': os.environ.get(
            'OMAS_TMP_DIR', os.sep.join([tempfile.gettempdir(), os.environ.get('USER', 'dummy_user'), 'OMAS_TMP_DIR'])
        ),
        'fake_imas_dir': os.environ.get(
            'OMAS_FAKE_IMAS_DIR', os.sep.join([os.environ.get('HOME', tempfile.gettempdir()), 'tmp', 'OMAS_FAKE_IMAS_DIR'])
        ),
        'allow_fake_imas_fallback': bool(int(os.environ.get('OMAS_ALLOW_FAKE_IMAS_FALLBACK', '0'))),
        'default_imas_version': _default_imas_version,
        'default_mongo_server': 'mongodb+srv://{user}:{pass}@omasdb-xymmt.mongodb.net',
        'pickle_protocol': 4,
    }
)


@contextmanager
def rcparams_environment(**kw):
    old_omas_rcparams = omas_rcparams.copy()
    omas_rcparams.update(kw)
    try:
        yield omas_rcparams
    finally:
        omas_rcparams.update(old_omas_rcparams)


# --------------------------------------------
# additional data structures
# --------------------------------------------
add_datastructures = {}


def omas_testdir(filename_topic=''):
    """
    Return path to temporary folder where OMAS TEST file are saved/loaded

    NOTE: If directory does not exists it is created

    :return: string with path to OMAS TEST folder
    """
    if filename_topic:
        filename_topic = os.path.splitext(os.path.split(filename_topic)[-1])[0] + '/'
    tmp = tempfile.gettempdir() + '/OMAS_TESTS/' + filename_topic
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    return tmp
