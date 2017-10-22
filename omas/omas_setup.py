from __future__ import print_function, division, unicode_literals

#--------------------------------------------
# external imports
#--------------------------------------------
import os
import sys
import glob
import json
import copy
import pandas
from collections import OrderedDict
import re
import numpy
from pprint import pprint
import weakref
import cPickle as pickle

#--------------------------------------------
# rcparams
#--------------------------------------------
omas_rcparams={
    'consistency_check':True,
}

#--------------------------------------------
# configuration of directories and IMAS infos
#--------------------------------------------
imas_json_dir=os.path.abspath(str(os.path.dirname(unicode(__file__, sys.getfilesystemencoding())))+'/imas_structures/')

separator='.'

if 'IMAS_VERSION' in os.environ:
    default_imas_version=os.environ['IMAS_VERSION']
else:
    default_imas_version='3.10.1'
