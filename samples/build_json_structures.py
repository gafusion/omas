from __future__ import print_function, division, unicode_literals
import os, re, glob
from omas import *

for imas_version in map(lambda x: os.path.split(x)[-1], glob.glob(imas_json_dir + os.sep + '*')):
    aggregate_imas_html_docs(imas_version=imas_version)

    create_json_structure(imas_version=imas_version)

    create_html_documentation(imas_version=imas_version)
