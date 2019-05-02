from __future__ import print_function, division, unicode_literals

import os, re, glob
from pprint import pprint

os.environ['OMAS_DEBUG_TOPIC'] = '*'

from omas import *

force_build_json = 'last'

# get the tags of the data-dictionary repository
imas_versions.clear()
imas_versions.update(generate_xml_schemas())
pprint(imas_versions.keys())

# loops over the available IDSDef.xml files and generates .json and omas_doc.html files
for imas_version in imas_versions:
    print('Processing IMAS data structures `%s`' % imas_version)

    filename = os.path.abspath(os.sep.join([imas_json_dir, imas_versions[imas_version], 'omas_doc.html']))

    if not os.path.exists(filename) or force_build_json is True or (force_build_json == 'last' and (imas_version == list(imas_versions.keys())[-1]) or imas_version == 'develop/3'):
        generate_xml_schemas(imas_version=imas_version)
        create_json_structure(imas_version=imas_version)
        create_html_documentation(imas_version=imas_version)
