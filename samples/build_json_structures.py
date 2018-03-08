from __future__ import print_function, division, unicode_literals

import os, re, glob
os.environ['OMAS_DEBUG_TOPIC'] = '*'

from omas import *

generate_IDSDef_xml=True
force_build_json = False

# loops through the tags of the data-dictionary repository and generates the IDSDef.xml files for each one
if generate_IDSDef_xml:
    generate_xml_schemas()

# loops over the available IDSDef.xml files and generates .json and omas_doc.html files
for imas_version in sorted(map(lambda x: os.path.split(x)[-1], glob.glob(imas_json_dir + os.sep + '*'))):

    print('Processing IMAS data structures v%s' % re.sub('_', '.', imas_version))
    filename = os.path.abspath(os.sep.join([imas_json_dir, imas_version, 'omas_doc.html']))

    if force_build_json or not os.path.exists(filename):
        create_json_structure(imas_version=imas_version)
        create_html_documentation(imas_version=imas_version)
