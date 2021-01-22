import os, sys, re, glob
from pprint import pprint

os.environ['OMAS_DEBUG_TOPIC'] = '*'

omas_dir = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
sys.path.insert(0, omas_dir)
from omas import *
from omas.omas_structure import *

force_build_json = 'last'

# get the tags of the data-dictionary repository
imas_versions.clear()
imas_versions.update(generate_xml_schemas())
pprint(imas_versions.keys())

# loops over the available IDSDef.xml files and generates .json and omas_doc.html files
for imas_version in imas_versions:
    print('Processing IMAS data structures `%s`' % imas_version)

    filename = os.path.abspath(os.sep.join([imas_json_dir, imas_versions[imas_version], 'omas_doc.html']))

    if (
        not os.path.exists(filename)
        or force_build_json is True
        or (force_build_json == 'last' and (imas_version == list(imas_versions.keys())[-1]) or imas_version == 'develop/3')
    ):
        generate_xml_schemas(imas_version=imas_version)
        create_json_structure(imas_version=imas_version)
        create_html_documentation(imas_version=imas_version)

# generate symlinks between imas versions
symlink_imas_structure_versions(test=False)

# update IMAS badge in README.md and index.rst file with latest verison
for filename in [omas_dir + '/README.md', omas_dir + '/sphinx/source/index.rst']:
    with open(filename, 'r') as f:
        txt = f.read()
    txt = re.sub('IMAS-([0-9\.]+)-yellow', 'IMAS-' + list(imas_versions.keys())[-1] + '-yellow', txt)
    with open(filename, 'w') as f:
        f.write(txt)
