from __future__ import print_function, division, unicode_literals

from .omas_utils import *


# --------------------------------------------
# generation of the imas structure json files
# --------------------------------------------
def generate_xml_schemas():
    """
    Generate IMAS IDSDef.xml files by:
     1. clone the IMAS data-dictionary repository (access to git.iter.org required)
     2. download the Saxon XSLT and XQuery Processor
     3. generate IDSDef.xml in imas_structures folder
    """
    import subprocess

    # clone the IMAS data-dictionary repository
    dd_folder = os.sep.join([imas_json_dir, '..', 'data-dictionary'])
    if not os.path.exists(dd_folder):
        subprocess.Popen(
            'cd %s ; git clone ssh://git@git.iter.org/imas/data-dictionary.git' % os.sep.join([imas_json_dir, '..']),
            stdout=subprocess.PIPE, shell=True).communicate()[0]

    # download Saxon
    sax_folder = os.sep.join([imas_json_dir, '..', 'SaxonHE9-6-0-10J'])
    if not os.path.exists(sax_folder):
        subprocess.Popen("""
        cd %s
        curl https://netcologne.dl.sourceforge.net/project/saxon/Saxon-HE/9.6/SaxonHE9-6-0-10J.zip > SaxonHE9-6-0-10J.zip
        unzip -d SaxonHE9-6-0-10J SaxonHE9-6-0-10J.zip
        rm SaxonHE9-6-0-10J.zip""" % os.sep.join([imas_json_dir, '..']), shell=True).communicate()

    # find IMAS data-dictionary tags
    result = subprocess.Popen('cd %s;git tag' % dd_folder, stdout=subprocess.PIPE, shell=True).communicate()[0]
    tags = filter(lambda x: str(x).startswith('3.') and int(x.split('.')[1]) >= 10, result.split())

    # fetch data structure updates
    subprocess.Popen("""
    cd {dd_folder}
    git fetch
    """.format(dd_folder=dd_folder), shell=True).communicate()

    # loop over tags and generate IDSDef.xml files
    for tag in tags:
        _tag = re.sub('\.', '_', tag)
        if not os.path.exists(os.sep.join([imas_json_dir, _tag, 'IDSDef.xml'])):
            subprocess.Popen("""
                export CLASSPATH={imas_json_dir}/../SaxonHE9-6-0-10J/saxon9he.jar;
                cd {dd_folder}
                git checkout {tag}
                make clean
                make
                mkdir {imas_json_dir}/{_tag}/
                cp IDSDef.xml {imas_json_dir}/{_tag}/
                """.format(tag=tag, _tag=_tag, imas_json_dir=imas_json_dir,
                           dd_folder=dd_folder), shell=True).communicate()


# additional data structures (NOTE:info information carries shot/run/version/machine/user info through different save formats)
add_datastructures = {}
add_datastructures['info'] = {
    "info.shot": {
        "full_path": "info.shot",
        "coordinates": [],
        "data_type": "INT_0D",
        "description": "shot number"
    },
    "info.imas_version": {
        "full_path": "info.imas_version",
        "coordinates": [],
        "data_type": "STR_0D",
        "description": "imas version"
    },
    "info.machine": {
        "full_path": "info.machine",
        "coordinates": [],
        "data_type": "STR_0D",
        "description": "machine name"
    },
    "info.user": {
        "full_path": "info.user",
        "coordinates": [],
        "data_type": "STR_0D",
        "description": "user name"
    },
    "info.run": {
        "full_path": "info.run",
        "coordinates": [],
        "data_type": "INT_0D",
        "description": "run number"
    }
}


def create_json_structure(imas_version=default_imas_version):

    import xmltodict
    file = imas_json_dir + os.sep + re.sub('\.', '_', imas_version) + os.sep + 'IDSDef.xml'
    tmp = xmltodict.parse(open(file).read())

    for file in glob.glob(imas_json_dir + os.sep + re.sub('\.', '_', imas_version) + os.sep + '*.json'):
        print('Remove '+file)
        os.remove(file)

    def process_path(inv):
        inv = re.sub(r'/', '.', inv)
        inv = inv.split(' OR ')[0]
        inv = remove_parentheses(inv, '[:]')
        inv = re.sub('\[:\]$', '', inv)
        return inv

    def traverse(me, hout, path, fout):
        me = copy.copy(me)
        hout_propagate = hout
        path_propagate = copy.deepcopy(path)

        if '@structure_reference' in me and me['@structure_reference']=='self':
            return hout, fout

        if '@name' in me:
            name = me['@name']
            path_propagate.append(name)
            hout_propagate = {}
            hout[name] = hout_propagate
            if '@path_doc' in me:
                fname = path_propagate[0] + '/' + me['@path_doc']
                me['@full_path'] = path_propagate[0] + '/' + me['@path_doc']
                del me['@path_doc']
            else:
                fname = path_propagate[0]
                me['@full_path'] = path_propagate[0]
            if '@path' in me:
                del me['@path']
            for coord in [c for c in me if c.startswith('@coordinate')]:
                if '...' not in me[coord]:
                    me[coord] = path_propagate[0] + '/' + me[coord]
            fname = process_path(fname)
            fout[fname] = {}

        if isinstance(me, list):
            keys = range(len(me))
        else:
            keys = me.keys()

        is_leaf = True
        for kid in keys:
            if isinstance(me[kid], (dict, list)):
                is_leaf = False
                traverse(me[kid], hout_propagate, path_propagate, fout)
            elif kid not in ['@name', '@xmlns:fn']:
                hout_propagate[kid] = me[kid]
                fout[fname][kid] = me[kid]

                # if is_leaf:
                # print(path_propagate)

        return hout, fout

    hout, fout = traverse(tmp, {}, [], {})

    # format conversions
    for item in sorted(fout):
        coords = []
        for key in list(fout[item].keys()):
            if key != '@coordinates' and key.startswith('@coordinate'):
                coords.append(process_path(fout[item][key]))
                del fout[item][key]
            elif key.startswith('@path'):
                fout[item][key] = process_path(fout[item][key])
        if len(coords):
            fout[item]['@coordinates'] = coords

    # check dimensions
    for item in sorted(fout):
        coord = []
        if '@coordinates' in fout[item]:
            for coord in fout[item]['@coordinates']:
                if '...' in coord:
                    continue
                if coord not in fout:
                    printe('%s --> %s' % (item, coord))

    # cleanup entries
    for item in sorted(fout):
        keys = fout[item].keys()
        values = fout[item].values()
        keys = map(lambda x: re.sub('^@', '', x), keys)
        fout[item] = dict(zip(keys, values))

    # break into pieces
    hout = {}
    for item in sorted(fout):
        #    if re.findall('(_error_upper|_error_lower|_error_index)$', item):
        #        continue
        ds = item.split(separator)[0]
        hout.setdefault(ds, {})[item] = fout[item]

    # additional data structures
    hout.update(add_datastructures)

    # prepare directory structure
    if not os.path.exists(imas_json_dir + os.sep + re.sub('\.', '_', imas_version)):
        os.makedirs(imas_json_dir + os.sep + re.sub('\.', '_', imas_version))
    for item in glob.glob(imas_json_dir + os.sep + re.sub('\.', '_', imas_version) + os.sep + '*.json'):
        os.remove(item)

    # deploy imas structures as json
    for structure in sorted(hout):
        if structure=='time':
            continue
        print(imas_json_dir + os.sep + re.sub('\.', '_', imas_version) + os.sep + structure + '.json')
        dump_string = json.dumps(hout[structure], default=json_dumper, indent=1, separators=(',', ': '))
        #dump_string = pickle.dumps(hout[structure],protocol=pickle.HIGHEST_PROTOCOL)
        open(imas_json_dir + os.sep + re.sub('\.', '_', imas_version) + os.sep + structure + '.json', 'w').write(dump_string)


def create_html_documentation(imas_version=default_imas_version):
    filename = os.path.abspath(os.sep.join([imas_json_dir, re.sub('\.', '_', imas_version), 'omas_doc.html']))

    table_header = "<table border=1, width='100%'>"
    sub_table_header = '<tr>' \
                       '<th style="width:25%">Path</th>' \
                       '<th style="width:25%">Dimensions</th>' \
                       '<th>Type</th>' \
                       '<th>Units</th>' \
                       '<th>Description</th>' \
                       '</tr>'

    column_style='style="word-wrap:break-word;word-break:break-all"'
    lines = []
    for structure_file in list_structures(imas_version=imas_version):
        print('Adding to html documentation: ' + structure_file)
        structure = load_structure(structure_file, imas_version=imas_version)[0]
        lines.append('<!-- %s -->' % structure_file)
        lines.append(table_header)
        lines.append(sub_table_header)
        for item in sorted(structure):
            if not any([item.endswith(k) for k in ['_error_index', '_error_lower', '_error_upper']]):
                try:
                    lines.append(
                        '<tr>'
                        '<td {column_style}><p>{item}</p></td>' \
                        '<td {column_style}><p>{coordinates}</p></td>' \
                        '<td><p>{data_type}</p></td>' \
                        '<td><p>{units}</p></td>' \
                        '<td><p>{description}</p></td>'
                        '</tr>'.format(
                            item=item,
                            coordinates=re.sub('\[\]', '', re.sub('[\'\"]', '', re.sub(',', ',<br>', str(
                                map(str, structure[item].get('coordinates', '')))))),
                            data_type=structure[item].get('data_type', ''),
                            units=structure[item].get('units', ''),
                            description=structure[item].get('documentation', ''),
                            column_style=column_style
                        ))
                except Exception:
                    printe(item)
                    raise
        lines.append('</table>')
        lines.append('</table><p></p>')

    with open(filename, 'w') as f:
        if sys.version_info < (3, 0):
            f.write('\n'.join(lines).encode('utf-8'))
        else:
            f.write('\n'.join(lines))
