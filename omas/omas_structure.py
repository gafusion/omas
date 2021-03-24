'''functions for handling IMAS XML data dictionaries

-------
'''

from .omas_utils import *

# add support for occurrences to each IDS
number_of_omas_only_add_datastructures_entries = 0
for structure in sorted(list(structures_filenames(omas_rcparams['default_imas_version']).keys())):
    add_datastructures[structure] = {
        f"{structure}.ids_properties.occurrence": {
            "full_path": f"{structure}.ids_properties.occurrence",
            "data_type": "INT_0D",
            "description": "occurrence number [NOTE: this field only exists in OMAS and is not part of the ITER PDM]",
        }
    }
    number_of_omas_only_add_datastructures_entries = len(add_datastructures[structure])

# --------------------------------------------
# generation of the imas structure json files
# IDS's XML documentation can be found where IMAS is installed at: $IMAS_PREFIX/share/doc/imas/html_documentation.html
# --------------------------------------------
def generate_xml_schemas(imas_version=None):
    """
    Generate IMAS IDSDef.xml files by:
     1. clone the IMAS data-dictionary repository (access to git.iter.org required)
     2. download the Saxon XSLT and XQuery Processor
     3. generate IDSDef.xml in imas_structures folder
    """
    import subprocess

    saxon_version = 'SaxonHE9-9-1-4J'
    saxon_major_version = '9.' + saxon_version.split('-')[1]

    # clone the IMAS data-dictionary repository
    dd_folder = os.sep.join([imas_json_dir, '..', 'data-dictionary'])
    if not os.path.exists(dd_folder):
        subprocess.Popen(
            'cd %s ; git clone ssh://git@git.iter.org/imas/data-dictionary.git' % os.sep.join([imas_json_dir, '..']),
            stdout=subprocess.PIPE,
            shell=True,
        ).communicate()[0]

    # download Saxon
    sax_folder = os.sep.join([imas_json_dir, '..', saxon_version])
    if not os.path.exists(sax_folder):
        subprocess.Popen(
            """
        cd {install_dir}
        curl https://iweb.dl.sourceforge.net/project/saxon/Saxon-HE/9.9/{saxon_version}.zip > {saxon_version}.zip
        unzip -d {saxon_version} {saxon_version}.zip
        rm {saxon_version}.zip""".format(
                install_dir=os.sep.join([imas_json_dir, '..']), saxon_major_version=saxon_major_version, saxon_version=saxon_version
            ),
            shell=True,
        ).communicate()

    # fetch data structure updates
    subprocess.Popen(
        """
    cd {dd_folder}
    git fetch
    """.format(
            dd_folder=dd_folder
        ),
        shell=True,
    ).communicate()

    # find IMAS data-dictionary tags
    result = b2s(subprocess.Popen('cd %s;git tag' % dd_folder, stdout=subprocess.PIPE, shell=True).communicate()[0])
    tags = list(filter(lambda x: str(x).startswith('3.') and int(x.split('.')[1]) >= 10, result.split()))
    # add development branch at the beginning of list of tags
    tags.insert(0, 'develop/3')
    imas_versions = OrderedDict()
    for item in tags:
        imas_versions[item] = item.replace('.', '_').replace('/', '_')
    if imas_version is None:
        return imas_versions

    # generate IDSDef.xml files for given imas_version
    _imas_version = imas_versions[imas_version]
    executable = """
export CLASSPATH={imas_json_dir}/../{saxon_version}/saxon9he.jar;
cd {dd_folder}
git checkout {tag}
git pull
export JAVA_HOME=$(dirname $(dirname `which java`))
make clean
make
rm -rf {imas_json_dir}/{_imas_version}/
mkdir {imas_json_dir}/{_imas_version}/
cp IDSDef.xml {imas_json_dir}/{_imas_version}/
""".format(
        tag=imas_version, _imas_version=_imas_version, imas_json_dir=imas_json_dir, dd_folder=dd_folder, saxon_version=saxon_version
    )
    print(executable)
    subprocess.Popen(executable, shell=True).communicate()


def create_json_structure(imas_version=omas_rcparams['default_imas_version']):
    import xmltodict

    file = imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + 'IDSDef.xml'
    tmp = xmltodict.parse(open(file).read())

    for file in glob.glob(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '*.json'):
        print('Remove ' + file)
        os.remove(file)

    def process_path(inv):
        inv = inv.replace(r'/', '.')
        inv = inv.split(' OR ')[0]
        inv = remove_parentheses(inv, '[:]')
        inv = re.sub('\[:\]$', '', inv)
        return inv

    def traverse(me, hout, path, fout, parent):
        try:
            fname = None
            me = copy.copy(me)
            hout_propagate = hout
            path_propagate = copy.deepcopy(path)
            parent = copy.deepcopy(parent)

            if '@structure_reference' in me and me['@structure_reference'] == 'self':
                return hout, fout

            if '@name' in me:
                name = me['@name']
                path_propagate.append(name)
                hout_propagate = {}
                hout[name] = hout_propagate
                # paths
                if '@path_doc' in me:
                    fname = path_propagate[0] + '/' + me['@path_doc']
                    me['@full_path'] = path_propagate[0] + '/' + me['@path_doc']
                    del me['@path_doc']
                else:
                    fname = path_propagate[0]
                    me['@full_path'] = path_propagate[0]
                if '@path' in me:
                    del me['@path']
                # coordinates
                for coord in [c for c in me if c.startswith('@coordinate')]:
                    if '...' not in me[coord]:
                        me[coord] = path_propagate[0] + '/' + me[coord]
                # identifiers documentation
                if '@doc_identifier' in me:
                    doc_id = xmltodict.parse(open(imas_json_dir + '/../data-dictionary/' + me['@doc_identifier']).read())
                    hlp = doc_id['constants']['int']
                    doc = []
                    for row in hlp:
                        doc.append('%s) %s : %s' % (row['#text'], row['@name'], row['@description']))
                    me['@documentation'] = me['@documentation'].strip() + '\n' + '\n'.join(doc)

                fname = process_path(fname)
                fout[fname] = {}

            if isinstance(me, list):
                keys = range(len(me))
            else:
                keys = list(me.keys())

            if '@units' in me:
                if fname == 'equilibrium.time_slice[:].constraints.q':  # bug fix for v3.18.0
                    me['@units'] = '-'
                if fname in [
                    'equilibrium.time_slice[:].profiles_1d.geometric_axis.r',
                    'equilibrium.time_slice[:].profiles_1d.geometric_axis.z',
                ]:
                    me['@coordinate'] = ['equilibrium.time_slice[:].profiles_1d.psi']
                if me['@units'] in ['as_parent', 'as parent', 'as_parent_level_2']:
                    me['@units'] = parent['units']
                parent['units'] = me['@units']

            # children inherit lifecycle status from parent
            if '@lifecycle_status' in me:
                parent['lifecycle_status'] = me['@lifecycle_status']
            elif parent['lifecycle_status'] and not isinstance(me, list):
                me['@lifecycle_status'] = parent['lifecycle_status']
                keys.append('@lifecycle_status')

            is_leaf = True
            for kid in keys:
                if isinstance(me[kid], (dict, list)):
                    is_leaf = False
                    traverse(me[kid], hout_propagate, path_propagate, fout, parent)
                elif fname and fname in fout and kid not in ['@name', '@xmlns:fn']:
                    hout_propagate[kid] = me[kid]
                    fout[fname][kid] = me[kid]

                    # if is_leaf:
                    # print(path_propagate)
        except Exception:
            pprint(fout)
            raise
        return hout, fout

    parent = {}
    parent['units'] = '?'
    parent['lifecycle_status'] = ''
    hout, fout = traverse(me=tmp, hout={}, path=[], fout={}, parent=parent)

    # format conversions
    for item in sorted(fout):
        coords = []
        for key in sorted(list(fout[item].keys())):
            if key.endswith('AosParent_relative') or key.endswith('_same_as'):
                del fout[item][key]
            elif key != '@coordinates' and key.startswith('@coordinate'):
                coords.append(fout[item][key])
                del fout[item][key]
            elif key.startswith('@path'):
                fout[item][key] = process_path(fout[item][key])
            elif key == '@data_type':
                if fout[item][key] == 'flt_type':
                    fout[item][key] = 'FLT_0D'
                else:
                    fout[item][key] = fout[item][key].upper()
        if len(coords):
            # # this is a check for duplicated coordinates, which it is not an error per se
            # if len(numpy.unique(list(filter(lambda x: not x.startswith('1...'), coords)))) != len(list(filter(lambda x: not x.startswith('1...'), coords))):
            #     printe('%s -X-> %s' % (item, coords))
            coords = list(map(process_path, coords))
            fout[item]['@coordinates'] = coords

    # deal with cross-referenced coordinates
    for item in sorted(fout):
        if '@coordinates' in fout[item]:
            fout[item]['@coordinates'] = list(map(lambda x: re.sub('^.*\.IDS:', '', x), fout[item]['@coordinates']))

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
        keys = list(map(lambda x: re.sub('^@', '', x), keys))
        fout[item] = dict(zip(keys, values))

    # break into pieces
    hout = {}
    for item in sorted(fout):
        #    if re.findall('(_error_upper|_error_lower|_error_index)$', item):
        #        continue
        ds = item.split('.')[0]
        hout.setdefault(ds, {})[item] = fout[item]
        if ds in add_datastructures:
            hout[ds].update(add_datastructures[ds])

    # additional data structures
    for ds in add_datastructures:
        if ds not in hout and len(add_datastructures[ds]) > number_of_omas_only_add_datastructures_entries:
            hout[ds] = add_datastructures[ds]

    # prepare directory structure
    if not os.path.exists(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version)):
        os.makedirs(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version))
    for item in glob.glob(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '*.json'):
        os.remove(item)

    # deploy imas structures as json
    for structure in sorted(hout):
        if structure == 'time':
            continue
        print(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + structure + '.json')
        dump_string = json.dumps(hout[structure], default=json_dumper, indent=1, separators=(',', ': '), sort_keys=True)
        open(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + structure + '.json', 'w').write(dump_string)

    # generate coordinates cache file
    coords = extract_coordinates(imas_version=imas_version)
    dump_string = json.dumps(coords, default=json_dumper, indent=1, separators=(',', ': '), sort_keys=True)
    open(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '_coordinates.json', 'w').write(dump_string)

    # generate times cache file
    times = extract_times(imas_version=imas_version)
    dump_string = json.dumps(times, default=json_dumper, indent=1, separators=(',', ': '), sort_keys=True)
    open(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '_times.json', 'w').write(dump_string)

    # generate global_quantities cache file
    global_quantities = extract_global_quantities(imas_version=imas_version)
    dump_string = json.dumps(global_quantities, default=json_dumper, indent=1, separators=(',', ': '), sort_keys=True)
    open(imas_json_dir + os.sep + imas_versions.get(imas_version, imas_version) + os.sep + '_global_quantities.json', 'w').write(
        dump_string
    )


def create_html_documentation(imas_version=omas_rcparams['default_imas_version']):
    filename = os.path.abspath(os.sep.join([imas_json_dir, imas_versions.get(imas_version, imas_version), 'omas_doc.html']))

    table_header = "<table border=1, width='100%'>"
    sub_table_header = (
        '<tr>'
        '<th style="width:25%">Path</th>'
        '<th style="width:25%">Dimensions</th>'
        '<th>Type</th>'
        '<th>Units</th>'
        '<th>Description</th>'
        '</tr>'
    )

    column_style = 'style="word-wrap:break-word;word-break:break-all"'
    lines = []
    for structure_file in list_structures(imas_version=imas_version):
        print('Adding to html documentation: ' + structure_file)
        # load structure information
        structure = load_structure(structure_file, imas_version=imas_version)[0]
        # idenfity coordinates in the structure
        coords = []
        for item in structure:
            if 'coordinates' in structure[item]:
                coords.extend(structure[item]['coordinates'])
        coords = list(filter(lambda x: '...' not in x, set(coords)))
        # generate output
        lines.append('<!-- %s -->' % structure_file)
        lines.append(table_header)
        lines.append(sub_table_header)
        for item in sorted(structure):
            if not any(item.endswith(k) for k in ['_error_index', '_error_lower', '_error_upper']):
                # uncertain quantities
                is_uncertain = ''
                if item + '_error_upper' in structure:
                    is_uncertain = ' (uncertain)'
                # lifecycle status
                status = ''
                if 'lifecycle_status' in structure[item] and structure[item]['lifecycle_status'] not in ['active']:
                    color_mapper = {'alpha': 'blue', 'obsolescent': 'red'}
                    status = '</p><p><font color="%s">(%s)</font>' % (
                        color_mapper.get(structure[item]['lifecycle_status'], 'orange'),
                        structure[item]['lifecycle_status'],
                    )
                # highlight entries that are a coordinate
                item_with_coordinate_highlight = item
                if item in coords:
                    item_with_coordinate_highlight = '<strong>%s</strong>' % item
                try:
                    lines.append(
                        '<tr>'
                        '<td {column_style}><p>{item_with_coordinate_highlight}{status}</p></td>'
                        '<td {column_style}><p>{coordinates}</p></td>'
                        '<td><p>{data_type}</p></td>'
                        '<td><p>{units}</p></td>'
                        '<td><p>{description}</p></td>'
                        '</tr>'.format(
                            item_with_coordinate_highlight=item_with_coordinate_highlight,
                            coordinates=re.sub(
                                '\[\]',
                                '',
                                re.sub('[\'\"]', '', re.sub(',', ',<br>', str(list(map(str, structure[item].get('coordinates', '')))))),
                            ),
                            data_type=structure[item].get('data_type', '') + is_uncertain,
                            units=structure[item].get('units', ''),
                            description=re.sub('\n', '<br>', structure[item].get('documentation', '')),
                            status=status,
                            column_style=column_style,
                        )
                    )
                except Exception:
                    printe(item)
                    raise
        lines.append('</table>')
        lines.append('</table><p></p>')

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def extract_coordinates(imas_version=omas_rcparams['default_imas_version']):
    """
    return list of strings with coordinates across all structures

    :param imas_version: imas version

    :return: list with coordinate
    """
    from omas.omas_utils import list_structures
    from omas.omas_utils import load_structure

    coordinates = set()
    for structure in list_structures(imas_version=imas_version):
        tmp = load_structure(structure, imas_version)[0]
        coords = []
        for item in tmp:
            if 'coordinates' in tmp[item]:
                coords.extend(map(i2o, tmp[item]['coordinates']))
        coords = list(filter(lambda x: '...' not in x, set(coords)))
        coordinates.update(coords)

    return sorted(list(coordinates))


def extract_times(imas_version=omas_rcparams['default_imas_version']):
    """
    return list of strings with .time across all structures

    :param imas_version: imas version

    :return: list with times
    """
    from omas.omas_utils import list_structures
    from omas.omas_utils import load_structure

    times = []
    for structure in list_structures(imas_version=imas_version):
        tmp = load_structure(structure, imas_version)[0]

        for item in tmp:
            if not item.endswith('.time') or 'data_type' not in tmp[item] or tmp[item]['data_type'] == 'STRUCTURE':
                continue
            times.append(item)

    return sorted(times)


def extract_global_quantities(imas_version=omas_rcparams['default_imas_version']):
    """
    return list of strings with .global_quantities across all structures

    :param imas_version: imas version

    :return: list with times
    """
    from omas.omas_utils import list_structures
    from omas.omas_utils import load_structure

    global_quantities = []
    for structure in list_structures(imas_version=imas_version):
        tmp = load_structure(structure, imas_version)[0]

        for item in tmp:
            if any(item.endswith(error) for error in ['_error_index', '_error_lower', '_error_upper']):
                continue
            elif not item.endswith('.global_quantities.' + item.split('.')[-1]):
                continue
            global_quantities.append(item)

    return sorted(global_quantities)


def extract_ggd(imas_version=omas_rcparams['default_imas_version']):
    """
    return list of strings endingwith .ggd or .grids_ggd across all structures

    :param imas_version: imas version

    :return: list with times
    """
    from omas.omas_utils import list_structures
    from omas.omas_utils import load_structure

    omas_ggd = []
    for structure in list_structures(imas_version=imas_version):
        tmp = load_structure(structure, imas_version)[0]

        for item in tmp:
            if item.endswith('.ggd') or item.endswith('.grids_ggd'):
                omas_ggd.append(item)

    return sorted(omas_ggd)


def extract_cocos(imas_version=omas_rcparams['default_imas_version']):
    """
    return dictionary of entries with cocos transformations across all structures

    :param imas_version: imas version

    :return: dictionary with cocos transformations
    """
    from omas.omas_utils import list_structures
    from omas.omas_utils import load_structure
    from omas.omas_utils import i2o

    cocos_mapper = {}
    cocos_mapper["'1'"] = '--delete--'
    cocos_mapper[".sigma_ip_eff"] = 'TOR'
    cocos_mapper[".fact_q"] = 'Q'
    cocos_mapper[".fact_psi"] = 'PSI'
    cocos_mapper[".sigma_b0_eff"] = 'TOR'
    cocos_mapper[".fact_dodpsi"] = 'dPSI'
    cocos_mapper[".fact_dtheta"] = 'POL'
    cocos_mapper[".sigma_rphiz_eff"] = 'TOR'
    cocos_mapper["grid_type_transformation(index_grid_type,1)"] = '--delete--'
    cocos_mapper["grid_type_transformation(index_grid_type,2)"] = '--delete--'
    cocos_mapper["grid_type_transformation(index_grid_type,3)"] = '--delete--'
    cocos_mapper["grid_type_transformation(index_grid_type,4)"] = '--delete--'
    cocos_mapper[".fact_dim1*.fact_dim1"] = '--delete--'
    cocos_mapper[".fact_dim1*.fact_dim2"] = '--delete--'
    cocos_mapper[".fact_dim1*.fact_dim3"] = '--delete--'
    cocos_mapper[".fact_dim2*.fact_dim2"] = '--delete--'
    cocos_mapper[".fact_dim2*.fact_dim3"] = '--delete--'
    cocos_mapper[".fact_dim3*.fact_dim3"] = '--delete--'

    omas_cocos = {}
    for structure in list_structures(imas_version=imas_version):
        tmp = load_structure(structure, imas_version)[0]
        for item in tmp:
            if 'cocos_transformation_expression' in tmp[item]:
                cocos = tmp[item]['cocos_transformation_expression']
                cocos = cocos_mapper.get(cocos, cocos)
                if cocos != '--delete--':
                    omas_cocos[i2o(item)] = cocos

    return omas_cocos


def symlink_imas_structure_versions(test=True, verbose=True):
    """
    Generate symbolic links in imas_structures so that no files are added when there are no changes between IDSs

    :param test: whether to actually apply symlink commands

    :param verbose: wheter to print to screen structures strides

    :returns: dictionary with structure stides per IDS
    """

    import subprocess
    from pprint import pprint
    from omas.omas_setup import IMAS_versions
    from omas.omas_utils import structures_filenames, imas_json_dir

    imas_versions = IMAS_versions('tagged')

    # check if two files are identical
    def same_ds(a, b):
        if not os.path.exists(a) or not os.path.exists(b):
            return False
        with open(a) as a:
            with open(b) as b:
                return a.read() == b.read()

    # see when DS files have changed
    structures_strides = {}
    previous_tag_structures = {}
    for version in list(imas_versions.keys()):
        this_tag_structures = structures_filenames(version)
        for ds in this_tag_structures:
            if ds not in structures_strides:
                structures_strides[ds] = [[version]]
            elif ds in previous_tag_structures:
                if same_ds(previous_tag_structures[ds], this_tag_structures[ds]):
                    structures_strides[ds][-1].append(version)
                else:
                    structures_strides[ds].append([version])
        previous_tag_structures = this_tag_structures

    # print strides
    if verbose or not test:
        pprint(structures_strides)

    # apply symlinks
    if not test:
        for ds in structures_strides:
            for stride in structures_strides[ds]:
                if len(stride) > 1:
                    for version in stride[:-1]:
                        dir = imas_json_dir + '/'
                        this = structures_filenames(stride[-1])[ds][len(dir) :]
                        prev = structures_filenames(version)[ds]
                        command = 'cd %s; ln -s -f ../%s %s' % (os.path.dirname(prev), this, os.path.basename(prev))
                        subprocess.Popen(command, shell=True).communicate()
    return structures_strides


def add_extra_structures(extra_structures, lifecycle_status='tmp'):
    '''
    Function used to extend the IMAS data dictionary with user defined structures

    :param extra_structures: dictionary with extra IDS entries to assign

    :param lifecycle_status: default lifecycle_status to assign
    '''
    from . import omas_utils

    # reset structure caches
    omas_utils._structures = {}
    omas_utils._structures_dict = {}

    # add _structures
    for _ids in extra_structures:
        omas_utils._extra_structures.setdefault(_ids, {}).update(extra_structures[_ids])
        for _item in extra_structures[_ids]:
            omas_utils._extra_structures[_ids][_item].setdefault('lifecycle_status', lifecycle_status)
