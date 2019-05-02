'''save/load from UDA routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS

try:
    _pyyda_import_excp = None
    import pyuda
except ImportError as _excp:
    _pyyda_import_excp = _excp


    # replace pyuda class by a simple exception throwing class
    class pyuda(object):
        """Import error UDA class"""

        def __init__(self, *args, **kwargs):
            raise _pyuda_import_excp


def load_omas_uda(server=None, port=None, pulse=None, run=0, paths=None,
                  imas_version=os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version']),
                  skip_uncertainties=False, skip_ggd=True, verbose=True):
    '''
    Load UDA data to OMAS

    :param server: UDA server

    :param port: UDA port

    :param pulse: UDA pulse

    :param run: UDA run

    :param paths: list of paths to load from IMAS

    :param imas_version: IMAS version

    :param skip_uncertainties: do not load uncertain data

    :param skip_ggd: do not load ggd structure

    :param verbose: print loading progress

    :return: OMAS data set
    '''

    if pulse is None or run is None:
        raise Exception('`pulse` and `run` must be specified')

    if server is not None:
        pyuda.Client.server = server
    elif not os.environ['UDA_HOST']:
        raise pyuda.UDAException('Must set UDA_HOST environmental variable')

    if port is not None:
        pyuda.Client.port = port
    elif not os.environ['UDA_PORT']:
        raise pyuda.UDAException('Must set UDA_PORT environmental variable')

    # set this to get pyuda metadata (maybe of interest for future use):
    # pyuda.Client.set_property(pyuda.Properties.PROP_META, True)

    client = pyuda.Client()

    # if paths is None then figure out what IDS are available and get ready to retrieve everything
    if paths is None:
        requested_paths = [[structure] for structure in list_structures(imas_version=imas_version)]
    else:
        requested_paths = map(p2l, paths)

    available_ds = []
    for ds in numpy.unique([p[0] for p in requested_paths]):
        if ds in add_datastructures.keys():
            continue

        if uda_get(client, [ds, 'ids_properties', 'homogeneous_time'], pulse, run) is None:
            if verbose:
                print('- ', ds)
            continue
        if verbose:
            print('* ', ds)
        available_ds.append(ds)

    ods = ODS(consistency_check=False)
    for k, ds in enumerate(available_ds):
        filled_paths_in_uda(ods, client, pulse, run, load_structure(ds, imas_version=imas_version)[1],
                            path=[], paths=[], requested_paths=requested_paths,
                            skip_uncertainties=skip_uncertainties, skip_ggd=skip_ggd,
                            perc=[float(k) / len(available_ds) * 100, float(k + 1) / len(available_ds) * 100, float(k) / len(available_ds) * 100])
    ods.consistency_check = True
    ods.prune()
    if verbose:
        print()
    return ods


def filled_paths_in_uda(ods, client, pulse, run, ds, path, paths, requested_paths, skip_uncertainties, skip_ggd, perc=[0., 100., 0.]):
    '''
    Recursively traverse ODS and populate it with data from UDA

    :param ods: ODS to be filled

    :param client: UDA client

    :param pulse: UDA pulse

    :param run: UDA run

    :param ds: hierarchical data schema as returned for example by load_structure('equilibrium')[1]

    :param path: []

    :param paths: []

    :param requested_paths: list of paths that are requested

    :param skip_uncertainties: do not load uncertain data

    :param skip_ggd: do not load ggd structure

    :return: filled ODS
    '''
    # leaf
    if not len(ds):
        return paths

    # keys
    keys = list(ds.keys())
    if keys[0] == ':':
        n = uda_get_shape(client, path, pulse, run)
        if n is None:
            return paths
        keys = range(n)

    # kid must be part of this list
    if len(requested_paths):
        request_check = [p[0] for p in requested_paths]

    # traverse
    n = float(len(keys))
    for k, kid in enumerate(keys):

        # skip ggd structures
        if skip_ggd and kid in ['ggd', 'grids_ggd']:
            continue

        if isinstance(kid, basestring):
            if skip_uncertainties and kid.endswith('_error_upper'):
                continue
            if kid.endswith('_error_lower') or kid.endswith('_error_index'):
                continue
            kkid = kid
        else:
            kkid = ':'

        # leaf
        if not len(ds[kkid]):
            # append path if it has data
            data = uda_get(client, path + [kid], pulse, run)
            if data is not None:
                # print(l2o(path))
                ods[kid] = data
                paths.append(path + [kid])

            pp = perc[0] + (k + 1) / n * (perc[1] - perc[0])
            if (pp - perc[2]) > 2:
                perc[2] = pp
                print('\rLoading: %3.1f%%' % pp, end='')

        propagate_path = copy.copy(path)
        propagate_path.append(kid)

        # generate requested_paths one level deeper
        propagate_requested_paths = requested_paths
        if len(requested_paths):
            if kid in request_check or (isinstance(kid, int) and ':' in request_check):
                propagate_requested_paths = [p[1:] for p in requested_paths if len(p) > 1 and (kid == p[0] or p[0] == ':')]
            else:
                continue

        # recursive call
        pp0 = perc[0] + k / n * (perc[1] - perc[0])
        pp1 = perc[0] + (k + 1) / n * (perc[1] - perc[0])
        pp2 = perc[2]
        paths = filled_paths_in_uda(ods[kid], client, pulse, run, ds[kkid], propagate_path, [], propagate_requested_paths, skip_uncertainties, skip_ggd, [pp0, pp1, pp2])

    # generate uncertain data
    if not skip_uncertainties and isinstance(ods.omas_data, dict):
        for kid in list(ods.omas_data.keys()):
            if kid.endswith('_error_upper') and kid[:-len('_error_upper')] in ods.omas_data:
                try:
                    if isinstance(ods[kid], ODS):
                        pass
                    elif isinstance(ods[kid], float):
                        ods[kid[:-len('_error_upper')]] = ufloat(ods[kid[:-len('_error_upper')]], ods[kid])
                    else:
                        ods[kid[:-len('_error_upper')]] = uarray(ods[kid[:-len('_error_upper')]], ods[kid])
                    del ods[kid]
                except Exception as _excp:
                    printe('Error loading uncertain data: %s' % kid)
    return paths


def uda_get_shape(client, path, pulse, run):
    '''
    Get the number of elements in a structure of arrays

    :param client: pyuda.Client object

    :param path: ODS path expressed as list

    :param pulse: UDA pulse

    :param run: UDA run

    :return: integer
    '''
    return uda_get(client, path + ['Shape_of'], pulse, run)


def offset(path, off):
    '''
    IMAS UDA indexing starts from one

    :param path: ODS path expressed as list

    :param off: offset to apply

    :return: path with applied offset
    '''
    return [p if isinstance(p, basestring) else p + off for p in path]


def uda_get(client, path, pulse, run):
    '''
    Get the data from UDA

    :param client: pyuda.Client object

    :param path: ODS path expressed as list

    :param pulse: UDA pulse

    :param run: UDA run

    :return: data
    '''
    try:
        location = l2o(offset(path, +1)).replace('.', '/')
        tmp = client.get(location, pulse)
        if isinstance(tmp, pyuda._string.String):
            return tmp.str
        else:
            return tmp.data
    except pyuda.UDAException:
        return None
