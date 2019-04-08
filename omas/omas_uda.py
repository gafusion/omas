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
                  verbose=True, assume_uniform_array_structures=False, skip_ggd=True):
    """Load pyuda data to OMAS


    .....

    :return: populated ODS
    """

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

    client = pyuda.Client()

    # if paths is None then figure out what IDS are available and get ready to retrieve everything
    if paths is None:
        requested_paths = [[structure] for structure in list_structures(imas_version=imas_version)]
    else:
        requested_paths = map(p2l, paths)

    available_ds=[]
    for ds in numpy.unique([p[0] for p in requested_paths]):
        if ds in add_datastructures.keys():
            continue

        if uda_get(client, [ds,'ids_properties','homogeneous_time'], pulse, run) is None:
            if verbose:
                print('- ', ds)
            continue
        if verbose:
            print('* ', ds)
        available_ds.append(ds)

    ods = ODS()
    for ds in available_ds:
        filled_paths_in_uda(ods, client, pulse, run, load_structure(ds, imas_version=imas_version)[1], [], [], requested_paths, skip_ggd=skip_ggd)

    return ods


def filled_paths_in_uda(ods, client, pulse, run, ds, path=None, paths=None, requested_paths=None, assume_uniform_array_structures=False, skip_ggd=False):
    """
    Taverse an IDS and list leaf paths (with proper sizing for arrays of structures)

    ....

    :param ds: hierarchical data schema as returned for example by load_structure('equilibrium')[1]

    :param assume_uniform_array_structures: assume that the first structure in an array of structures has data in the same nodes locations of the later structures in the array

    :param skip_ggd: do not traverse ggd structures

    :return: returns list of paths in an IDS that are filled
    """
    if path is None:
        path = []

    if paths is None:
        paths = []

    if requested_paths is None:
        requested_paths = []

    # leaf
    if not len(ds):
        # append path if it has data
        data = uda_get(client, path, pulse, run)
        if data is not None:
            print(l2o(path))
            ods[path] = data
            paths.append(path)
        return paths

    # keys
    keys = list(ds.keys())
    if keys[0] == ':':
        n = uda_get_shape(client, path, pulse, run)
        if n is None:
            return paths
        keys = range(n)
        if len(keys) and assume_uniform_array_structures:
            keys = [0]

    # kid must be part of this list
    if len(requested_paths):
        request_check = [p[0] for p in requested_paths]

    # traverse
    for kid in keys:

        # skip ggd structures
        if skip_ggd and kid in ['ggd', 'grids_ggd']:
            continue

        propagate_path = copy.copy(path)
        propagate_path.append(kid)

        # generate requested_paths one level deeper
        propagate_requested_paths = requested_paths
        if len(requested_paths):
            if kid in request_check or (isinstance(kid, int) and ':' in request_check):
                propagate_requested_paths = [p[1:] for p in requested_paths if len(p)>1 and (kid == p[0] or p[0]==':')]
            else:
                continue

        # recursive call
        if isinstance(kid, basestring):
            subtree_paths = filled_paths_in_uda(ods, client, pulse, run, ds[kid], propagate_path, [], propagate_requested_paths, assume_uniform_array_structures, skip_ggd=skip_ggd)
        else:
            subtree_paths = filled_paths_in_uda(ods, client, pulse, run, ds[':'], propagate_path, [], propagate_requested_paths, assume_uniform_array_structures, skip_ggd=skip_ggd)
        paths += subtree_paths

        # assume_uniform_array_structures
        if assume_uniform_array_structures and keys[0] == 0:
            zero_paths = subtree_paths
            for key in range(1, len(ids)):
                subtree_paths = copy.deepcopy(zero_paths)
                for p in subtree_paths:
                    p[len(path)] = key
                paths += subtree_paths

    return paths


def uda_get_shape(client, path, pulse, run):
    return uda_get(client, path+['Shape_of'], pulse, run)


def offset(path, off):
    return [p if isinstance(p,basestring) else p+off for p in path]


def uda_get(client, path, pulse, run):
    try:
        location = l2o(offset(path,+1)).replace('.', '/')
        tmp=client.get(location, pulse)
        if isinstance(tmp,pyuda._string.String):
            return tmp.str
        else:
            return tmp.data
    except pyuda.UDAException:
        return None
