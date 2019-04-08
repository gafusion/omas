'''save/load from UDA routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_core import ODS
from collections import Sequence
import numpy

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

def get_uda(client, location, shot, run):
    try:
        location = location.replace('.', '/')
        return client.get(location, shot)
    except pyuda.UDAException:
        return None

def load_omas_uda(server=None, port=None, pulse=None, run=0, paths=None,
                  imas_version=os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version']),
                  verbose=True):
    """Load pyuda data to OMAS

    :param hdc: input data structure

    :param consistency_check: verify that data is consistent with IMAS schema

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

    fetch_paths = []
    for ds in numpy.unique([p[0] for p in requested_paths]):
        if ds in add_datastructures.keys():
            continue

        if uda_get(client, location=ds + '.ids_properties.homogeneous_time', pulse=pulse, run=run) is None:
            if verbose:
                print('- ', ds)
            continue
        if verbose:
            print('* ', ds)

    return ods
