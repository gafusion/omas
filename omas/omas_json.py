'''save/load from JSON routines

-------
'''

from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS


# ---------------------------
# save and load OMAS to Json
# ---------------------------
def save_omas_json(ods, filename, objects_encode=None, **kw):
    """
    Save an OMAS data set to Json

    :param ods: OMAS data set

    :param filename: filename or file descriptor to save to

    :param objects_encode: how to handle non-standard JSON objects
        * True: encode numpy arrays, complex, and uncertain
        * None: numpy arrays as lists, encode complex, and uncertain
        * False: numpy arrays as lists, fail on complex, and uncertain

    :param kw: arguments passed to the json.dumps method
    """

    printd('Saving OMAS data to Json: %s' % filename, topic=['Json', 'json'])

    kw.setdefault('indent', 0)
    kw.setdefault('separators', (',', ': '))
    kw.setdefault('sort_keys', True)

    json_string = json.dumps(ods, default=lambda x: json_dumper(x, objects_encode), **kw)

    if isinstance(filename, basestring):
        with open(filename, 'w') as f:
            f.write(json_string)
    else:
        f = filename
        f.write(json_string)


def load_omas_json(filename, consistency_check=True, imas_version=omas_rcparams['default_imas_version'], **kw):
    """
    Load an OMAS data set from Json

    :param filename: filename or file descriptor to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :param imas_version: imas version to use for consistency check

    :param kw: arguments passed to the json.loads mehtod

    :return: OMAS data set
    """

    printd('Loading OMAS data to Json: %s' % filename, topic='json')

    def cls():
        tmp = ODS(imas_version=imas_version)
        tmp.consistency_check = False
        return tmp

    if isinstance(filename, basestring):
        with open(filename, 'r') as f:
            json_string = f.read()
    else:
        f = filename
        json_string = f.read()

    tmp = json.loads(json_string, object_pairs_hook=lambda x: json_loader(x, cls), **kw)
    tmp.consistency_check = consistency_check
    return tmp


def through_omas_json(ods):
    """
    Test save and load OMAS Json

    :param ods: ods

    :return: ods
    """
    if not os.path.exists(tempfile.gettempdir() + '/OMAS_TESTS/'):
        os.makedirs(tempfile.gettempdir() + '/OMAS_TESTS/')
    filename = tempfile.gettempdir() + '/OMAS_TESTS/test.json'
    save_omas_json(ods, filename)
    ods1 = load_omas_json(filename)
    return ods1
