'''save/load from JSON routines

-------
'''

from .omas_utils import *
from .omas_core import ODS, ODC


# ---------------------------
# save and load OMAS to Json
# ---------------------------
def save_omas_json(ods, filename, objects_encode=None, **kw):
    """
    Save an ODS to Json

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

    if isinstance(filename, str):
        with open(filename, 'w') as f:
            f.write(json_string)
    else:
        f = filename
        f.write(json_string)


def load_omas_json(filename, consistency_check=True, imas_version=omas_rcparams['default_imas_version'], cls=ODS, **kw):
    """
    Load ODS or ODC from Json

    :param filename: filename or file descriptor to load from

    :param consistency_check: verify that data is consistent with IMAS schema

    :param imas_version: imas version to use for consistency check

    :param cls: class to use for loading the data

    :param kw: arguments passed to the json.loads mehtod

    :return: OMAS data set
    """

    printd('Loading OMAS data from Json: %s' % filename, topic='json')

    if isinstance(filename, str):
        with open(filename, 'r') as f:
            json_string = f.read()
    else:
        json_string = filename.read()

    # allow for empty json file
    if not len(json_string.strip()):
        return ODS(imas_version=imas_version, consistency_check=consistency_check)

    def base_class(x):
        clsODS = lambda: ODS(imas_version=imas_version, consistency_check=False)
        clsODC = lambda: ODC(imas_version=imas_version, consistency_check=False)
        try:
            tmp = json_loader(x, clsODS, null_to=numpy.NaN)
        except Exception:
            tmp = json_loader(x, clsODC, null_to=numpy.NaN)
        return tmp

    tmp = json.loads(json_string, object_pairs_hook=lambda x: base_class(x), **kw)

    # convert to cls
    tmp.__class__ = cls

    # perform consistency check
    tmp.consistency_check = consistency_check

    return tmp


def through_omas_json(ods, method=['function', 'class_method'][1]):
    """
    Test save and load OMAS Json

    :param ods: ods

    :return: ods
    """
    filename = omas_testdir(__file__) + '/test.json'
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
    if method == 'function':
        save_omas_json(ods, filename)
        ods1 = load_omas_json(filename)
    else:
        ods.save(filename)
        ods1 = ODS().load(filename)
    return ods1
