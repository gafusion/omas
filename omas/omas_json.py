from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import ODS


# ---------------------------
# save and load OMAS to Json
# ---------------------------
def save_omas_json(ods, filename, **kw):
    """
    Save an OMAS data set to Json

    :param ods: OMAS data set

    :param filename: filename to save to

    :param kw: arguments passed to the json.dumps method
    """

    printd('Saving OMAS data to Json: %s' % filename, topic=['Json', 'json'])

    json_string = json.dumps(ods, default=json_dumper, indent=0, separators=(',', ': '), **kw)
    open(filename, 'w').write(json_string)


def load_omas_json(filename, **kw):
    """
    Load an OMAS data set from Json

    :param filename: filename to load from

    :param kw: arguments passed to the json.loads mehtod

    :return: OMAS data set
    """

    printd('Loading OMAS data to Json: %s' % filename, topic='json')

    if isinstance(filename, basestring):
        filename = open(filename, 'r')

    def cls():
        tmp = ODS()
        tmp.consistency_check = False
        return tmp

    tmp = json.loads(filename.read(), object_pairs_hook=lambda x: json_loader(x, cls), **kw)

    tmp1 = ODS()
    for item in tmp.flat():
        tmp1[item] = tmp[item]

    return tmp1


def test_omas_json(ods):
    """
    test save and load OMAS Json

    :param ods: ods

    :return: ods
    """
    filename = 'test.json'
    save_omas_json(ods, filename)
    ods1 = load_omas_json(filename)
    return ods1
