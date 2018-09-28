'''save/load from JSON routines

-------
'''

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

    json_string = json.dumps(ods, default=json_dumper, indent=0, separators=(',', ': '), sort_keys=True, **kw)
    open(filename, 'w').write(json_string)


def load_omas_json(filename, **kw):
    """
    Load an OMAS data set from Json

    :param filename: filename to load from

    :param kw: arguments passed to the json.loads mehtod

    :return: OMAS data set
    """

    printd('Loading OMAS data to Json: %s' % filename, topic='json')

    def cls():
        tmp = ODS()
        tmp.consistency_check = False
        return tmp

    if isinstance(filename, basestring):
        filename = open(filename, 'r')
    with filename:
        tmp = json.loads(filename.read(), object_pairs_hook=lambda x: json_loader(x, cls), **kw)
    tmp.consistency_check=True
    return tmp


def through_omas_json(ods):
    """
    test save and load OMAS Json

    :param ods: ods

    :return: ods
    """
    if not os.path.exists(tempfile.gettempdir()+'/OMAS_TESTS/'):
        os.makedirs(tempfile.gettempdir()+'/OMAS_TESTS/')
    filename = tempfile.gettempdir()+'/OMAS_TESTS/test.json'
    save_omas_json(ods, filename)
    ods1 = load_omas_json(filename)
    return ods1
