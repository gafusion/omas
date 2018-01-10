from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import save_omas_pkl, load_omas_pkl


def _base_S3_uri(user):
    return 's3://omas3/{user}/'.format(user=user)


# --------------------------------------------
# save and load OMAS with S3
# --------------------------------------------
def save_omas_s3(ods, filename, user=os.environ['USER'], **kw):
    """
    Save an OMAS data set to pickle and upload it to S3

    :param ods: OMAS data set

    :param filename: filename to save to

    :param kw: arguments passed to the save_omas_pkl function
    """
    printd('Saving to %s on S3' % (_base_S3_uri(user) + filename), topic='s3')

    save_omas_pkl(ods, filename, **kw)
    return remote_uri(_base_S3_uri(user), filename, 'up')


def load_omas_s3(filename, user=os.environ['USER']):
    """
    Download an OMAS data set from S3 and read it as pickle

    :param filename: filename to load from

    :return: OMAS data set
    """
    printd('loading from %s on S3' % (_base_S3_uri(user) + filename), topic='s3')

    remote_uri(_base_S3_uri(user) + filename, None, 'down')
    return load_omas_pkl(os.path.split(filename)[1])


def test_omas_s3(ods):
    """
    test save and load S3

    :param ods: ods

    :return: ods
    """
    filename = 'test.pkl'
    save_omas_s3(ods, filename, user='omas_test')
    ods1 = load_omas_s3(filename, user='omas_test')
    return ods1
