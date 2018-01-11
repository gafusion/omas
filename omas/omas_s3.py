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
    Save an OMAS object to pickle and upload it to S3

    :param ods: OMAS data set

    :param filename: filename to save to

    :param user: username where to look for the file

    :param kw: arguments passed to the save_omas_pkl function
    """
    printd('Saving to %s on S3' % (_base_S3_uri(user) + filename), topic='s3')

    save_omas_pkl(ods, filename, **kw)
    return remote_uri(_base_S3_uri(user), filename, 'up')


def load_omas_s3(filename, user=os.environ['USER']):
    """
    Download an OMAS object from S3 and read it as pickle

    :param filename: filename to load from

    :param user: username where to look for the file

    :return: OMAS data set
    """
    printd('loading from %s on S3' % (_base_S3_uri(user) + filename), topic='s3')

    remote_uri(_base_S3_uri(user) + filename, None, 'down')
    return load_omas_pkl(os.path.split(filename)[1])


def list_omas_s3(user=''):
    """
    List S3 content

    :param user: username where to look for the file

    :return: OMAS data set
    """
    return remote_uri(_base_S3_uri(user), None, 'list')


def del_omas_s3(filename, user=os.environ['USER']):
    """
    Delete an OMAS object from S3

    :param user: username where to look for the file

    :return: OMAS data set
    """
    remote_uri(_base_S3_uri(user) + filename, None, 'del')


def scenario_database(machine=None, shot=None, run=None):
    """
    List and retrieve available IMAS scenarios

    :param machine: string with the machine name of the scenario

    :param shot: shot number

    :param run: run number

    :return: OMAS data set with the requested scenario
    """
    if not machine and not shot and not run:
        remote_uri(_base_S3_uri('omas_shared') + 'scenario_summary.txt', None, 'down')
        return open('scenario_summary.txt', 'r').read()

    elif machine and shot and run:
        print('Fetching scenario file: {machine}_{shot}_{run}.pkl'.format(machine=machine, shot=shot, run=run))
        return load_omas_s3('{machine}_{shot}_{run}.pkl'.format(machine=machine, shot=shot, run=run),
                            user='omas_shared')

    else:
        raise (Exception('machine, shot, run must either all be None or all be set'))


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
