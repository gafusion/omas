from __future__ import print_function, division, unicode_literals

from .omas_utils import *
from .omas_core import save_omas_pkl, load_omas_pkl

def _base_S3_uri(user):
    return 's3://omas3/{user}/'.format(user=user)

# --------------------------------------------
# save and load OMAS with S3
# --------------------------------------------
def save_omas_s3(ods, filename, user=os.environ['USER'], tmp_dir=omas_rcparams['tmp_imas_dir'], **kw):
    """
    Save an OMAS object to pickle and upload it to S3

    :param ods: OMAS data set

    :param filename: filename to save to

    :param user: username where to look for the file

    :param tmp_dir: temporary folder for storing S3 file on local workstation

    :param kw: arguments passed to the save_omas_pkl function
    """
    printd('Saving to %s on S3' % (_base_S3_uri(user) + filename), topic='s3')

    if not os.path.exists(os.path.abspath(tmp_dir)):
        os.makedirs(os.path.abspath(tmp_dir))
    save_omas_pkl(ods, os.path.abspath(tmp_dir) + os.sep + os.path.split(filename)[1], **kw)
    return remote_uri(_base_S3_uri(user), os.path.abspath(tmp_dir) + os.sep + os.path.split(filename)[1], 'up')

def load_omas_s3(filename, user=os.environ['USER'], tmp_dir=omas_rcparams['tmp_imas_dir']):
    """
    Download an OMAS object from S3 and read it as pickle

    :param filename: filename to load from

    :param user: username where to look for the file

    :param tmp_dir: temporary folder for storing S3 file on local workstation

    :return: OMAS data set
    """
    printd('loading from %s on S3' % (_base_S3_uri(user) + filename), topic='s3')

    if not os.path.exists(os.path.abspath(tmp_dir)):
        os.makedirs(os.path.abspath(tmp_dir))
    remote_uri(_base_S3_uri(user) + filename, os.path.abspath(tmp_dir) + os.sep + os.sep + os.path.split(filename)[1],
               'down')
    return load_omas_pkl(os.path.abspath(tmp_dir) + os.sep + os.path.split(filename)[1])

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

def omas_scenario_database(machine=None, shot=None, run=None,
                           tmp_dir=omas_rcparams['fake_imas_dir'] + os.sep + 'scenarios', skip_existing=True):
    """
    List and retrieve available IMAS scenarios

    :param machine: string with the machine name of the scenario

    :param shot: shot number

    :param run: run number

    :param tmp_dir: temporary folder for storing S3 file on local workstation

    :param skip_existing: do not download S3 file if already present

    :return: OMAS data set with the requested scenario
    """
    if not machine and not shot and not run:
        remote_uri(_base_S3_uri('omas_shared') + 'scenario_summary.txt',
                   omas_rcparams['fake_imas_dir'] + os.sep + 'scenarios' + os.sep + 'scenario_summary.txt', 'down')
        return open('scenario_summary.txt', 'r').read()

    elif machine and shot and run:
        filename = os.path.abspath(tmp_dir) + os.sep + '{machine}_{shot}_{run}.pkl'.format(machine=machine, shot=shot,
                                                                                           run=run)
        if skip_existing and os.path.exists(filename):
            print('Loading scenario file from storage: %s' % filename)
            return load_omas_pkl(filename)
        else:
            print('Fetching scenario file: %s' % os.path.split(filename)[1])
            return load_omas_s3(os.path.split(filename)[1], user='omas_shared', tmp_dir=os.path.split(filename)[0])

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
