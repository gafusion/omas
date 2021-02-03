'''save/load from S3 routines

-------
'''

from .omas_utils import *
from .omas_core import save_omas_pkl, load_omas_pkl, ODS


def _base_S3_uri(user):
    return 's3://omas3/{user}/'.format(user=user)


# --------------------------------------------
# save and load OMAS with S3
# --------------------------------------------
def remote_uri(uri, filename, action):
    """
    :param uri: uri of the container of the file

    :param filename: filename to act on

    :param action: must be one of [`up`, `down`, `list`, `del`]
    """
    if not re.match('\w+://\w+.*', uri):
        return uri

    tmp = uri.split('://')
    system = tmp[0]
    location = '://'.join(tmp[1:])

    if action not in ['down', 'up', 'list', 'del']:
        raise AttributeError('remote_uri action attribute must be one of [`up`, `down`, `list`, `del`]')

    if system == 's3':
        import boto3
        from boto3.s3.transfer import TransferConfig

        s3bucket = location.split('/')[0]
        s3connection = boto3.resource('s3')
        s3filename = '/'.join(location.split('/')[1:])

        if action == 'list':
            printd('Listing %s' % (uri), topic='s3')
            files = list(map(lambda x: x.key, s3connection.Bucket(s3bucket).objects.all()))
            s3filename = s3filename.strip('/')
            if s3filename:
                files = filter(lambda x: x.startswith(s3filename), files)
            return files

        if action == 'del':
            if filename is None:
                filename = s3filename.split('/')[-1]
            printd('Deleting %s' % uri, topic='s3')
            s3connection.Object(s3bucket, s3filename).delete()

        elif action == 'down':
            if filename is None:
                filename = s3filename.split('/')[-1]
            printd('Downloading %s to %s' % (uri, filename), topic='s3')
            obj = s3connection.Object(s3bucket, s3filename)
            if not os.path.exists(os.path.abspath(os.path.split(filename)[0])):
                os.makedirs(os.path.abspath(os.path.split(filename)[0]))
            obj.download_file(filename, Config=TransferConfig(use_threads=False))

        elif action == 'up':
            printd('Uploading %s to %s' % (filename, uri), topic='s3')
            from botocore.exceptions import ClientError

            if s3filename.endswith('/'):
                s3filename += filename.split('/')[-1]
            try:
                s3connection.meta.client.head_bucket(Bucket=s3bucket)
            except ClientError as _excp:
                # If a client error is thrown, then check that it was a 404 error.
                # If it was a 404 error, then the bucket does not exist.
                error_code = int(_excp.response['Error']['Code'])
                if error_code == 404:
                    s3connection.create_bucket(Bucket=s3bucket)
                else:
                    raise
            bucket = s3connection.Bucket(s3bucket)
            with open(filename, 'rb') as data:
                bucket.put_object(Key=s3filename, Body=data)  # , Metadata=meta)


def save_omas_s3(ods, filename, user=os.environ.get('USER', 'dummy_user'), tmp_dir=omas_rcparams['tmp_omas_dir'], **kw):
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


def load_omas_s3(
    filename, user=os.environ.get('USER', 'dummy_user'), consistency_check=None, imas_version=None, tmp_dir=omas_rcparams['tmp_omas_dir']
):
    """
    Download an OMAS object from S3 and read it as pickle

    :param filename: filename to load from

    :param user: username where to look for the file

    :param consistency_check: verify that data is consistent with IMAS schema (skip if None)

    :param imas_version: imas version to use for consistency check (leave original if None)

    :param tmp_dir: temporary folder for storing S3 file on local workstation

    :return: OMAS data set
    """
    printd('loading from %s on S3' % (_base_S3_uri(user) + filename), topic='s3')

    if not os.path.exists(os.path.abspath(tmp_dir)):
        os.makedirs(os.path.abspath(tmp_dir))
    remote_uri(_base_S3_uri(user) + filename, os.path.abspath(tmp_dir) + os.sep + os.sep + os.path.split(filename)[1], 'down')
    return load_omas_pkl(
        os.path.abspath(tmp_dir) + os.sep + os.path.split(filename)[1], consistency_check=consistency_check, imas_version=imas_version
    )


def list_omas_s3(user=''):
    """
    List S3 content

    :param user: username where to look for the file

    :return: OMAS data set
    """
    return remote_uri(_base_S3_uri(user), None, 'list')


def del_omas_s3(filename, user=os.environ.get('USER', 'dummy_user')):
    """
    Delete an OMAS object from S3

    :param user: username where to look for the file

    :return: OMAS data set
    """
    remote_uri(_base_S3_uri(user) + filename, None, 'del')


def through_omas_s3(ods, method=['function', 'class_method'][1]):
    """
    Test save and load S3

    :param ods: ods

    :return: ods
    """
    filename = 'test.pkl'
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
    if method == 'function':
        save_omas_s3(ods, filename, user='omas_test')
        ods1 = load_omas_s3(filename, user='omas_test')
    else:
        ods.save('s3', filename=filename, user='omas_test')
        ods1 = ODS().load('s3', filename=filename, user='omas_test')
    return ods1
