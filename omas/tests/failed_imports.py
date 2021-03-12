from omas.omas_setup import omas_rcparams
import os
import warnings
from omas.tests.warning_setup import hard_warnings, set_omas_warnings
import socket

if 'iter' in socket.gethostname():
    not_running_on_iter_cluster = False
else:
    not_running_on_iter_cluster = 'Not running on ITER cluster'

if 'partnaires.cea.rf' in socket.gethostname():
    not_running_on_cea_cluster = False
else:
    not_running_on_cea_cluster = 'Not running on CEA cluster'


try:
    import imas

    failed_IMAS = False
except ImportError as _excp:
    failed_IMAS = _excp

try:
    import pyuda

    failed_UDA = False
except ImportError as _excp:
    failed_UDA = _excp

try:
    import hdc

    failed_HDC = False
except ImportError as _excp:
    failed_HDC = _excp

try:
    from botocore.exceptions import NoCredentialsError
    import boto3

    if not os.path.exists(os.environ.get('AWS_CONFIG_FILE', os.environ['HOME'] + '/.aws/config')):
        raise RuntimeError('Missing AWS configuration file ~/.aws/config')
    failed_S3 = False
except (ImportError, RuntimeError, NoCredentialsError) as _excp:
    failed_S3 = _excp

try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError
    from omas.omas_mongo import get_mongo_credentials

    up = get_mongo_credentials(server=omas_rcparams['default_mongo_server'])
    client = MongoClient(omas_rcparams['default_mongo_server'].format(**up), serverSelectionTimeoutMS=1000)
    client.server_info()
    failed_MONGO = False
except (ImportError, ServerSelectionTimeoutError) as _excp:
    failed_MONGO = _excp

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from omfit_classes.omfit_eqdsk import OMFITgeqdsk, OMFITsrc

        failed_OMFIT = False
    except ImportError as _excp:
        failed_OMFIT = _excp

try:
    import MDSplus

    failed_MDS = False

    try:
        conn = MDSplus.Connection('atlas.gat.com:8000')
        failed_D3D_MDS = False
    except Exception as _excp:
        failed_D3D_MDS = _excp

except ImportError as _excp:
    failed_MDS = _excp
    failed_D3D_MDS = _excp

__all__ = [
    'failed_IMAS',
    'failed_HDC',
    'failed_S3',
    'failed_MONGO',
    'failed_OMFIT',
    'failed_UDA',
    'failed_MDS',
    'failed_D3D_MDS',
    'not_running_on_iter_cluster',
    'not_running_on_cea_cluster',
]
