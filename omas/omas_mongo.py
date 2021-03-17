'''save/load from MongoDB routines

-------
'''

# to start a mongodb server on the local workstation
# mongod --dbpath $DIRECTORY_WHERE_TO_STORE_DATA

from .omas_utils import *
from .omas_core import ODS


# -----------------------------
# save and load OMAS to MongoDB
# -----------------------------
def save_omas_mongo(ods, collection, database='omas', server=omas_rcparams['default_mongo_server']):
    """
    Save an ODS to MongoDB

    :param ods: OMAS data set

    :param collection: collection name in the database

    :param database: database name on the server

    :param server: server name

    :return: unique `_id` identifier of the record
    """

    printd('Saving OMAS data to MongoDB: collection=%s database=%s  server=%s' % (collection, database, server), topic='MongoDB')

    # importing module
    from pymongo import MongoClient

    # connect
    client = MongoClient(server.format(**get_mongo_credentials(server, database, collection)))

    # access database
    db = client[database]

    # access collection
    coll = db[collection]

    # a cheap way to encode data
    kw = {'indent': 0, 'separators': (',', ': '), 'sort_keys': True}
    json_string = json.dumps(ods, default=lambda x: json_dumper(x, None), **kw)
    data = json.loads(json_string)

    # avoid insert_one() to modify data
    data = copy.copy(data)

    # insert record
    res = coll.insert_one(data)
    return str(res.inserted_id)


def load_omas_mongo(
    find,
    collection,
    database='omas',
    server=omas_rcparams['default_mongo_server'],
    consistency_check=True,
    imas_version=omas_rcparams['default_imas_version'],
    limit=None,
):
    """
    Load an ODS from MongoDB

    :param find: dictionary to find data in the database

    :param collection: collection name in the database

    :param database: database name on the server

    :param server: server name

    :param consistency_check: verify that data is consistent with IMAS schema

    :param imas_version: imas version to use for consistency check

    :param limit: return at most `limit` number of results

    :return: list of OMAS data set that match find criterion
    """

    # importing module
    from pymongo import MongoClient
    from bson.objectid import ObjectId

    # allow search by _id
    if not isinstance(find, dict):
        raise TypeError('load_omas_mongo find attribute must be a dictionary')
    if '_id' in find:
        find = copy.deepcopy(find)
        find['_id'] = ObjectId(find['_id'])

    printd('Loading OMAS data from MongoDB: collection=%s database=%s  server=%s' % (collection, database, server), topic='MongoDB')

    # connect
    client = MongoClient(server.format(**get_mongo_credentials(server, database, collection)))

    # access database
    db = client[database]

    # access collection
    coll = db[collection]

    # find all the matching records
    found = coll.find(find)
    if limit is not None:
        found = found.limit(limit)

    # populate ODSs
    results = {}
    for record in found:
        ods = ODS(consistency_check=consistency_check, imas_version=imas_version)
        _id = record['_id']
        del record['_id']
        ods.from_structure(record)
        results[_id] = ods

    return results


def get_mongo_credentials(server='', database='', collection=''):
    """
    Users can specify their credentials in a `~/.omas/mongo_credentials` json file
    formatted like this:

    >> {"default": {"user": "mydefaultuser",
    >>              "pass": "mydefaultpass"},
    >>              "omasdb-xymmt.mongodb.net": {"user": "myuser1"
    >>                                           "pass": "mypass1",
    >>                                           "specific_database": {"user": "myuser2",
    >>                                                                 "pass": "mypass2",
    >>                                                                 "specific_collection": {"user": "myuser3",
    >>                                                                                         "pass": "mypass3"}
    >> }}}

    if no `~/.omas/mongo_credentials` file is found: {'user': 'omas_test', 'pass': 'omas_test'}

    if no matching server is found, the `default` is returned

    :param server: server to look credentials for
        * server can have `{user}:{pass}` placeholders: mongodb+srv://{user}:{pass}@omasdb-xymmt.mongodb.net

    :param database: database name in server to look credentials for

    :param collection: collection name in database to look credentials for

    :return: dictionary with 'user' and 'pass' keys
    """
    server = server.split('@')[-1]
    up = {'user': 'omas_test', 'pass': 'omas_test'}
    config = {}
    filename = os.environ['HOME'] + '/.omas/mongo_credentials'
    if os.path.exists(filename):
        with open(filename) as f:
            config = json.loads(f.read())
    if 'default' in config:
        up = config['default']
    if server in config:
        up = config[server]
        if database in config[server]:
            up = config[server][database]
            if collection in config[server][collection]:
                up = config[server][database][collection]
    return up


def through_omas_mongo(ods, method=['function', 'class_method'][1]):
    """
    Test save and load OMAS MongoDB

    :param ods: ods

    :return: ods
    """
    ods = copy.deepcopy(ods)  # make a copy to make sure save does not alter entering ODS
    if method == 'function':
        _id = save_omas_mongo(ods, collection='test', database='test')
        results = load_omas_mongo({'_id': _id}, collection='test', database='test')
        if len(results) != 1:
            raise Exception('through_omas_mongo failed')
        ods1 = list(results.values())[0]
        return ods1
    else:
        _id = ods.save('mongo', collection='test', database='test')
        ods1 = ODS().load('mongo', {'_id': _id}, collection='test', database='test')
        return ods1
