import os
import sys
from contextlib import contextmanager
from .omas_core import baseODS, list_structures, latest_imas_version, omas_rcparams
from .omas_utils import p2l, o2i, l2u, _extra_structures

def empty(data_type):
    if 'STR' in data_type:
        value = ''
    elif 'INT' in data_type:
        value = -999999999
    elif 'FLT' in data_type:
        value = -9e40
    return value


class imasdef:
    MDSPLUS_BACKEND = 'MDSPLUS_BACKEND'
    HDF5_BACKEND = 'HDF5_BACKEND'
    MEMORY_BACKEND = 'MEMORY_BACKEND'
    UDA_BACKEND = 'UDA_BACKEND'
    ASCII_BACKEND = 'ASCII_BACKEND'


class IDS(baseODS):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __getattr__(self, attr):
        location = p2l(self.location + '.' + attr)
        if o2i(l2u(location)) in _extra_structures.get(location[0], {}):
            raise AttributeError(f'{attr} is not part of IMAS')
        try:
            return super().__getitem__(attr)
        except ValueError:
            # return empty IMAS value
            return empty(self.info(attr)['data_type'])

    def __setattr__(self, attr, value):
        if attr in ['parent', 'imas_version', 'cocos', 'cocosio', 'coordsio', 'unitsio', 'dynamic', 'consistency_check'] or attr.startswith(
            '_'
        ):
            return super().__setattr__(attr, value)
        else:
            return super().__setitem__(attr, value)

    def __getitem__(self, key, cocos_and_coords=True):
        return super().__getitem__(key, cocos_and_coords)

    def deepcopy(self):
        return self.copy()

    def put(self, occurrence, DB):
        return DB.put(self, occurrence)

    def get(self, occurrence, DB):
        self._omas_data = DB.get(self._toplocation, occurrence)._omas_data
        for item in self._omas_data:
            if isinstance(self._omas_data[item], baseODS):
                self._omas_data[item].parent = self


class DBEntry(dict):
    def __init__(self, backend, machine, pulse, run, user, imas_major_version):
        self.backend = backend
        self.machine = machine
        self.pulse = pulse
        self.run = run
        self.user = user
        self.imas_major_version = imas_major_version
        self.mode = None

    @property
    def filename(self):
        return (
            omas_rcparams['fakeimas_dir']
            + os.sep
            + '_'.join(map(str, [self.backend, self.machine, self.pulse, self.run, self.user, self.imas_major_version]))
            + '.json'
        )

    def create(self):
        self.ids = IDS()
        self.mode = 'create'
        return 0, 0

    def open(self):
        if not os.path.exists(self.filename):
            raise IOError(f'{self.filename} does not exist')
        print(self.filename)
        self.ids = IDS()
        self.ids.load(self.filename, cls=IDS)
        self.mode = 'open'
        return 0, 0

    def close(self):
        if self.mode == 'create':
            print(self.filename)
            if not os.path.exists(os.path.split(self.filename)[0]):
                os.makedirs(os.path.split(self.filename)[0])
            self.ids.save(self.filename)
        self.mode = None
        return 0, 0

    def get(self, idsname, occurrence=0):
        return self.ids[idsname]

    def put(self, ids, occurrence=0):
        self.ids[ids.location] = ids


for ds in list_structures(latest_imas_version):
    exec(
        f'''
def {ds}():
    ids = IDS()
    ids._toplocation = {ds!r}
    return ids
'''
    )


def fake_module(enabled, injected=None):
    '''
    :param enabled: True/False/'fallback'
                    'fallback' is used to turn on the OMAS fake IMAS module if the original IMAS is not installed
    '''
    if enabled == 'fallback':
        try:
            import imas
        except ImportError:
            enabled = True
    if enabled:
        injected = False
        if 'imas' in sys.modules and 'imas_orig_' not in sys.modules:
            sys.modules['imas_orig_'] = sys.modules['imas']
            del sys.modules['imas']
        if 'imas' not in sys.modules:
            sys.modules['imas'] = sys.modules['omas.fakeimas']
            injected = True
        return injected
    else:
        if injected is not False:
            del sys.modules['imas']
            if 'imas_orig_' in sys.modules:
                sys.modules['imas'] = sys.modules['imas_orig_']
                del sys.modules['imas_orig_']


@contextmanager
def fake_environment(enabled=True):
    '''
    :param enabled: True/False/'fallback'
    '''
    injected = fake_module(enabled)
    try:
        yield sys.modules['imas']
    finally:
        fake_module(False, injected)
