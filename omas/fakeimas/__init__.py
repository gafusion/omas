import os
import sys
from contextlib import contextmanager
from ..omas_core import ODS, list_structures, latest_imas_version, omas_rcparams

working_omas_imas_folder = omas_rcparams['fake_imas_dir']


class imasdef:
    MDSPLUS_BACKEND = 'MDSPLUS_BACKEND'
    HDF5_BACKEND = 'HDF5_BACKEND'
    MEMORY_BACKEND = 'MEMORY_BACKEND'
    UDA_BACKEND = 'UDA_BACKEND'
    ASCII_BACKEND = 'ASCII_BACKEND'


class IDS(ODS):
    def __getattr__(self, attr):
        return super().__getitem__(attr)

    def __setattr__(self, attr, value):
        if attr in ['parent', 'imas_version', 'cocos', 'cocosio', 'coordsio', 'unitsio', 'dynamic'] or attr.startswith('_'):
            return super().__setattr__(attr, value)
        else:
            return super().__setitem__(attr, value)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def deepcopy(self):
        return self.copy()


class DBEntry(dict):
    def __init__(self, backend, machine, pulse, run, user, imas_major_version):
        self.backend = backend
        self.machine = machine
        self.pulse = pulse
        self.run = run
        self.user = user
        self.imas_major_version = imas_major_version

    @property
    def filename(self):
        return (
            working_omas_imas_folder
            + os.sep
            + '_'.join(map(str, [self.backend, self.machine, self.pulse, self.run, self.user, self.imas_major_version]))
            + '.json'
        )

    def create(self):
        self.ods = ODS()

    def open(self):
        self.ods = ODS()
        self.ods.load(self.filename)
        return 0, 0

    def close(self):
        print(self.filename)
        self.ods.save(self.filename)
        return 0, 0

    def get(self, idsname, occurrence=0):
        return self.ods[idsname]

    def put(self, ids, occurrence=0):
        self.ods[ids.location] = ids


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
def fake_environment():
    injected = fake_module(True)
    print(injected)
    try:
        yield sys.modules['imas']
    finally:
        fake_module(False, injected)
