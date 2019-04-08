import os
from omas import *
from pprint import pprint

server='idam0.mast.ccfe.ac.uk'
port=56563

tmp = load_omas_uda(server=server, port=port, pulse=30420, run=0, paths=None,#paths=['magnetics'],
                  imas_version=os.environ.get('IMAS_VERSION', omas_rcparams['default_imas_version']),
                  verbose=True)

pprint(tmp)
