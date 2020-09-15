'''
Quickly check that python packages required for proper running of OMAS are all installed
'''
import os
from pprint import pprint
import sys
import re
import warnings

warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=FutureWarning)

mapper = {'dnspython': None, 'sphinx-bootstrap-theme': None, 'sphinx-gallery': None, 'Sphinx': 'sphinx', 'Pillow': 'PIL'}

filename = os.path.split(os.path.abspath(__file__))[0] + os.sep + 'requirements.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
lines = [
    list(map(lambda x: re.sub('([\w-]*).*', r'\1', x.strip()), line.strip().split('#')))
    for line in lines
    if line.strip() and not line.strip().startswith('#')
]
lines = [
    (mapper.get(package, package), required)
    for package, required in lines
    if mapper.get(package, package is not None) and required == 'required'
]

good = []
for k, (package, required) in enumerate(lines):
    try:
        exec('import %s' % package) in globals()
        print('  OK  %03d/%03d : %s' % (k + 1, len(lines), package))
        good.append(True)
    except ImportError as _excp:
        print('ERROR %03d/%03d : %s' % (k + 1, len(lines), 'Package %s failed to import! %s' % (package, repr(_excp))), file=sys.stderr)
        good.append(False)
if not all(good):
    print('ERROR!!! Some python packages failed to import. Not all OMAS functionalites will be available :(', file=sys.stderr)
