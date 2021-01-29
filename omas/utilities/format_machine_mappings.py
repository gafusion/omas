'''
Utility to generate the omas/machine_mappings/*.json files
'''

import os
import re
import json
from pprint import pprint

from omas import ODS, machines, machine_mappings

for machine, filename in machines().items():
    print()
    print('=' * (len(machine) + len(filename) + 2))
    print(f'{machine}: {filename}')
    print('=' * (len(machine) + len(filename) + 2))
    with open(filename, 'r') as f:
        tmp = json.load(f)
    with open(filename, 'w') as f:
        json.dump(tmp, f, indent=1, separators=(',', ': '), sort_keys=True)

    mappings = machine_mappings(machine, '')

    if os.path.exists(os.path.splitext(filename)[0] + '.py'):
        namespace = {}
        with open(os.path.splitext(filename)[0] + '.py', 'r') as f:
            exec(f.read(), namespace)
        for function in namespace['__all__']:
            print('-' * (len(machine) + len(function) + 2))
            print(f'{machine}: {function}')
            print('-' * (len(machine) + len(function) + 2))

            mapped = False
            for item in mappings:
                if 'PYTHON' in mappings[item] and re.match('\\b' + function + '\\b', mappings[item]['PYTHON']):
                    print('  ' + item)
                    mapped = True
