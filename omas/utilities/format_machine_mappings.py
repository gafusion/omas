'''
Utility to generate the omas/machine_mappings/*.json files
'''

import json
from omas.omas_machine import machines

for machine, filename in machines().items():
    print(machine, filename)
    with open(filename, 'r') as f:
        tmp = json.load(f)
    with open(filename, 'w') as f:
        json.dump(tmp, f, indent=1, separators=(',', ': '), sort_keys=True)
