import os, sys, re

omas_dir = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
sys.path.insert(0, omas_dir)

with open(omas_dir + os.sep + 'omas' + os.sep + 'omas_symbols.py', 'r') as f:
    lines = f.read().split('\n')

groups = {'symbols': [], 'units': []}
active = None
for action in ['gather', 'apply']:
    for group in groups:
        groups[group].sort()
        for k, line in enumerate(lines):
            if re.match(f'# {group} start', line):
                active = group
            elif re.match(f'# {group} end', line):
                active = None
            elif active:
                if action == 'gather':
                    groups[active].append(line)
                else:
                    lines[k] = groups[active].pop(0)

print('\n'.join(lines))

with open(omas_dir + os.sep + 'omas' + os.sep + 'omas_symbols.py', 'w') as f:
    f.write('\n'.join(lines))
