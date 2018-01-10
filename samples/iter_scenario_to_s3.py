from __future__ import print_function, division, unicode_literals

import subprocess
import re
import os
from omas import *

force_write_existing = False

os.environ['OMAS_DEBUG_TOPIC'] = 's3'

# parse scenario_summary output
lines = subprocess.Popen('scenario_summary -c machine,shot,run,ref_name,workflow,idslist', stdout=subprocess.PIPE,
                         shell=True).communicate()[0].split('\n')
scenarios = []
ksep = 0
for line in lines:
    if line.startswith('----'):
        ksep += 1
    elif ksep == 2:
        items = line.strip().split()
        scenarios.append({'machine': items[0], 'shot': int(items[1]), 'run': int(items[2]), 'ref_name': items[3],
                          'workflow': items[4], 'idslist': items[5:]})

# setup environmental variables
tmp = subprocess.Popen('imasdb /work/imas/shared/iterdb/3 ; env | grep MDSPLUS_TREE_BASE', stdout=subprocess.PIPE,
                       shell=True).communicate()[0].split('\n')
for line in tmp:
    if 'MDSPLUS_TREE_BASE' in line:
        env, value = re.sub(';', '', re.sub('export', '', line)).strip().split('=')
        print(env, value)
        os.environ[env] = value

# find out existing scenarios
existing = list_omas_s3('omas_shared')

# loop over scenarios
for scenario in scenarios:
    # skip scenarios that have already been processed
    if not force_write_existing and 'omas_shared/{machine}_{shot}_{run}.pkl'.format(**scenario) in existing:
        print('Skip scenario: {machine} {shot} {run}'.format(**scenario))
        continue

    # fetch data
    print('Fetching scenario: {machine} {shot} {run}'.format(**scenario))
    complete_ods = load_omas_imas(user=None, shot=scenario['shot'], run=scenario['run'], paths=None)

    # save data as complete ods (locally and remotely) as well as individual odss (locally only)
    save_omas_s3(complete_ods, '{machine}_{shot}_{run}.pkl'.format(**scenario), user='omas_shared')
    for ids in complete_ods:
        if ids == 'info':
            continue
        ods = omas()
        ods[ids] = complete_ods[ids]
        ods['info'] = complete_ods['info']
        save_omas_pkl(ods, '{machine}_{shot}_{run}__{ids}.pkl'.format(ids=ids, **scenario))
