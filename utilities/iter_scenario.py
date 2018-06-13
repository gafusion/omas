from __future__ import print_function, division, unicode_literals

import subprocess
import re
import os
from omas import *

process_existing = False
save_local = False
s3_username = os.environ['USER']

os.environ['OMAS_DEBUG_TOPIC'] = 's3'

# parse scenario_summary output
what=['machine', 'shot','run','ref_name','ip','b0','fuelling','confinement','workflow']
scenario_summary = subprocess.Popen('scenario_summary -c '+','.join(what), stdout=subprocess.PIPE,
                         shell=True).communicate()[0].split('\n')
print('\n'.join(scenario_summary))
scenarios = []
ksep = 0
for line in scenario_summary:
    if line.startswith('----'):
        ksep += 1
    elif ksep == 2:
        items = line.strip().split()
        scenarios.append(dict(zip(what, items)))

# setup environmental variables
tmp = subprocess.Popen('imasdb /work/imas/shared/iterdb/3 ; env | grep MDSPLUS_TREE_BASE', stdout=subprocess.PIPE,
                       shell=True).communicate()[0].split('\n')
for line in tmp:
    if 'MDSPLUS_TREE_BASE' in line:
        env, value = re.sub(';', '', re.sub('export', '', line)).strip().split('=')
        os.environ[env] = value

# find out existing scenarios
if save_local:
    existing = glob.glob('*.pkl')
else:
    existing = map(lambda x=x.split(x)[1], list_omas_s3(s3_username))

# loop over scenarios
for scenario in scenarios:
    # skip scenarios that have already been processed
    if not process_existing and '{machine}_{shot}_{run}.pkl'.format(**scenario) in existing:
        print('Skip scenario: {machine} {shot} {run}'.format(**scenario))
        continue

    # fetch data
    print('Fetching scenario: {machine} {shot} {run}'.format(**scenario))
    complete_ods = load_omas_imas(user=None, shot=int(scenario['shot']), run=int(scenario['run']), paths=None)

    # save data as complete ods (locally and remotely) as well as individual odss (locally only)
    if save_local:
        save_omas_pkl(complete_ods, '{machine}_{shot}_{run}.pkl'.format(ids=ids, **scenario))
    else:
        save_omas_s3(complete_ods, '{machine}_{shot}_{run}.pkl'.format(**scenario), user=s3_username)

# upload scenario_summary
if not save_local:
    open('scenario_summary.txt', 'w').write('\n'.join(scenario_summary))
    omas_s3.remote_uri(omas_s3._base_S3_uri(s3_username), 'scenario_summary.txt', 'up')
