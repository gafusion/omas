import omas
from matplotlib.pyplot import *
import copy

tmp = {}
for version in omas.imas_versions:
    print(version)
    if version.startswith("3"):
        ods = omas.omas_info(imas_version=version)
        for ds in ods:
            for key in ods[ds].flat():
                if 'lifecycle_status' in ods[ds][key]:
                    if ods[ds][key]['lifecycle_status'] == 'alpha':
                        tmp.setdefault(ds + '.' + key, []).append(version)

alp = copy.deepcopy(tmp)
bla = dict()
for key in list(alp.keys()):
    if '3.36.0' in alp[key] and len(alp[key]):
        bla.setdefault(alp[key][0], 0)
        bla[alp[key][0]] += 1

bar(list(bla.keys()), list(bla.values()))
xticks(rotation=90)
title('Nodes still in alpha version since IMAS version:')
tight_layout()
show()
