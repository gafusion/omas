from omfit.omfit_tree import *
import glob
import os

txt = []
txt.append('Classes')
txt.append('-' * len(txt[-1]))
for d in dir():
    if hasattr(eval(d), 'to_omas') or hasattr(eval(d), 'from_omas'):
        txt.append(f'* {d}\n')
        c = eval(d)
        for a in ['to_omas', 'from_omas']:
            if hasattr(c, a):
                txt.append(f"  *  `{d}.{a}() <https://omfit.io/code.html#{c.__module__}.{d}.{a}>`_")
        txt.append('\n')

txt.append('Modules')
txt.append('-' * len(txt[-1]))
for filename in glob.glob(OMFITsrc + '/../modules/*/OMFITsave.txt'):
    module = os.path.split(os.path.split(filename)[0])[1]
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
    tmp = []
    for line in lines:
        if 'to_omas' in line or 'from_omas' in line:
            tmp.append(line.split(' <-:-:-> '))
    if len(tmp):
        txt.append(f'* {module}\n')
        for k in range(len(tmp)):
            txt.append(
                f'  *  `{tmp[k][0]} <https://github.com/gafusion/OMFIT-source/tree/unstable/modules/{module}/{tmp[k][2].lstrip("./")}>`_'
            )
        txt.append('\n')

print('\n'.join(txt))

from omas import omas_dir

with open(omas_dir + '/../sphinx/source/omfit_to_from_omas.rst', 'w') as f:
    f.write('\n'.join(txt))
