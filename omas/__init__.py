import socket
import os
import sys

# Unfortunately ITER OMAS module is not kept up-to-date.
# Here we detect if this OMAS version is running at ITER
# and is the one would get by doing a `module load OMAS`
# If so we import OMAS from where it is constantly kept
# up to date by OMAS developers themeselves
if (
    os.path.exists('/home/ITER/menghio/atom/omas')
    and '.iter.org' in socket.gethostname()
    and '/work/imas/opt/EasyBuild/software/OMAS' in os.path.abspath(__file__)
):
    sys.path.insert(0, '/home/ITER/menghio/atom/omas')
    sys.path.append('/home/ITER/menghio/atom/omas/site-packages')
    del sys.modules['omas']
    from omas.omas_core import *
else:
    from .omas_core import *

__all__ = [str(_item) for _item in locals().keys() if not (_item.startswith('__') and _item.endswith('__'))]
