from .omas_core import *

import os as _os
print(_os.environ.get('OMAS_DEBUG_TOPIC', ''))
if _os.environ.get('OMAS_DEBUG_TOPIC', '').endswith('_dump'):
    if _os.path.exists('omas_dump.txt'):
        _os.remove('omas_dump.txt')
