import sys
if sys.version_info.major==2:
    from test_omas_core import *
    from test_omas_physics import *
    from test_omas_plot import *
    from test_omas_suite import *
    from test_omas_utils import *
else:
    from .test_omas_core import *
    from .test_omas_physics import *
    from .test_omas_plot import *
    from .test_omas_suite import *
    from .test_omas_utils import *
