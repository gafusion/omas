import os
from omas import *

if not os.path.exists(os.sep.join([imas_json_dir,default_imas_version,'clean.xls'])):
    aggregate_imas_html_docs(default_imas_html_dir, default_imas_version)

create_json_structure(default_imas_version)#,['equilibrium'])

create_html_documentation(default_imas_version)