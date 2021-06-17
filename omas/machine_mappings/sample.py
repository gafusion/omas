import numpy as np
from inspect import unwrap
from omas import *
from omas.omas_utils import printd
from omas.machine_mappings._common import *

__all__ = []
__regression_arguments__ = {'__all__': __all__}


# Use the @machine_mapping_function to automatically fill out the .json mapping file
# All these functions must accept `ods` as their first argument
# Other arguments should have a default value defined via the machine_mapping_function decorator, as this serves two purposes:
#  1. run the test_machine_mapping_functions
#  2. let automatically fill the __options__ entry in the .json mapping file


@machine_mapping_function(__regression_arguments__, pulse=123456)
def sample_function(ods, pulse, user_argument='this is a test'):
    ods['dataset_description.ids_properties.comment'] = f'Comment for {pulse}: {user_argument}'


# =====================
if __name__ == '__main__':
    test_machine_mapping_functions(__all__, globals(), locals())
