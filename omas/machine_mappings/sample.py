from omas import *

__all__ = []


# Use the @machine_mapping_function to automatically fill out the .json mapping file
# All these functions must accept `ods` as their first argument
# All other arguments should have a default value defined, as this serves two purposes:
#  1. run the run_machine_mapping_functions
#  2. automatically fill the __options__ entry in the .json mapping file


@machine_mapping_function(__all__)
def sample_function(ods, pulse=123456, user_argument='this is a test'):
    ods['dataset_description.ids_properties.comment'] = f'Comment for {pulse}: {user_argument}'


# =====================
if __name__ == '__main__':
    run_machine_mapping_functions(__all__, globals(), locals())
