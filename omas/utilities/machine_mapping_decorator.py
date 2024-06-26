from omas.omas_setup import omas_git_repo
import os
import functools
import numpy
from omas.omas_core import o2u
__all__ = [
    'machine_mapping_function'
]

    
# ===================
# machine mapping functions
# ===================
def machine_mapping_function(__regression_arguments__, **regression_args):
    """
    Decorator used to identify machine mapping functions

    :param \**regression_args: arguments used to run regression test

    NOTE: use `inspect.unwrap(function)` to call a function decorated with `@machine_mapping_function`
          from another function decorated with `@machine_mapping_function`
    """

    __all__ = __regression_arguments__['__all__']

    def machine_mapping_decorator(f, __all__):
        __all__.append(f.__name__)
        if __regression_arguments__ is not None:
            __regression_arguments__[f.__name__] = regression_args

        @functools.wraps(f)
        def machine_mapping_caller(*args, **kwargs):
            clean_ods = True
            if len(args[0]):
                clean_ods = False
            if clean_ods and omas_git_repo:
                import inspect

                # figure out the machine name from where the function `f` is defined
                machine = os.path.splitext(os.path.split(inspect.getfile(f))[1])[0]
                if machine == '<string>':  # if `f` is called via exec then we need to look at the call stack to figure out the machine name
                    machine = os.path.splitext(os.path.split(inspect.getframeinfo(inspect.currentframe().f_back)[0])[1])[0]

                # call signature
                argspec = inspect.getfullargspec(f)
                f_args_str = ", ".join('{%s!r}' % item for item in argspec.args if not item.startswith('_'))
                # f_args_str = ", ".join(item + '={%s!r}' % item for item in argspec.args if not item.startswith('_')) # to use keywords arguments
                call = f"{f.__qualname__}({f_args_str})".replace('{ods!r}', 'ods').replace('{pulse!r}', '{pulse}')
                default_options = None
                if argspec.defaults:
                    default_options = dict(zip(argspec.args[::-1], argspec.defaults[::-1]))
                    default_options = {item: value for item, value in default_options.items() if not item.startswith('_')}

            # call
            update_mapping = None
            if "update_callback" in kwargs:
                update_mapping = kwargs.pop("update_callback")
            out = f(*args, **kwargs)
            #update mappings definitions
            if not update_mapping is None:
                if clean_ods and omas_git_repo:
                    for ulocation in numpy.unique(list(map(o2u, args[0].flat().keys()))):
                        update_mapping(machine, ulocation, {'PYTHON': call}, 11, default_options, update_path=True)

            return out

        return machine_mapping_caller

    return lambda f: machine_mapping_decorator(f, __all__)
