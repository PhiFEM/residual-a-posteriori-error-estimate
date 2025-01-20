from   collections.abc import Callable
from   copy import deepcopy
import functools
import inspect
import numpy as np
import os

debug_env = os.getenv("DEBUG")
if debug_env=="True":
    debug = True
else:
    debug = False

def immutable(*arg_names: str) -> Callable:
    """ Decorator that check if the arguments specified have been modified (only if the DEBUG environment variable is set to True).

    Args:
        arg_names: the names of the wanted parameters to be immutable.
    """ 
    def decorator_mutable(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            if debug:
                # Get func's signature (needed because pytest messes up signatures)
                sig = inspect.signature(func)
                func_args = sig.bind(*args, **kwargs)
                func_args.apply_defaults()

                arg_copies = []
                for arg_name in arg_names:
                    arg_copies.append(deepcopy(func_args.arguments[arg_name]))
            
            result = func(*args, **kwargs)

            if debug:
                for i, arg_name in enumerate(arg_names):
                    if isinstance(func_args.arguments[arg_name], np.ndarray):
                        if not np.array_equal(arg_copies[i], func_args.arguments[arg_name]):
                            raise ValueError(f"Immutable parameter {arg_name} has been modified.")
                    else:
                        if arg_copies[i] != func_args.arguments[arg_name]:
                            raise ValueError(f"Immutable parameter {arg_name} has been modified.")
    
            return result
        return decorated
    return decorator_mutable