import sys

from src.tasks.implementations import *
from src.tasks.ssl import *


def get_task_with_name(name):
    """
    Checks the provided `name` is within the module definitions, raises `NameError` if it isn't
       :returns: an object referencing the desired task (specified in input)
    """
    try:
        identifier = getattr(sys.modules[__name__],
                             name)  # get a reference to the module itself through the sys.modules dictionary
    except AttributeError:
        raise NameError(f"{name} is not a valid task.")
    if isinstance(identifier, type):
        return identifier
    raise TypeError(f"{name} is not a valid task.")
