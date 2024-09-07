import collections.abc
import abc
import numpy as np
import typing
from typing import Any
import os


def is_iterable(obj: Any, exclude_tuples: bool = False,
                exclude_sets: bool = False) -> bool:
    '''
    Checks whether and object is iterable. This boils down to checking
    whether the object defines an __iter__ method and is not a string or bytes.

    string and bytes are iterables, but usually you want to exclude them

    Arguments
    ---------
    obj: object
        The object to test
    exclude_tuples: Bool
        If True, tuples will not be considered iterables
    exclude_sets: Bool
        If True, sets will not be considered iterables

    Returns
    -------
    True or False
    '''
    return (
        isinstance(obj, collections.abc.Iterable)
        and not isinstance(obj, (str, bytes))
        and not (exclude_tuples and isinstance(tuple, obj))
        and not (exclude_sets and isinstance(set, obj))
    )

def is_callable(obj):
    '''
    Returns True is ibj is Callable

    Parameters
    ----------
    obj : Any
        The object to check
        
    Returns
    -------
    bool
        Whether obj is Callable
    '''
    return isinstance(obj, typing.Callable)

class Pathlike(abc.ABC):
    @classmethod
    def __subclasshook__(cls, candidate):
        if cls is Pathlike:
            return issubclass(candidate, (str, os.PathLike))
        return NotImplemented

def is_list(obj: Any) -> bool:
    '''
    Checks whether the object is a list (or a numpy array)

    Arguments
    ---------
    obj: object
        The object to check

    Returns
    -------
    True or False
    '''
    return isinstance(obj, (list, np.ndarray))
