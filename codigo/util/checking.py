#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:37:52 2021

@author: alumno
"""
from collections.abc import Iterable, Mapping
import functools

def partial_wrapping(func, *args, **kwargs):
    return functools.update_wrapper(functools.partial(func, *args, **kwargs), func)

def check_file(file_path, mode="r"):
    '''
    Checks whether file_path exists

    Parameters
    ----------
    file_path : str
        Path to file
    mode : str, optional
        Type of access

    Returns
    -------
    None.

    '''
    with open(file_path, mode):
        pass

# Con el dispatch se logra que si se pasa una instancia se llame a check_type
# y si se pasa una clase se llame a check_subclass
@functools.singledispatch
def check_type(obj, klass, name=""):
    '''
    Comprueba que obj es una instancia de klass o de sus subclases

    Parameters
    ----------
    obj : Any
        El objeto a comprobar
    klass : type
        La clase
    name : str, optional
        Una string para prefijar al mensaje de error de la excepción

    Raises
    ------
    TypeError
        Si obj no es instancia de klass

    Returns
    -------
    None.
    '''
    if not isinstance(obj, klass):
        msg = "{} {!r} does not follow the {!r} interface"
        raise TypeError(msg.format(name, obj, klass))

def check_subclass(obj, klass, name=""):
    '''
    Comprueba que obj es una subclase de klass

    Parameters
    ----------
    obj : type
        La clase que se desea testear
    klass : type
        La clase ascendiente
    name : str, optional
        Una string para prefijar al mensaje de error de la excepción

    Raises
    ------
    TypeError
        Si obj no es subclase de klass

    Returns
    -------
    None.
    '''
    if not issubclass(obj, klass):
        msg = "{} {!r} does not follow the {!r} interface"
        raise TypeError(msg.format(name, obj, klass))

check_type.register(type, check_subclass)

def check_query(key, query_func, klass, name=""):
    """
    Function to access an item in the registers and check that it
    implements a particular interface
    """
    obj = key
    try:
        obj = query_func(key)
    finally:
        check_type(obj, klass, name=name)
        return obj
    
def check_query_strict(key, query_func, klass, name=""):
    '''
    Comprueba que hay un objeto identificado por key en el registro al que
    se consulta llamando a query_func y ese objeto soporta la interfaz definida
    por klass, si se cumple, devuelve el objeto.

    Parameters
    ----------
    key : any
        El identificador del objeto
    query_func : Callable
        Función que recibe como argumento ky y devuelve un objeto registrado
    klass : type
        La clase cuya interfaz debe cumplir el objeto devuelto por query_func
    name : str, optional
        Una string para prefijar al mensaje de error de la excepción

    Returns
    -------
    obj : Any
        El objeto devuelto por query_func 

    '''
    obj = query_func(key)
    check_type(obj, klass, name=name)
    return obj 

def base_check(arg, check_func):
    '''
    Aplica una función a un argumento. 
    
    Se usa principalmente en combinación con check_query para comprobar
    la integridad de los objetos registrados.

    Parameters
    ----------
    arg : Any
        El objeto que se pasa como argumento a check_func
    check_func : Callable
        La función a aplicar

    Returns
    -------
    Any
        El valor de retorno de check_func

    '''
    return check_func(arg)

# Con el dispatch se logra que se pueda aplicar la función a objetos 
# individuales, listas o diccionarios más fácilmente
check_arg = functools.singledispatch(base_check)

def _check_array(array_arg, check_func):
    return [check_func(arg) for arg in array_arg]

check_arg.register(Iterable, _check_array)

def _check_mapping(mapping_arg, check_func):
    return {k: check_func(mapping_arg[k]) for k in mapping_arg}

check_arg.register(Mapping, _check_mapping)

def _check_iterable_to_dict(iterable_arg, check_func):
    return {k: check_func(k) for k in iterable_arg}

check_arg_metric = functools.singledispatch(base_check)

check_arg_metric.register(Iterable, _check_iterable_to_dict)
check_arg_metric.register(Mapping, _check_mapping)