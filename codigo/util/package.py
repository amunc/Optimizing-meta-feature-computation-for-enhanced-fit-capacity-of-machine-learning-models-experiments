#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:45:21 2021

@author: alumno
"""
import importlib
import importlib.util
import pkgutil
import os

def import_all_modules(module_spec):
    '''
    Importa todos los submodulos del paquete definido por module_spec

    Parameters
    ----------
    module_spec : _frozen_importlib.ModuleSpec
        Especificación del módulo base

    Raises
    ------
    TypeError
        Si module_spec no es un _frozen_importlib.ModuleSpec

    Returns
    -------
    None.

    '''
    if not isinstance(module_spec, importlib.machinery.ModuleSpec):
        raise TypeError("argument must be a module spec")
    module_list = sorted(
        pkgutil.iter_modules([os.path.split(module_spec.origin)[0]]),
        key = lambda module: module.ispkg
    )
    for module_info in module_list:
        full_module_path = "%s.%s" % (module_spec.name, module_info.name)
        importlib.import_module(full_module_path)