#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registro de los clasificadores
"""

from ..util.builder import ClassBuilder

    
_register = ClassBuilder()

register_classifier = _register.register
construct_classifier = _register.construct
query_classifier = _register.query
registered_classifiers = _register.registered
available_classifiers = _register.available
_register.set_name("classifier")

from ..util.package import import_all_modules
import_all_modules(__spec__)