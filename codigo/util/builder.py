#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 18:35:42 2021

@author: alumno
"""
from collections import UserDict
from .exceptions import NotRegisteredError

class ClassBuilder(UserDict):
    '''
    Clase usada para registrar los objetos necesarios para
    el funcionamiento del script como los algoritmos,
    extractores de features, etc.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = ""
        
    def register(self, key, klass, name=""):
        self[key] = klass
        
    def set_name(self, name):
        self.name = name
        
    def construct(self, key, *args, **kwargs):
        return self[key](*args, **kwargs)
    
    def __getitem__(self, key):
        if key not in self:
            msg = "{} {!r} is not registered"
            raise NotRegisteredError(msg.format(self.name, key))
        return super().__getitem__(key)
    
    def query(self, key):
        return self[key]
    
    def available(self):
        return list(self)
    
    def registered(self):
        return dict(self)

class DefaultClassBuilder(ClassBuilder):
    '''
    Subclase de ClassBuilder que incluye la posibilidad de fijar
    un valor por defecto para todas las consultas
    '''
    def __init__(self, default=None, *args, **kwargs):
        ClassBuilder.__init__(self, *args, **kwargs)
        self.default = default
        
    def set_default(self, default):
        self.default = default
        
    def __getitem__(self, key):
        try:
            return ClassBuilder.__getitem__(self, key)
        except NotRegisteredError:
            return self.default