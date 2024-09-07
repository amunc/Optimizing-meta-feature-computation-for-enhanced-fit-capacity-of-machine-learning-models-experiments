#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:40:42 2021

@author: alumno
"""
import functools
from ..util.checking import partial_wrapping, check_arg, check_query_strict, check_query
from .classifier_base import Classifier
from . import query_classifier

check_classifier = partial_wrapping(
    #check_query_strict, 
    check_query, query_func=query_classifier, klass=Classifier, 
    name="classifier"    
)

check_classifiers = functools.partial(check_arg, check_func=check_classifier)
