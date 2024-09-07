#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:59:05 2021

@author: alumno
"""

import abc
from typing import Hashable

class Classifier(abc.ABC):
    '''
    Clase base para los predictores
    '''
    model_name = "base_classifier"
    @classmethod
    def __subclasshook__(cls, candidate):
        if cls is Classifier:
            if (not hasattr(candidate, "model_name") 
            or not isinstance(getattr(candidate, "model_name"), Hashable)):
                return False
            sought_attrs = ["fit", "predict", "predict_proba"]
            for sought_attr in sought_attrs:
                if not hasattr(candidate, sought_attr):
                    return False
            return True
        return NotImplemented
    
    
    @abc.abstractmethod
    def fit(self, X, y):
        '''
        Entrena el predictor
        
        Parameters
        ----------
        X: numpy.ndarray shape(n, k)
            Datos de entrenamiento
        y: numpy.ndarray shape(n,)
            Etiquetas de datos de entrenamiento
        '''
        return self
    
    @abc.abstractmethod
    def predict(self, X):
        '''
        Predice sobre los datos pasados como argumento

        Parameters
        ----------
        X : numpy.ndarray shape(m, k)
            Datos de test sobre los que predecir

        Returns
        -------
        numpy.ndarray shape(m,)
            Etiquetas asignadas por el predictor a los datos de X
        '''
        pass
    
    @abc.abstractmethod
    def predict_proba(self, X):
        '''
        Predice la probabilidad de que cada elemento de X
        pertenezca a una clase concreta
        
        Parameters
        ----------
        X : numpy.ndarray shape(m, k)
            Datos de test sobre los que predecir

        Returns
        -------
        numpy.ndarray shape(m, number of unique labels passed in y in fit)
            Probabilidad para cada elemento de X de pertenecer a cada clase
        '''
        pass