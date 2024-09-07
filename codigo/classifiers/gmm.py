#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:31:18 2021

@author: alumno
"""

from sklearn.mixture import GaussianMixture
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
)
from . import register_classifier
from .implemented.clustering_classifier import BaseClusteringPredict

class GMM(BaseClusteringPredict):
    model_name = "GMM"
    model = GaussianMixture
    
    def __init__(self, n_components=20, max_iter=20000, reg_covar=1e-5, 
                 n_init = 1, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.params = {"n_components": self.n_components, 
                       "random_state": self.random_state,
                       "init_params": "kmeans",
                       "covariance_type": "full",
                       "max_iter": self.max_iter,
                       "reg_covar": reg_covar,
                       "n_init": n_init
                       }
        
    @staticmethod
    def get_configuration_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=4, upper=400, default_value=20
        )

        max_iter = Constant("max_iter", 20000)
        n_init = Constant("n_init", 3)
        cs.add_hyperparameters([
            n_components, max_iter, n_init
        ])
        return cs
    
    def get_number_of_clusters(self):
        return self.n_components


register_classifier(GMM.model_name, GMM)
