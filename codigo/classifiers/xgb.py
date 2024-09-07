#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:31:18 2021

@author: alumno
"""

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter, CategoricalHyperparameter,
    UniformFloatHyperparameter, Constant
)
from . import register_classifier
from .classifier_base import Classifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

class XGB(xgb.XGBClassifier, Classifier):
    """
    tree-method influences accuracy: hist is worse than XGBClassifier's
    default, although it is faster
    """
    model_name = "XGB"
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, 
                 n_jobs=-1, tree_method="hist", subsample=1.0,
                 colsample_bytree=1.0):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                         n_jobs=n_jobs, subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         learning_rate=learning_rate, tree_method=tree_method)

    def fit(self, X, y, *args, **kwargs):
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        y = self.encoder.transform(y)
        super().fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        preds = super().predict(X, *args, **kwargs)
        return self.encoder.inverse_transform(preds)
    
    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=10
        )
        #max_depth = UniformIntegerHyperparameter(
        #    name="max_depth", lower=3, upper=6, default_value=3
        #)
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=10, upper=500, default_value=120
        )
        #n_estimators = CategoricalHyperparameter(
        #    "n_estimators", [150, 250, 500, 1000]    
        #)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.0001, upper=0.3, default_value=0.3
        )
        subsample = CategoricalHyperparameter(
            "subsample", [0.5 + 0.025 * x for x in range(21)]
        )
        colsample_bytree = CategoricalHyperparameter(
            "colsample_bytree", [0.5 + 0.025 * x for x in range(21)]
        )
        #learning_rate = Constant("learning_rate", 0.3)
        cs.add_hyperparameters([
            max_depth, n_estimators, learning_rate, colsample_bytree,
            subsample
        ])
        return cs

register_classifier(XGB.model_name, XGB)
