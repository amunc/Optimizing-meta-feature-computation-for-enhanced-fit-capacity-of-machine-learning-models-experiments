# -*- coding: utf-8 -*-
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from . import register_classifier
from .classifier_base import Classifier
import functools
import copy

excluded = {"classifier": ["bernoulli_nb", "multinomial_nb", "gaussian_nb",
                          "liblinear_svc", "libsvm_svc", 
                          # Might want to delete knn since consumes
                          # a bit of memory and am going to run in parallel
                          #"k_nearest_neighbors"
                          ]}    


class Autosklearn(SimpleClassificationPipeline, Classifier):
    model_name = "autosklearn"
    params = SimpleClassificationPipeline(
        #exclude = excluded  
    ).get_hyperparameter_search_space(
             dataset_properties={"signed": True}
        )
    def __init__(self, **kwargs):
        self.config = kwargs or None
        super().__init__(config=kwargs)

    @classmethod
    @functools.lru_cache
    # cache necessary in order to ensure that composite_autosklearn works
    # otherwise the search space gets updated after adding to autosklearn
    # alternatively, set it as class parameter and this method as classmethod
    # in order to have a single static copy
    def get_configuration_space(cls):
        return copy.deepcopy(cls.params)
    
    def get_params(self, deep=True):
        return self.config or {}
    
register_classifier(Autosklearn.model_name, Autosklearn)

import autosklearn
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm 
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT 
from ConfigSpace.configuration_space import ConfigurationSpace

class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace() # Return an empty configuration as there is None

# Add NoPreprocessing component to auto-sklearn.
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing) 

class AutosklearnNoPreprocessing(Autosklearn):
    model_name = "autosklearn_no_preprocessing"
    params = SimpleClassificationPipeline(
        include={"feature_preprocessor": ["no_preprocessing"], 
                 "data_preprocessor": ["NoPreprocessing"]},
        exclude = excluded 
    ).get_hyperparameter_search_space(dataset_properties={"signed": True})

register_classifier(AutosklearnNoPreprocessing.model_name, AutosklearnNoPreprocessing)


excluded_restricted = {"classifier": [
                          #"bernoulli_nb", "multinomial_nb", "gaussian_nb",
                          "liblinear_svc", "libsvm_svc", 
                          # Might want to delete knn since consumes
                          # a bit of memory and am going to run in parallel
                          #"k_nearest_neighbors"
                          ],
            "feature_preprocessor": ["kitchen_sinks", "liblinear_svc_preprocessor",
                                     "kernel_pca", "nystroem_sampler", "polynomial"]
            }    

class AutosklearnRestricted(Autosklearn):
    model_name = "autosklearn_restricted"
    params = SimpleClassificationPipeline(
        exclude = excluded_restricted 
    ).get_hyperparameter_search_space(dataset_properties={"signed": True})

register_classifier(AutosklearnRestricted.model_name, AutosklearnRestricted)