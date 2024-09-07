#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:21:54 2022

@author: alumno
"""
from ..classifier_base import Classifier
from abc import ABC
import numpy as np
import sklearn.preprocessing


def predict_to_predict_proba(self, X):
    '''
    Function to add a predict_proba method to clustering methods
    without one. Simply assigns probability of 1 to the cluster the instance
    belongs to.
    
    Somewhat brittle since n_clusters does not appear in the interface, 
    but it will work for now
    '''
    preds = self.predict(X)
    try:
        if self.n_clusters is None:
            raise AttributeError("Could not")
        probs = np.zeros([X.shape[0], self.n_clusters])
        #probs = np.zeros([X.shape[0], self.purities.shape[0]])
    except (TypeError, AttributeError) as e:
        # For Birch
        print(e)
        probs = np.zeros([X.shape[0], len(self.subcluster_labels_ )])
    probs[np.indices((probs.shape[0], )), preds] = 1

    return probs
    

class BaseClusteringPredict(Classifier, ABC):        
    def fit(self, X, y):
        self.estimator = self.model(**self.params)
        self.estimator.fit(X, y)
        
        self.encoder = sklearn.preprocessing.LabelEncoder()
        self.encoder.fit(y)
        
        # Assume each cluster has at least one member Get the purities for each cluster Assumes all available classes are present in training
        preds = self.estimator.predict(X)
        # Transform to 0, n-labels - 1
        transformed_labels = self.encoder.transform(y)
        
        #Assumes all clusters have at least a member
        unique = np.unique(preds)

        #clusters = unique if max(unique) < len(unique) else np.arange(max(unique))
        clusters = np.arange(self.get_number_of_clusters())
        #clusters = np.array(range(len(self.estimator.weights_)))
                
        self.purities = np.zeros([clusters.size, self.encoder.classes_.size])
        
        # Unique sorts so the first cluster_id will be zero and so on
        for cluster_id in clusters:
            pred_mask = preds == cluster_id
            cluster_labels = transformed_labels[pred_mask]
            # sorted, so position 0 corresponds to label 0
            labels, counts = np.unique(cluster_labels, return_counts=True)
            tot = counts.sum()
            #if not tot:
            #    #raise ValueError("Found cluster without members")
            #    raise MemoryError("Found cluster without members")
            self.purities[cluster_id, labels] = counts / tot
        
        return self
            
    def predict_proba(self, X):
        """
        Translate probabilities of cluster membership to probabilities
        of class membership based on cluster purities
        (percentages of instances of each class)
        """
        preds = self.estimator.predict_proba(X)
        
        p = preds.dot(self.purities)
         
        # If we assume that the error prediction probability not within [0, 1]! is caused by small imprecisions in floatign point arithmetic 
        # After clipping we divide by sum, to ensure that probabilities sum up one
        #p = p.clip(0, 1)
        return p / p.sum(axis=1, keepdims=True)
            
    def predict(self, X):
        probas = self.predict_proba(X)
        
        # Translate back to classes
        return self.encoder.inverse_transform(probas.argmax(axis=1))


class BaseClusteringNoPredict(Classifier, ABC):        
    def fit(self, X, y):
        self.estimator = self.model(**self.params)

        self.encoder = sklearn.preprocessing.LabelEncoder()
        self.encoder.fit(y)
        
        self.X = X
        self.y = y
                
        return self
            
    def predict_proba(self, X):
        """
        Translate probabilities of cluster membership to probabilities
        of class membership based on cluster purities
        (percentages of instances of each class)
        """
        preds = self.estimator.fit_predict(np.concatenate([self.X, X], axis=0))
        train_preds = preds[:self.X.shape[0]]
        test_preds = preds[self.X.shape[0]:]
        
        if self.get_number_of_clusters() is not None:
            clusters = np.arange(self.get_number_of_clusters())
        else:
            #clusters = np.unique(preds)
            #clusters = np.unique(train_preds)
            clusters = np.unique(preds)
        
        #if clusters.size != known_clusters.size:
        #    raise ValueError("Cluster with no train samples found")
        
        self.purities = np.zeros([clusters.size, self.encoder.classes_.size])
        
        transformed_labels = self.encoder.transform(self.y)
        
        for cluster_id in clusters:
            pred_mask = train_preds == cluster_id
            cluster_labels = transformed_labels[pred_mask]
            # sorted, so position 0 corresponds to label 0
            labels, counts = np.unique(cluster_labels, return_counts=True)
            tot = counts.sum()
            #if not tot:
            #    #raise ValueError("Found cluster without members")
            #    raise MemoryError("Found cluster without members")
            self.purities[cluster_id, labels] = counts / tot
        
        probs = np.zeros([X.shape[0], clusters.size])
        probs[np.indices((probs.shape[0], )), test_preds] = 1
        p = probs.dot(self.purities)
         
        # If we assume that the error prediction probability not within [0, 1]! is caused by small imprecisions in floatign point arithmetic 
        # After clipping we divide by sum, to ensure that probabilities sum up one
        #p = p.clip(0, 1)
        return p / p.sum(axis=1, keepdims=True)
            
    def predict(self, X):
        probas = self.predict_proba(X)
        
        # Translate back to classes
        return self.encoder.inverse_transform(probas.argmax(axis=1))
