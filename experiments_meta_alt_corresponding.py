#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:11:11 2023

@author: david
"""
import itertools
import sys
import os
import pandas as pd
import pathlib

from codigo.classifiers.xgb import XGB
from codigo.classifiers.gmm import GMM
from codigo.classifiers.autosklearn import (
    AutosklearnRestricted as Autosklearn
)
import sklearn.metrics
import numpy as np
import time

from functools import reduce
from sklearn.utils import Bunch



def sensitivity(true, pred):
    per_class = [
        compute_elements(true, pred, label)
        for label in np.unique(true)
    ]
    
    total = reduce(lambda x, y: Bunch(**{k: x[k] + y[k] for k in x}), per_class, 
                   {"tp": np.int64(0), "tn": np.int64(0), "fn": np.int64(0), 
                    "fp": np.int64(0)})
    
    return sensitivity_from_elements(**total)

def specificity(true, pred):
    per_class = [
        compute_elements(true, pred, label)
        for label in np.unique(true)
    ]
    
    total = reduce(lambda x, y: Bunch(**{k: x[k] + y[k] for k in x}), per_class,
                   {"tp": np.int64(0), "tn": np.int64(0), "fn": np.int64(0), 
                    "fp": np.int64(0)})
    
    return specificity_from_elements(**total)

def sensitivity_from_elements(tp, tn, fp, fn):
    return tp / (tp + fn)

def specificity_from_elements(tp, tn, fp, fn):
    return tn / (tn + fp)

def compute_elements(true, pred, ref_label):
    true = true.copy()
    pred = pred.copy()
    
    mask_true_ref = true == ref_label
    mask_true_no_ref = true != ref_label
    
    mask_pred_ref = pred == ref_label
    mask_pred_no_ref = pred != ref_label
    
    tp = (mask_true_ref & mask_pred_ref).sum()
    tn = (mask_true_no_ref & mask_pred_no_ref).sum()
    fp = (mask_true_no_ref & mask_pred_ref).sum()
    fn = (mask_true_ref & mask_pred_no_ref).sum()
    
    return Bunch(tp=tp, tn=tn, fp=fp, fn=fn)



def save_results_dict(dictionary, file_path):
    pd.DataFrame([dictionary]).to_csv(file_path, index = False,
                                      mode = "a",
                                      header = not os.path.exists(file_path))

def get_files_in_subdirs(path):
    return list(
            itertools.chain(
                *(
                    [os.path.join(x[0], c) for c in x[2]]
                    for x in os.walk(path)
                )
            )
        )

def extract_pairs_datasets(model, ttc = None, method = None, 
                           base_dir = "ip_datasets_meta/changes_new"):
    search_dir = os.path.join(base_dir, model)
    if method is not None and ttc is None:
        raise ValueError("Must specify a ttc if method is specified")
    if ttc is not None:
        search_dir = os.path.join(search_dir, str(ttc))
    if method is not None:
        search_dir = os.path.join(search_dir, method)
    return get_pairs_of_files(get_files_in_subdirs(search_dir))


def get_pairs_of_files(file_list):
    ttc_dict = {0: (0, 1), 1: (0, 2), 2: (1, 0), 3: (1, 2), 4: (2, 0), 5: (2, 1)}
    return list(
            filter(lambda x: (
                x[0] != x[1] and "part_" in x[0] and "part_" in x[1]
                and x[0].rsplit("_", 2)[0] == x[1].rsplit("_", 2)[0]
                and int(x[0].rsplit("_", 1)[-1]) == ttc_dict[int(x[0].split(os.path.sep, 4)[3])][0]
                ),
                   itertools.product(file_list, file_list)
            )
        )

def split_path(path):
    drive, rest = os.path.splitdrive(path)
    parts = []
    while rest != "":
        rest, part = os.path.split(rest)
        parts.append(part)
    parts.reverse()
    return parts

def experiments(models, arg_slice, num_evals = 10, out_file = "results_ip_meta.csv"):
    label_col = "severity"
    model_dict = {"XGB": XGB, "GMM": GMM, "Autosklearn": Autosklearn}
            
    numbers = list(range(num_evals))
    file_pairs = list(
        itertools.chain(*(itertools.product([m], extract_pairs_datasets(m)) for m in models))
    )
    total = list(itertools.product(numbers, file_pairs))
    
    for i, (number, (model, (train_path, test_path))) in enumerate(total[arg_slice]):
        pars = model_dict[model].get_configuration_space().sample_configuration()
        m_ins = model_dict[model](**pars)
        
        dc = {}
        dc["model"] = model
        dc["number"] = number
        dc["ttc"] = int(split_path(train_path)[3])
        dc["method"] = split_path(train_path)[4]
        dc["categorize_country"] = False
        dc["timeseries_features"] = False
        dc["normalize_features"] = False
        print(i, number, model, dc["ttc"], dc["method"])

        
    
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        cols = list(df_train.columns.difference([label_col]))
        
        X_train = df_train[cols].to_numpy()
        y_train = df_train[label_col].to_numpy()
        
        X_test = df_test[cols].to_numpy()
        y_test = df_test[label_col].to_numpy()
        
        t = time.time()
        try:
            preds = m_ins.fit(X_train, y_train).predict(X_test)
        except Exception as e:
            preds = np.ones(X_test.shape[0]) * -1
        tt = time.time() - t
        
        combinacion_features_dict ={
            (True, False, False): 1,
            (False, True, False): 2,
            (False, False, True): 3,
            (True, True, False): 4,
            (True, False, True): 5,
            (False, True, True): 6,
            (True, True, True): 7
        }
            
        dc["train_path"] = str(train_path)
        dc["test_path"] = str(test_path)
        dc["base_features"] = os.path.basename(train_path).split("_")[3] == "True"
        dc["normalized_frequency"] = os.path.basename(train_path).split("_")[6] == "True"
        dc["neighborhood_features"] = os.path.basename(train_path).split("_")[7] == "True"
        dc["combinacion_features"] = combinacion_features_dict[
            (dc["base_features"], dc["neighborhood_features"], 
             dc["normalized_frequency"])
        ]
        dc["accuracy"] = sklearn.metrics.accuracy_score(y_test, preds)
        dc["mcc"] = sklearn.metrics.matthews_corrcoef(y_test, preds)
        dc["confusion_matrix"] = sklearn.metrics.confusion_matrix(y_test, preds).tolist()
        dc["model_params"] = pars.get_dictionary()
        dc["f1-score_micro"] = sklearn.metrics.f1_score(y_test, preds, average = "micro")
        dc["f1-score_macro"] = sklearn.metrics.f1_score(y_test, preds, average = "macro")
        dc["sensitivity"] = sensitivity(y_test, preds)
        dc["specificity"] = specificity(y_test, preds)
        dc["time"] = tt
                
        save_results_dict(dc, out_file)
        
if __name__ == "__main__":
    if len(sys.argv) == 3:
        n = int(sys.argv[1])
        c = int(sys.argv[2])
        arg = slice(n * c, n * (c + 1))
    elif len(sys.argv) == 2:
        c = int(sys.argv[1])
        arg = slice(c, c + 1)
    elif len(sys.argv) == 4:
        start = int(sys.argv[1])
        stop = int(sys.argv[3])
        if sys.argv[2] != "-":
            raise ValueError("Format: {start} - {end}")
        arg = slice(start, stop)
    else:
        #raise ValueError("Pass arguments")
        arg = slice(None)
    
    models = ["Autosklearn", "XGB", "GMM"]
    experiments(models, arg, num_evals = 10, out_file = "results_ip_meta.csv")
