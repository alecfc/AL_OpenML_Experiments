#!/usr/bin/env python
# coding: utf-8

import os, sys;

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import warnings
from collections import Counter
from copy import deepcopy
import time
import torch
import torch.optim as optim
import joblib

sys.modules['sklearn.externals.joblib'] = joblib
import torch.utils.data as data_utils
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sn
# from pip command
from imblearn.under_sampling import RandomUnderSampler
from libact.base.dataset import Dataset
from libact.base.interfaces import Labeler, QueryStrategy
import libact.models
from modAL.disagreement import max_disagreement_sampling, vote_entropy_sampling
from modAL.models import ActiveLearner, Committee
from numba import vectorize, jit, cuda
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.io import arff
from AL_methods import *
from AL_plot_results import *
from AL_dataloader import *
from AL_utils import *

warnings.filterwarnings(
    'ignore')  # precision often has cases where there are no predictions for the minority class, which leads to zero-division. This keeps that warning from showing up.

ML_results_fully_trained = []

def test_model_performance(model, X_test, y_test):
    result_dict = {'Accuracy': 0.0, 'F1': 0.0, 'Recall': 0.0, 'Precision': 0.0, "AUC": 0.0, "Log Loss": 0.0}
    y_pred = model.predict(X_test)
    result_dict['Accuracy'] = accuracy_score(y_test, y_pred)
    result_dict['F1'] = f1_score(y_test, y_pred, average='macro')
    result_dict['Recall'] = recall_score(y_test, y_pred, average='macro')
    result_dict['Precision'] = precision_score(y_test, y_pred, average='macro')
    result_dict['AUC'] = roc_auc_score(y_test, y_pred, average='macro')
    result_dict['Log Loss'] = log_loss(y_test, y_pred)
    return result_dict


# In[9]:


def evaluate_fully_trained(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

    result_dict = test_model_performance(model.fit(X_train, y_train), X_test, y_test)
    print(type(model).__name__ + ': Fully trained classifier performance:')
    print(' - Accuracy: ', result_dict['Accuracy'])
    print(' - F1: ', result_dict['F1'])
    print(' - Recall: ', result_dict['Recall'])
    print(' - Precision: ', result_dict['Precision'])
    print(' - AUC: ', result_dict['AUC'])
    print(' - Log Loss: ', result_dict['Log Loss'])
    return X_train, X_test, y_train, y_test, result_dict


# In[10]:


def evaluate_subsample_trained(model, X_train, X_test, y_train, y_test, X_list, y_list):
    subsample_trained_results, biased_subsample_trained_results = [], []
    # Go through all subsamples and calculate performance on both unbiased (original) test set and on biased (sampled) test set
    for idx, X_subsample in enumerate(X_list):
        # Split subsampled data into train and test
        X_train_biased, X_test_biased, y_train_biased, y_test_biased = train_test_split(
            X_subsample, y_list[idx], test_size=0.3)
        subsample_trained_results.append(
            test_model_performance(model.fit(X_train_biased, y_train_biased), X_test, y_test))
        biased_subsample_trained_results.append(
            test_model_performance(model.fit(X_train_biased, y_train_biased), X_test_biased, y_test_biased))
    return subsample_trained_results, biased_subsample_trained_results


# In[11]:


def evaluate_all_models(X, y, X_list, y_list):
    ML_results_fully_trained = []
    ML_results_subsample_trained = []
    ML_results_subsample_trained_biased = []
    for model_number, model in ML_switcher.items():
        X_train, X_test, y_train, y_test, fully_trained_results = evaluate_fully_trained(model, X, y)
        ML_results_fully_trained.append(fully_trained_results)
        subsample_trained_results, biased_subsample_trained_results = evaluate_subsample_trained(model, X_train, X_test,
                                                                                                 y_train, y_test,
                                                                                                 X_list, y_list)
        ML_results_subsample_trained.append(ML_results_subsample_trained)
        ML_results_subsample_trained_biased.append(biased_subsample_trained_results)

    return X_train, X_test, y_train, y_test, ML_results_fully_trained, ML_results_subsample_trained, ML_results_subsample_trained_biased

# Specific learners used for the committee in QBC
QBC_LEARNERS = [2, 3]

# The number of learners of each classifier type specified by QBC_LEARNERS to be used in QBC
N_QBC_LEARNERS = 4

# Selection of QBC disagreement measure
# vote_entropy_sampling, consensus_entropy_sampling, max_disagreement_sampling
QBC_STRATEGY = max_disagreement_sampling


# Method for altering settings between runs. Settings that should generally stay consistent between runs are not changed by this method. 
# This can also be done by the main run_test method, making this method obsolete.
def set_settings(k_, execs_, n_queries_, n_init_, train_sum_, sep_class_, ml_method_, ml_sep_method_, al_method_,
                 qbc_learners_, n_qbc_learners_, weighting_factor_, save_figs_, save_ratio_, save_results_,
                 bandwidth_=0, sc_pca_=True):
    global K, EXECUTIONS, N_QUERIES, INITIALIZATION_SET, TRAIN_ON_SUM, SEPARATE_CLASSIFIER, ML_METHOD, ML_SEP_METHOD, AL_METHOD, QBC_LEARNERS, N_QBC_LEARNERS, OVERSAMPLE, UNDERSAMPLE, WEIGHTING_FACTOR, SAMPLING_RATIO, SAVEFIGS, SAVERATIO, SAVERESULTS, BANDWIDTH, SC_PCA
    K = k_
    EXECUTIONS = execs_
    N_QUERIES = n_queries_
    INITIALIZATION_SET = n_init_
    TRAIN_ON_SUM = train_sum_

    SEPARATE_CLASSIFIER = sep_class_
    ML_METHOD = ml_method_
    ML_SEP_METHOD = ml_sep_method_

    AL_METHOD = al_method_
    QBC_LEARNERS = qbc_learners_
    N_QBC_LEARNERS = n_qbc_learners_

    # normally calculated if set to 0
    BANDWIDTH = bandwidth_

    # Undersampling or oversampling according to the sampling ratio defined in the base settings
    UNDERSAMPLE = False
    OVERSAMPLE = False

    WEIGHTING_FACTOR = weighting_factor_
    SC_PCA = sc_pca_

    SAVEFIGS = save_figs_
    SAVERATIO = save_ratio_
    SAVERESULTS = save_results_


def determine_initial_set(x, y, prob_ratio, size_initial):
    number_class_1 = int(round(prob_ratio * size_initial))
    number_class_0 = int(size_initial - number_class_1)
    count_0 = 0
    count_1 = 0
    x_initial = []
    y_initial = []
    y = list(y)
    y_list = [None for i in y]
    while True:
        rand_idx = np.random.choice(range(len(y)), replace=True)
        if count_0 == number_class_0 and count_1 == number_class_1:
            return np.array(x_initial), np.array(y_initial), x, y, y_list
        if y[rand_idx] == 0 and (count_0 < number_class_0):
            x_initial.append(x[rand_idx])
            y_initial.append(y[rand_idx])
            y_list[rand_idx] = y[rand_idx]
            x, y = np.delete(x, rand_idx, axis=0), np.delete(y, rand_idx)
            count_0 += 1
            continue
        elif y[rand_idx] == 1 and (count_1 < number_class_1):
            x_initial.append(x[rand_idx])
            y_initial.append(y[rand_idx])
            y_list[rand_idx] = y[rand_idx]
            x, y = np.delete(x, rand_idx, axis=0), np.delete(y, rand_idx)
            count_1 += 1

# @jit
def run_AL_test(X, y, X_df, k_, execs_, n_queries_, n_instantiations_, original_class_ratio_, initial_ratio_,
                initial_size_, ml_method_,
                al_method_, qbc_learners_, n_qbc_learners_, save_results_, normalize_data_, prop_performance_,
                file_path_, ML_results_fully_trained_, exp_subtype_, al_dict_=AL_switcher):
    # Keep track of results for each query. Used later to calculate incremental performance. +1 because the first value is the performance directly after initialization
    accuracy_results = pd.DataFrame(columns=range(n_queries_ + 1))
    macro_f1_results = pd.DataFrame(columns=range(n_queries_ + 1))
    recall_results = pd.DataFrame(columns=range(n_queries_ + 1))
    precision_results = pd.DataFrame(columns=range(n_queries_ + 1))
    auc_results = pd.DataFrame(columns=range(n_queries_ + 1))
    loss_results = pd.DataFrame(columns=range(n_queries_ + 1))
    selected_labels_table = pd.DataFrame(columns=range(n_queries_))
    selected_instances_table = pd.DataFrame(columns=range(n_queries_))
    overall_weighted_risks = []

    class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
    temp_X = X.copy()
    temp_y = y.copy()
    # Keep track of feature importance during training.
    feat_importance_cons = pd.DataFrame([], columns=range(n_queries_ + 1), index=range(execs_ * k_))

    # Keep track of selection frequencies of individual trajectories
    instance_freq = pd.DataFrame(0, index=range(execs_), columns=X_df.index)

    # trajectory_selection_frequencies = pd.DataFrame(0, index=range(n_initial), columns = X[0])
    model = ML_switcher.get(ml_method_)
    # Perform K-fold validation for EXECUTIONS number of times. This gives more total repetitions, and thus a smoother view of learning performance
    for execs in range(execs_):
        weighted_risk = []
        # Take stratified folds to make sure each fold contains enough instances of the majority class
        skf = StratifiedKFold(n_splits=K, shuffle=True)
        train_set_indices = np.array(list(skf.split(temp_X, temp_y)))
        # K-fold cross validation
        for i in range(k_):
            # sys.stdout.write("Validating on fold: ", i + 1, "/", K, end="\r")
            print('\r', "Validating on fold: ", i + 1, "/", k_)
            sys.stdout.flush()
            X_train = X[train_set_indices[i][0]]
            y_train = y[train_set_indices[i][0]]

            X_test = X[train_set_indices[i][1]]
            y_test = y[train_set_indices[i][1]]

            for j in range(n_instantiations_):

                if al_method_ == 5 or al_method_ == 6 or al_method_ == 9:
                    x_initial, y_initial, X_temp, y_temp, y_list = determine_initial_set(X_train, y_train,
                                                                                         initial_ratio_,
                                                                                         initial_size_)
                    ds = libact.base.dataset.Dataset(X_train, y_list)
                else:
                    x_initial, y_initial, X_train, y_train, y_list = determine_initial_set(X_train, y_train,
                                                                                           initial_ratio_,
                                                                                           initial_size_)
                if al_method_ == 4:
                    committee_list = []
                    for qbc_model in qbc_learners_:  # qbc_learners
                        for n in range(n_qbc_learners_):  # n_qbc_learners
                            original_estimator = ML_switcher.get(qbc_model)
                            new_estimator = deepcopy(original_estimator)
                            committee_member = ActiveLearner(
                                estimator=new_estimator,
                                X_training=x_initial, y_training=y_initial
                            )
                            committee_list.append(committee_member)
                    # Assembling the committee
                    learner = Committee(
                        learner_list=committee_list,
                        query_strategy=QBC_STRATEGY
                    )
                    prediction = learner.predict(X_test)
                elif al_method_ == 5:
                    model = libact.models.LogisticRegression()
                    sub_qs = UncertaintySampling(
                        dataset=ds, method='lc', model=libact.models.LogisticRegression())
                    learner = HierarchicalSampling(
                        dataset=ds,  # Dataset object
                        classes=[0, 1],
                        active_selecting=True,
                        subsample_qs=sub_qs
                    )
                    model.train(ds)
                    # Make first set of predictions (before querying)
                    prediction = model.predict(X_test)
                elif al_method_ == 6:
                    model = libact.models.LogisticRegression()
                    learner = QUIRE(
                        dataset=ds,  # Dataset object
                        kernel='rbf'
                    )
                    model.train(ds)
                    # Make first set of predictions (before querying)
                    prediction = model.predict(X_test)
                elif al_method_ == 7:
                    weighted_dataset = OpenMLDataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                     X_initial=x_initial, y_initial=y_initial)
                    weighted_dataset.set_model(model)
                    model.fit(weighted_dataset.acquired_dataset[1], weighted_dataset.acquired_dataset[0])
                    prediction = model.predict(X_test)
                elif al_method_ == 9:
                    model = libact.models.LogisticRegression()
                    quota = 100
                    learner = ActiveLearningByLearning(
                        dataset=ds,
                        T=quota,  # qs.make_query can be called for at most 100 times
                        query_strategies=[
                            RandomSampling(ds, T=quota, model=model),
                            UncertaintySampling(ds, T=quota, model=model, method='lc'),
                            UncertaintySampling(ds, T=quota, model=model, method='sm'),
                            UncertaintySampling(ds, T=quota, model=model, method='entropy'),
                            DWUS(ds, T=quota, model=model),
                            DWUS(ds, T=quota, model=model, n_clusters=10)
                        ],
                        uniform_sampler=False,
                        model=model
                    )
                    model.train(ds)
                    # Make first set of predictions (before querying)
                    prediction = model.predict(X_test)
                else:
                    # print(x_initial)
                    # print(y_initial)
                    learner = ActiveLearner(
                        estimator=deepcopy(model),
                        query_strategy=AL_switcher.get(al_method_),
                        X_training=x_initial, y_training=y_initial
                    )
                    # Make first set of predictions (before querying)
                    prediction = learner.predict(X_test)

                unqueried_score_single = accuracy_score(y_test, prediction)
                unqueried_macro_f1_single = f1_score(y_test, prediction, average='macro')
                unqueried_recall_single = recall_score(y_test, prediction, average='macro')
                unqueried_precision_single = precision_score(y_test, prediction, average='macro', zero_division=0)
                unqueried_auc_single = roc_auc_score(y_test, prediction, average='macro')
                unqueried_loss_single = log_loss(y_test, prediction)

                # Labels of selected training instances. Used for calculating metrics later on
                selected_training_instance_labels = np.array([], dtype=int)

                # Labels of selected training instances. Used for calculating metrics later on
                selected_training_instances = np.array([])

                # Store performance of every calculated score
                accuracy_history = [unqueried_score_single]
                macro_f1_history = [unqueried_macro_f1_single]
                recall_history = [unqueried_recall_single]
                precision_history = [unqueried_precision_single]
                auc_history = [unqueried_auc_single]
                loss_history = [unqueried_loss_single]

                for idx in range(n_queries_):
                    if al_method_ == 5 or al_method_ == 6 or al_method_ == 9:
                        query_idx = learner.make_query()
                        ds.update(query_idx, y_train[query_idx])
                        model.train(ds)
                        predictions = model.predict(X_test)
                    elif al_method_ == 7 or al_method_ == 8:
                        query_idx, probability_distribution = weighted_dataset.proposal_distribution_uniform()
                        weighted_dataset.acquire_point(query_idx, probability_distribution)
                        model.fit(weighted_dataset.acquired_dataset[1], weighted_dataset.acquired_dataset[0],
                                  sample_weight=weighted_dataset.refined_weighting_scheme())
                        query_idx = query_idx[0].max()
                        weighted_risk.append(weighted_dataset.estimate_risk(weighting_scheme="refined"))
                        predictions = model.predict(X_test)
                    else:
                        # Query instance using AL method
                        query_idx, query_instance = learner.query(X_train)

                        X_temp, y_temp = X_train[query_idx].reshape(1, -1), y_train[query_idx].reshape(1, )

                        # Training a separate classifier (producer) used for selecting queries
                        learner.teach(X=X_temp, y=y_temp)
                        # Predict on test set
                        predictions = learner.predict(X_test)

                    # For measuring performance in terms of label selection
                    selected_training_instance_labels = np.append(selected_training_instance_labels, y_train[query_idx])

                    # For measuring algorithm behaviour through instance selection
                    selected_training_instances = np.append(selected_training_instances, query_idx)

                    # For measuring frequencies of selection of specific instances over the course of learning
                    instance_freq.iloc[execs][query_idx] += 1
                    # print(instance_freq.iloc[idx][query_idx])

                    # Calculate and report our model's accuracy.
                    model_accuracy = accuracy_score(y_test, predictions)
                    model_macro_f1 = f1_score(y_test, predictions, average='macro')
                    model_recall = recall_score(y_test, predictions, average='macro')
                    model_precision = precision_score(y_test, predictions, average='macro',
                                                      zero_division=0)  # zero_division because sometimes no predictions of looping class
                    model_auc = roc_auc_score(y_test, predictions, average='macro')
                    model_loss = log_loss(y_test, predictions)

                    # Save our model's performance for plotting.
                    accuracy_history.append(model_accuracy)
                    macro_f1_history.append(model_macro_f1)
                    recall_history.append(model_recall)
                    precision_history.append(model_precision)
                    auc_history.append(model_auc)
                    loss_history.append(model_loss)

                    if al_method_ < 5:
                        # Remove the queried instance from the unlabeled pool.
                        X_train, y_train = np.delete(X_train, query_idx, axis=0), np.delete(y_train, query_idx)

                # Save the specific labels chosen at each training point
                selected_labels_table = selected_labels_table.append(pd.Series(selected_training_instance_labels),
                                                                     ignore_index=True)

                selected_instances_table = selected_instances_table.append(pd.Series(selected_training_instances),
                                                                           ignore_index=True)
                # store results after each test
                accuracy_results = accuracy_results.append(pd.Series(accuracy_history), ignore_index=True)
                macro_f1_results = macro_f1_results.append(pd.Series(macro_f1_history), ignore_index=True)
                recall_results = recall_results.append(pd.Series(recall_history), ignore_index=True)
                precision_results = precision_results.append(pd.Series(precision_history), ignore_index=True)
                auc_results = auc_results.append(pd.Series(auc_history), ignore_index=True)
                loss_results = loss_results.append(pd.Series(loss_history), ignore_index=True)

    # measure_names = ['Selected Label Ratio', 'AUC']
    dataset_str = dataset.name + ' class ratio ' + str(class_ratio)
    al_str = al_dict_.get(al_method_).__name__ + ' initial size ' + str(initial_size_) + ' initial ratio ' + str(
        initial_ratio_)
    ml_str = type(ML_switcher[ml_method_]).__name__
    string = dataset_str + ' ' + al_str + ' ' + ml_str + ' for ' + str(n_queries_) + ' queries'

    # Plot the bias over time
    loss_results -= ML_results_fully_trained_[ml_method_ - 1]['Log Loss']
    plot_bias(y_initial, selected_labels_table, class_ratio, save_results_, file_path_, 'Bias of ' + string,
              dataset_name_=dataset.name)
    plot_class_per_sample(labels_=selected_labels_table, original_class_ratio_=class_ratio, name_=string, save_=True,
                          file_path_=file_path_, dataset_name_=dataset.name)

    # Plot the top selected instances over all executions
    plot_top_selected_instances(selected_instances_table, selected_labels_table, save_results_, file_path_,
                                'Top Instances of ' + string)

    selected_labels_table = selected_labels_table.astype(float)
    all_results = list(
        [accuracy_results, macro_f1_results, recall_results, precision_results, selected_labels_table, auc_results,
         loss_results])
    # all_results = list([selected_labels_table, auc_results])

    measure_names = ['Accuracy', 'F1', 'Recall', 'Precision', "Label Ratio", "AUC",
                     'Loss Difference']  # , "Feature Importance Consumer", "Feature Importance Producer"]
    measure_names = measure_names[:len(all_results)]

    for result, measure_name in zip(all_results, measure_names):
        if save_results_:
            if not os.path.exists("../Results/" + EXP_TYPE + '/' + exp_subtype_ + '/' + measure_name + '/'):
                os.makedirs("../Results/" + EXP_TYPE + '/' + exp_subtype_ + '/' + measure_name + '/')
            result.to_pickle(
                "../Results/" + EXP_TYPE + '/' + exp_subtype_ + '/' + measure_name + '/' + measure_name + ' of ' + string + ".pkl")
        plot_results(X, result, measure_name, ML_results_fully_trained, measure_name + ' of ' + string, al_method_,
                     ml_method_, save_results_, normalize_data_, prop_performance_, file_path_,
                     data_title_=dataset.name, al_dict_=al_dict_)


def dataset_performance_measure_results(filepath_, df_measure_results_, dataset_name_):
    file_found = False
    for exp_folder in os.scandir(filepath_):
        df_result = pd.read_pickle(exp_folder)
        if dataset_name_ in exp_folder.name:
            df_measure_results_ = df_measure_results_.append(df_result, ignore_index=True)
            if not file_found:
                file_name = ' '.join(exp_folder.name.split(' ')[3:])
                file_found = True
    return df_measure_results_, file_name


def aggregate_performance_measure_results(filepath_, agg_measure_results_):
    for exp_folder in os.scandir(filepath_):
        df_result = pd.read_pickle(exp_folder)
        agg_measure_results_ = agg_measure_results_.append(df_result, ignore_index=True)
    return agg_measure_results_


def read_and_replot_measure(exp_type_, exp_sub_type_, measure_, n_queries_):
    for idx, dataset_name in enumerate(dataset_list):
        dataset = openml.datasets.get_dataset(dataset_name)
        if not os.path.exists('../Figures/' + exp_type_ + '/' + dataset.name + '/' + exp_sub_type_ + '/'):
            os.makedirs('../Figures/' + exp_type_ + '/' + dataset.name + '/' + exp_sub_type_ + '/')
        X_df, y_df, X, y, number_majority, number_minority = preprocess_openML_dataset(dataset)
        if measure_ == 'Label Ratio':
            df = pd.DataFrame(columns=range(n_queries_))
        else:
            df = pd.DataFrame(columns=range(n_queries_ + 1))

        class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
        df, file_name = dataset_performance_measure_results(
            '../Results/' + exp_type_ + '/' + exp_sub_type_ + '/' + measure_ + '/', df, dataset.name)
        file_name = file_name.replace('.pkl','')
        if measure_ == 'Label Ratio' or measure_ == 'Loss Difference':
            al_method_name = (str(file_name.split(' ')[4]))
            ml_method_name = (str(file_name.split(' ')[11]))
        else:
            al_method_name = (str(file_name.split(' ')[3]))
            ml_method_name = (str(file_name.split(' ')[10]))
            file_name = dataset_name + ' ' + file_name
        al_method = [k for k, v in AL_switcher.items() if v.__name__ == al_method_name][0]
        ml_method = [k for k, v in ML_switcher.items() if type(v).__name__ == ml_method_name][0]
        if measure_ == 'Label Ratio':
            plot_class_per_sample(labels_=df, original_class_ratio_=class_ratio, name_=file_name, save_=True,
                                  file_path_='../Figures/' + exp_type_ + '/' + dataset.name + '/' + exp_sub_type_ + '/',
                                  dataset_name_=dataset.name)
        plot_results(X, df, measure_, ML_results_fully_trained, measure_ + ' of ' + file_name, al_method,
                     ml_method, True, False, False,
                     '../Figures/' + exp_type_ + '/' + dataset.name + '/' + exp_sub_type_ + '/',
                     data_title_=dataset.name, al_dict_=AL_switcher)
    return al_method, ml_method


def read_dataset_results(base_filepath_, n_queries_, dataset_name_, data_size=500):
    performance_metrics = {'Accuracy': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'F1': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'Recall': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'Precision': pd.DataFrame(columns=range(n_queries_ + 1)),
                           "Label Ratio": pd.DataFrame(columns=range(n_queries_)),
                           "AUC": pd.DataFrame(columns=range(n_queries_ + 1)),
                           "Loss Difference": pd.DataFrame(columns=range(n_queries_ + 1))}
    df_results = {}
    for exp_folder in os.scandir(base_filepath_):
        if data_size > 5000 and exp_folder.name == 'quire':
            continue
        df_results[exp_folder.name] = performance_metrics.copy()
        for exp_type_result in os.scandir(exp_folder):
            df_results[exp_folder.name][exp_type_result.name], file_name = dataset_performance_measure_results(
                exp_type_result.path, df_results[exp_folder.name][exp_type_result.name], dataset_name_)

    return df_results


def read_aggregate_results(base_filepath_, n_queries_):
    performance_metrics = {'Accuracy': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'F1': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'Recall': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'Precision': pd.DataFrame(columns=range(n_queries_ + 1)),
                           "Label Ratio": pd.DataFrame(columns=range(n_queries_)),
                           "AUC": pd.DataFrame(columns=range(n_queries_ + 1)),
                           "Loss Difference": pd.DataFrame(columns=range(n_queries_ + 1))}
    agg_results = {}
    for exp_folder in os.scandir(base_filepath_):
        agg_results[exp_folder.name] = performance_metrics.copy()
        for exp_type_result in os.scandir(exp_folder):
            agg_results[exp_folder.name][exp_type_result.name] = aggregate_performance_measure_results(
                exp_type_result.path, agg_results[exp_folder.name][exp_type_result.name])
    return agg_results


def compare_results_single_dataset(all_dataset_results_, file_path_, measure_name_, experiment_type_, setting_names_,
                                   dataset_name_, initial_labels_, original_class_ratio_):
    measure_results = []
    for subset, subset_results in all_dataset_results_.items():
        measure_results.append(all_dataset_results_[subset][measure_name_])

    if measure_name_ == 'Label Ratio':
        plot_multiple_bias(initial_labels_, measure_results, original_class_ratio_, True, file_path_, setting_names_, experiment_type_, dataset_name_)
    plot_3d_results(measure_results, measure_name_, save_=True,
                    file_path_=file_path_,
                    z_labels_=setting_names_,
                    experiment_type_=experiment_type_, dataset_name_=dataset_name_)
    plot_multiple(measure_results, measure_name_, setting_names_=setting_names_,
                  experiment_type_=experiment_type_,
                  save_=True, file_path_=file_path_, dataset_name_=dataset_name_)
    plot_multiple(measure_results, measure_name_, setting_names_=setting_names_, experiment_type_=experiment_type_,
                  save_=True, file_path_=file_path_, dataset_name_=dataset_name_)


def ci_run_single_dataset(X_list, y_list, al_method, ml_method, X_df, ML_results_fully_trained):
    # Get subsamples of OpenML dataset
    # Iterate through the three subsets with one AL method and one ML method
    subsets = ['Original', 'Balanced', '75-25', '95-5']
    original_class_ratio = round(Counter(y_list[0])[1] / (Counter(y_list[0])[0] + Counter(y_list[0])[1]), 3)
    for idx, X_data in enumerate(X_list):
        print('Evaluating on ' + subsets[idx] + ' class ratio data using', AL_switcher[al_method].__name__, 'and the',
              type(ML_switcher[ml_method]).__name__, 'classifier')
        file_path = "../Figures/Class_Imbalance/" + dataset.name + '/' + subsets[idx] + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        run_AL_test(X_data, y_list[idx], X_df, k_=5, execs_=20, n_queries_=100,
                    n_instantiations_=1, original_class_ratio_=original_class_ratio,
                    initial_ratio_=0.5, initial_size_=10, ml_method_=ml_method,
                    al_method_=al_method, qbc_learners_=[2, 3],
                    n_qbc_learners_=4,
                    save_results_=True, normalize_data_=False,
                    prop_performance_=False,
                    file_path_=file_path,
                    ML_results_fully_trained_=ML_results_fully_trained,
                    exp_subtype_=subsets[idx])

    all_dataset_results = read_dataset_results('../Results/Class_Imbalance/', 100, dataset.name)

    for measure_name, measure_results in all_dataset_results[subsets[0]].items():
        compare_results_single_dataset(all_dataset_results_=all_dataset_results,
                                       file_path_="../Figures/Class_Imbalance/" + dataset.name + '/',
                                       measure_name_=measure_name, experiment_type_='Class Imbalance',
                                       setting_names_=list(all_dataset_results.keys()), dataset_name_=dataset.name,
                                       initial_labels_=[0,0,0,0,0,1,1,1,1,1], original_class_ratio_=original_class_ratio)


def al_run_single_dataset(X, y, ml_method, X_df, ML_results_fully_trained, al_dict):
    active_learning_methods = []
    original_class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
    for al_method_number, al_method in al_dict.items():
        active_learning_methods.append(al_method.__name__)
        print('Evaluating', al_method.__name__, 'using the', type(ML_switcher[ml_method]).__name__, 'classifier')
        file_path = "../Figures/AL_Methods/" + dataset.name + '/' + al_method.__name__ + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if al_method_number == 4 or al_method_number == 6:
            execs = 5
        else:
            execs = 20
        if X.shape[0] > 5000 and al_method_number == 6:
            continue
        start = time.ctime()
        run_AL_test(X, y, X_df, k_=5, execs_=execs, n_queries_=100,
                    n_instantiations_=1, original_class_ratio_=original_class_ratio,
                    initial_ratio_=0.5, initial_size_=10, ml_method_=ml_method,
                    al_method_=al_method_number, qbc_learners_=[2, 3],
                    n_qbc_learners_=4,
                    save_results_=True, normalize_data_=False,
                    prop_performance_=False,
                    file_path_=file_path,
                    ML_results_fully_trained_=ML_results_fully_trained,
                    exp_subtype_=al_method.__name__, al_dict_=al_dict)
        end = time.ctime()
        print('started', start)
        print('finished', end)
    all_dataset_results = read_dataset_results('../Results/AL_Methods/', 100, dataset.name, X.shape[0])

    for measure_name, measure_results in all_dataset_results[active_learning_methods[0]].items():
        compare_results_single_dataset(all_dataset_results_=all_dataset_results,
                                       file_path_="../Figures/AL_Methods/" + dataset.name + '/',
                                       measure_name_=measure_name, experiment_type_='AL Methods',
                                       setting_names_=list(all_dataset_results.keys()), dataset_name_=dataset.name,
                                       initial_labels_=[0,0,0,0,0,1,1,1,1,1], original_class_ratio_=original_class_ratio)

def ml_run_single_dataset(X, y, al_method, X_df, ML_results_fully_trained):
    # Perform the active learning querying cycle for all different machine learning classifiers on a single dataset
    ml_method_numbers = [1, 2, 3]
    ml_methods = []
    for idx, key in enumerate(ml_method_numbers):
        ml_methods.append(type(ML_switcher[key]).__name__)
    original_class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
    for ml_method_number, ml_method in ML_switcher.items():
        print('Evaluating the ' + type(ML_switcher[ml_method_number]).__name__ + ' classifier', 'with',
              AL_switcher[al_method].__name__)
        file_path = "../Figures/ML_Methods/" + dataset.name + '/' + type(ML_switcher[ml_method_number]).__name__ + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        start = time.ctime()
        run_AL_test(X, y, X_df, k_=5, execs_=20, n_queries_=100,
                    n_instantiations_=1, original_class_ratio_=original_class_ratio,
                    initial_ratio_=0.5, initial_size_=10,
                    ml_method_=ml_method_number, al_method_=al_method,
                    qbc_learners_=[2, 3], n_qbc_learners_=4,
                    save_results_=True, normalize_data_=False,
                    prop_performance_=False,
                    file_path_=file_path,
                    ML_results_fully_trained_=ML_results_fully_trained,
                    exp_subtype_=type(ML_switcher[ml_method_number]).__name__)
        end = time.ctime()
        print('started', start)
        print('finished', end)
    all_dataset_results = read_dataset_results('../Results/ML_Methods/', 100, dataset.name)
    for measure_name, measure_results in all_dataset_results[ml_methods[0]].items():
        compare_results_single_dataset(all_dataset_results_=all_dataset_results,
                                       file_path_="../Figures/ML_Methods/" + dataset.name + '/',
                                       measure_name_=measure_name, experiment_type_='ML Methods',
                                       setting_names_=list(all_dataset_results.keys()), dataset_name_=dataset.name,
                                       initial_labels_=[0,0,0,0,0,1,1,1,1,1], original_class_ratio_=original_class_ratio)


def init_run_single_dataset(X, y, al_method, ml_method, X_df, ML_results_fully_trained):
    # Iterate through all active learning methods
    class_ratios = [0.1, 0.5, 0.25]
    original_class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
    for idx, init_ratio in enumerate(class_ratios):
        print('Evaluating initial class ratio of', str(init_ratio), 'with', type(ML_switcher[ml_method]).__name__,
              'classifier', 'and', AL_switcher[al_method].__name__)
        file_path = "../Figures/Initial_Class_Ratio/" + dataset.name + '/' + 'Initial class ratio ' + str(
            init_ratio) + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        run_AL_test(X, y, X_df, k_=5, execs_=20, n_queries_=100,
                    n_instantiations_=1, original_class_ratio_=original_class_ratio,
                    initial_ratio_=init_ratio, initial_size_=10,
                    ml_method_=ml_method, al_method_=al_method,
                    qbc_learners_=[2, 3], n_qbc_learners_=4,
                    save_results_=True, normalize_data_=False,
                    prop_performance_=False,
                    file_path_=file_path,
                    ML_results_fully_trained_=ML_results_fully_trained,
                    exp_subtype_='Initial class ratio ' + str(init_ratio))
    all_dataset_results = read_dataset_results('../Results/Initial_Class_Ratio/', 100, dataset.name)

    for measure_name, measure_results in all_dataset_results['Initial class ratio ' + str(class_ratios[0])].items():
        compare_results_single_dataset(all_dataset_results_=all_dataset_results,
                                       file_path_="../Figures/Initial_Class_Ratio/" + dataset.name + '/',
                                       measure_name_=measure_name, experiment_type_='Initial Class Ratio',
                                       setting_names_=list(all_dataset_results.keys()), dataset_name_=dataset.name,
                                       initial_labels_=[0,0,0,0,0,1,1,1,1,1], original_class_ratio_=original_class_ratio)

def preprocess_openML_dataset(dataset):
    X_df, y_df, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    labelencoder = LabelEncoder()
    X_categorical_cols = list(X_df.select_dtypes(include=["category"]))

    for feature in X_categorical_cols:
        X_df[feature] = labelencoder.fit_transform(X_df[feature])
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    y = labelencoder.fit_transform(y)
    number_majority = Counter(y)[0]
    number_minority = Counter(y)[1]

    return X_df, y_df, X, y, number_majority, number_minority

def create_openML_subsamples(X, y):
    rus = RandomUnderSampler(random_state=42, sampling_strategy=ratio_multiplier(y=y, ratio=0.5))
    X_balanced, y_balanced = rus.fit_resample(X, y)
    rus = RandomUnderSampler(random_state=42, sampling_strategy=ratio_multiplier(y=y, ratio=0.25))
    X_minor_imb, y_minor_imb = rus.fit_resample(X, y)
    rus = RandomUnderSampler(random_state=42, sampling_strategy=ratio_multiplier(y=y, ratio=0.05))
    X_high_imb, y_high_imb = rus.fit_resample(X, y)
    X_list, y_list = [], []
    X_list.append(X)
    y_list.append(y)
    X_list.append(X_balanced)
    X_list.append(X_minor_imb)
    X_list.append(X_high_imb)
    y_list.append(y_balanced)
    y_list.append(y_minor_imb)
    y_list.append(y_high_imb)
    for i in y_list:
        print(Counter(i))
    return X_list, y_list


def run_openML_test(experiment_):
    n_queries = 100
    ml_method = 2
    al_method = 2
    al_dict = AL_switcher
    global dataset
    global EXP_TYPE
    EXP_TYPE = experiment_

    for idx, dataset_name in enumerate(dataset_list):
        dataset = openml.datasets.get_dataset(dataset_name)
        print('Running experiments on', dataset.name)
        X_df, y_df, X, y, number_majority, number_minority = preprocess_openML_dataset(dataset)
        print('Class Ratio:', round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3))
        X_list, y_list = create_openML_subsamples(X, y)
        X_train, X_test, y_train, y_test, ML_results_fully_trained, ML_results_subsample_trained, ML_results_subsample_trained_biased = evaluate_all_models(
            X, y, X_list, y_list)
        # run_experiments_openML_dataset(X_train, y_train, X_list=X_list,y_list=y_list)
        if experiment_ == 'Class_Imbalance':
            ci_run_single_dataset(X_list, y_list, al_method, ml_method, X_df,
                                  ML_results_fully_trained)
        elif experiment_ == 'AL_Methods':
            al_run_single_dataset(X, y, ml_method, X_df,
                                  ML_results_fully_trained, al_dict)
        elif experiment_ == 'ML_Methods':
            ml_run_single_dataset(X, y, al_method, X_df,
                                  ML_results_fully_trained)
        if experiment_ == 'Initial_Class_Ratio':
            init_run_single_dataset(X, y, al_method, ml_method, X_df, ML_results_fully_trained)

    agg_measure_results = read_aggregate_results('../Results/' + experiment_ + '/', n_queries)
    plot_aggregate_results(experiment_, agg_measure_results, al_method, ml_method, al_dict)
    plot_aggregate_comparison(experiment_, agg_measure_results)


def plot_all(exp_type_, subsets_):
    exp_name = exp_type_.replace('_',' ')
    for measure in ['Accuracy', 'F1', 'Recall', 'Precision', "Label Ratio", "AUC", 'Loss Difference']:
        for subset in subsets_:
            al_method, ml_method = read_and_replot_measure(exp_type_=exp_type_, exp_sub_type_=subset, measure_=measure, n_queries_=100)
    for idx, dataset_name in enumerate(dataset_list):
        dataset = openml.datasets.get_dataset(dataset_name)
        all_dataset_results = read_dataset_results('../Results/'+ exp_type_ + '/', 100, dataset.name)
        X_df, y_df, X, y, number_majority, number_minority = preprocess_openML_dataset(dataset)
        original_class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
        for measure_name, measure_results in all_dataset_results[subsets_[0]].items():
            compare_results_single_dataset(all_dataset_results_=all_dataset_results,
                                           file_path_='../Figures/'+ exp_type_ + '/' + dataset.name + '/',
                                           measure_name_=measure_name, experiment_type_=exp_name,
                                           setting_names_=list(all_dataset_results.keys()), dataset_name_=dataset.name,
                                           initial_labels_=[0,0,0,0,0,1,1,1,1,1], original_class_ratio_=original_class_ratio)

    agg_measure_results = read_aggregate_results('../Results/'+ exp_type_ + '/', 100)
    plot_aggregate_results(exp_type_, agg_measure_results, al_method, ml_method, AL_switcher)
    plot_aggregate_comparison(exp_type_, agg_measure_results)

if __name__ == "__main__":
    # This is done based on the dataset name 'cylinder-bands',
    # amount of instances per dataset [cylinder-bands:540, monks-problems-3:554, qsar-biodeg:1055, banknote-authentication:1372, steel-plates-fault:1941, scene:2407,
    # ozone-level-8hr:2534, kr-vs-kp:3196, Bioresponse:3751, wilt:4839, churn:5000, spambase:4601, mushroom:8124, PhishingWebsites:11055, electricity:45300, creditcard:285000]
    # 'monks-problems-3', 'qsar-biodeg', 'hill-valley', 'banknote-authentication', 'steel-plates-fault', 'jasmine', 'scene', 'ozone-level-8hr',
    # 'kr-vs-kp', 'Bioresponse','spambase', 'wilt',
    # 'monks-problems-3', 'qsar-biodeg', 'hill-valley', 'banknote-authentication', 'steel-plates-fault', 'jasmine', 'scene', 'ozone-level-8hr',
    #      'kr-vs-kp', 'Bioresponse','spambase', 'wilt','churn', 'mushroom',
    # dataset_list = ['monks-problems-3', 'qsar-biodeg', 'hill-valley', 'banknote-authentication', 'steel-plates-fault', 'jasmine', 'scene', 'ozone-level-8hr',
    #      'kr-vs-kp', 'Bioresponse','spambase', 'wilt','churn', 'mushroom','PhishingWebsites']
    dataset_list = ['monks-problems-3', 'qsar-biodeg', 'hill-valley', 'banknote-authentication', 'steel-plates-fault', 'jasmine', 'scene', 'ozone-level-8hr',
          'kr-vs-kp', 'Bioresponse','spambase', 'wilt','churn', 'mushroom','PhishingWebsites']  # ,'ringnorm', 'mushroom']#, 'electricity', 'creditcard']
    # To plot aggregate results
    #agg_measure_results = read_aggregate_results('../Results/AL_Methods/', 100)
    #plot_aggregate_results('AL_Methods', agg_measure_results, 2, 1, AL_switcher)
    #plot_aggregate_comparison('AL_Methods', agg_measure_results)

    # Main experiments
    #run_openML_test(experiment_='Class_Imbalance')
    #run_openML_test(experiment_='AL_Methods')
    #run_openML_test(experiment_='ML_Methods')
    #run_openML_test(experiment_='Initial_Class_Ratio')

    # Replot single
    subsets_al1 = ['random_sampling', 'uncertainty_sampling', 'density_sampling', 'qbc_sampling']
    subsets_ci = ['75-25', '95-5', 'Balanced', 'Original']
    subsets_ml = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier']
    subsets_init = ['Initial class ratio 0.1', 'Initial class ratio 0.5', 'Initial class ratio 0.25']

    subsets_al2 = ['random_sampling', 'uncertainty_sampling', 'density_sampling', 'hierarchical_sampling', 'quire', 'albl']
    plot_all(exp_type_='ML_Methods', subsets_=subsets_ml)

    print('Done')
# Test example
# confusion_matrix = np.array([[41792,67554,19872,99459],[ 24901,11070,23452,15790],[20190,24793,34254,10582],[90190,24793,34254,20582]])
# confusion_matrix = pd.DataFrame(confusion_matrix, columns = ['trip1', 'trip2', 'trip3', 'trip4'])

# plot_heatmap(confusion_matrix, threshold = 1, bin_size = 2, name = "Test_case", target_ids = [])


# In[ ]:


# 'density_sampling_WF0_same_sum_maj_100__new_freq',
# result_strings = ['density_sampling_WF0_same_sum_maj_100__new_freq', 'uncertainty_sampling_spambase, class ratio 1,n_queries 100 _freq']
# thresholds = [2,2]
# for i, string in enumerate(result_strings):
#    traj_freq_matrix = pd.read_pickle("../Results/" + string + ".pkl")
#    print(traj_freq_matrix)
#    plot_heatmap(traj_freq_matrix, threshold = thresholds[i], bin_size = 2, name = string, target_ids = [])
