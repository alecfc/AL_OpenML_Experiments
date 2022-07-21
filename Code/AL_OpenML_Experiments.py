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

# In[2]:
openml_list = openml.datasets.list_datasets()  # returns a dict

# Show a nice table with some key data properties
datalist = pd.DataFrame.from_dict(openml_list, orient="index")
datalist = datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

print(f"First 10 of {len(datalist)} datasets...")
datalist.head(n=10)

# The same can be done with lesser lines of code
openml_df = openml.datasets.list_datasets(output_format="dataframe")
openml_df.head(n=10)

ML_results_fully_trained = []


# In[3]:


# In[8]:


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


# In[12]:


# The base definitions of all test settings. These are set with new values when calling the run_test method (or set_settings method).

# Ideally will have at least 50 repetitions of validation of one method to get a good idea of performance. repetitions = K * EXECUTIONS
K = 5  # number of folds for validation
EXECUTIONS = 10  # number of executions of k-fold validation

# Number of queries per round of learning. The main testing loops in run_test could be adapted to not be static. E.g. until convergence, or some other stopping criterion. 
# However, for evaluation it is easier if all methods are tested with the same number of queries (requirement of Area Under the Learning Curve measure (ALC)).
# Performance is evaluated after each instance is queried and added to the training pool.
N_QUERIES = 45

# size of initialization set. Needs to be at least the # classes, but more if QBC used.
INITIALIZATION_SET = 10

# Whether to train/evaluate only on the summarized trajectories, ele all instances of each trajectory are used for training.
TRAIN_ON_SUM = True

# Whether separte classifiers should be used for querying and classifying (producer/consumer respectively)
# *note, for QBC should petty much always use separate model. Really doesn't make sense not to.
SEPARATE_CLASSIFIER = False

# Classifier to be used:
# 1 = logistic regression, 2 = Random Forest, 3 = XGBoost, 4 = Decision Tree, 5 = SVM (linear version)
ML_METHOD = 3
ML_SEP_METHOD = 3  # only relevant if using separate classifier as query producer. Overridden when using QBC

# Active learning approach to be used:
# 1 = random sampling, 2 = uncertainty sampling, 3 = density-weighted sampling, 4 = EER, 5 = QBC, 6 = PAL, 7 = XPAL
AL_METHOD = 6
EXP_TYPE = 'Class_Imbalance'
# Specific learners used for the committee in QBC
QBC_LEARNERS = [2, 3, 5]

# The number of learners of each classifier type specified by QBC_LEARNERS to be used in QBC
N_QBC_LEARNERS = 3

# Selection of QBC disagreement measure
# vote_entropy_sampling, consensus_entropy_sampling, max_disagreement_sampling
QBC_STRATEGY = max_disagreement_sampling

# Set whether to oversample, undersample, or neither. Should NOT have over- and undersampling both be True
OVERSAMPLE = False
UNDERSAMPLE = False
SAMPLING_RATIO = 0.25

# Weighing factor, (in favor of the target class)
WEIGHTING_FACTOR = 0.15

# Whether to scale data and apply PCA (PCA currently disabled -> commented out)
SC_PCA = False

# whether to save results from current test as figures
SAVEFIGS = False
SAVERATIO = False
SAVERESULTS = True


# In[13]:


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
    number_class_0 = int(round(prob_ratio * size_initial))
    number_class_1 = int(size_initial - number_class_0)
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


# x = np.array(['a','d','s','a','r','e','w','w','sd','as','rr','tt'])
# y=np.array([0,1,0,1,1,0,1,0,0,1,0,1])
# x, y = determine_initial_set(x, y, 0.2, 6)

def train(
    model,
    train_loader,
    optimizer,
    weighting_scheme,
    _run=None,
    for_acquisition=False
):
    """
    Helper function to execute a single epoch of training.
    """
    model.train()
    # avg_train_loss = 0
    raw_loss = torch.nn.NLLLoss()

    # losses = []
    for batch_idx, (data_N_C_H_W, target, weight) in enumerate(train_loader):
        data_N_C_H_W = data_N_C_H_W.cuda()
        target = target.cuda()
        weight = weight.cuda()

        optimizer.zero_grad()

        prediction = model(data_N_C_H_W)  # Always uses 1 when not doing consistent.


        loss = (weight * raw_loss(prediction, target)).mean(0)

        loss.backward()
        # avg_train_loss = (avg_train_loss * batch_idx + loss.item()) / (batch_idx + 1)
        optimizer.step()

        # losses.append(loss.item())
        # print(f'Epoch: {epoch}:')
    # print(f'Train Set: Average Loss: {avg_train_loss:.6f}')

def evaluate(model, eval_loader):
    """We actually only want eval mode on when we're doing acquisition because of how consistent dropout works.
    """
    model_arch = None
    n_samples = 8
    weighting_scheme = 'refined'
    model.train()

    nll = 0
    weighted_nll = 0
    correct = 0

    with torch.no_grad():
        for data_N_C_H_W, target_N, weight_N in eval_loader:
            data_N_C_H_W = data_N_C_H_W.cuda()
            target_N = target_N.cuda()
            weight_N = weight_N.cuda()

            samples_V_N = torch.stack(
                [model(data_N_C_H_W) for _ in range(n_samples)]
            )
            prediction_N = torch.logsumexp(samples_V_N, dim=0) - math.log(n_samples)

            raw_nll_N = F.nll_loss(prediction_N, target_N, reduction="none")
            nll += torch.sum(raw_nll_N)
            if weighting_scheme == "none":
                weighted_nll = 0.
            else:
                weighted_nll += torch.sum(weight_N * raw_nll_N)

            # get the index of the max log-probability
            class_prediction = prediction_N.max(1, keepdim=True)[1]
            correct += (
                class_prediction.eq(target_N.view_as(class_prediction)).sum().item()
            )

    nll /= len(eval_loader.dataset)
    if weighting_scheme == "none":
        pass
    else:
        weighted_nll /= len(eval_loader.dataset)
        weighted_nll = weighted_nll.item()
    percentage_correct = 100.0 * correct / len(eval_loader.dataset)

    return nll.item(), weighted_nll, percentage_correct

def train_to_convergence(
    model,
    train_loader,
    validation_loader,
    for_acquisition,
):
    """
    Helper function to train multiple epochs until the set limit is reached or we lose patience.
    """
    print(f"Beginning training with {len(train_loader.dataset)} training points and {len(validation_loader.dataset)} validation.")
    best = np.inf
    best_model = model
    patience = 0
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001,
    )
    print(f"Digits used: {len(train_loader.dataset)}")
    for epoch in range(100):
        train(
            model,
            train_loader,
            optimizer,
            'refined',
            _run=None,
            for_acquisition=for_acquisition
        )
        valid_nll, _, valid_accuracy = evaluate(
            model,
            validation_loader
        )
        # _run.log_scalar("evaluation_loss", valid_loss)
        # _run.log_scalar("evaluation_accuracy", valid_accuracy)
        print(
            f"Epoch {epoch:0>3d} eval: Val nll: {valid_nll:.4f}, Val Accuracy: {valid_accuracy}"
        )

        if valid_nll < best:
            best = valid_nll
            best_model = deepcopy(model)
            patience = 0
        else:
            patience += 1

        if patience >= 20:
            print(f"Patience reached - stopping training. Best was {best}")
            break
    print("Completed training", end="")

    print(".")
    return best_model


# In[23]:

def run_AL_test_plain_levelled(X, y, X_df, k_, execs_, n_queries_, initial_ratio_, initial_size_, save_results_, file_path_):
    accuracy_results = pd.DataFrame(columns=range(n_queries_ + 1))
    macro_f1_results = pd.DataFrame(columns=range(n_queries_ + 1))
    recall_results = pd.DataFrame(columns=range(n_queries_ + 1))
    precision_results = pd.DataFrame(columns=range(n_queries_ + 1))
    auc_results = pd.DataFrame(columns=range(n_queries_ + 1))
    loss_results = pd.DataFrame(columns=range(n_queries_ + 1))
    selected_labels_table = pd.DataFrame(columns=range(n_queries_))
    selected_instances_table = pd.DataFrame(columns=range(n_queries_))
    skf = StratifiedKFold(n_splits=K, shuffle=True)
    overall_weighted_risks = []
    overall_unweighted_risks = []
    overall_rb_risks = []
    overall_true_risks = []
    overall_refined_rb_risks = []
    overall_uniform_risks = []
    train_set_indices = np.array(list(skf.split(X, y)))

    for execs in range(execs_):
        weighted_risk = []
        # K-fold cross validation
        for i in range(k_):
            # sys.stdout.write("Validating on fold: ", i + 1, "/", K, end="\r")
            print('\r', "Validating on fold: ", i + 1, "/", k_)
            sys.stdout.flush()
            X_train = X[train_set_indices[i][0]]
            y_train = y[train_set_indices[i][0]]

            X_test = X[train_set_indices[i][1]]
            y_test = y[train_set_indices[i][1]]
            dataset = OpenMLDataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            temp_tensor = torch.from_numpy(X_train)
            model = ML_switcher.get(1)
            dataset.set_model(model)
            temp = np.expand_dims(dataset.pool_dataset[1], axis=1)
            model.fit(dataset.pool_dataset[1], dataset.pool_dataset[0])


            for idx in range(n_queries_):

                sampled_idx, probability_distribution = dataset.proposal_epsilon_greedy()
                dataset.acquire_point(sampled_idx, probability_distribution)
                # Estimate the risk of the fixed model using those points.
                if idx > 0:
                    weighted_risk.append(dataset.estimate_risk(weighting_scheme="refined"))
                #if (idx % 20 == 1) and execs == 1:
                #    dataset.plot_weights()
            overall_weighted_risks.append(weighted_risk)

    true_risk = dataset.true_risk()
    plot_risk(overall_weighted_risks, overall_unweighted_risks, overall_rb_risks, overall_refined_rb_risks,
              overall_uniform_risks, true_risk, execs_)

def plot_risk(overall_weighted_risks, overall_unweighted_risks, overall_rb_risks, overall_refined_rb_risks, overall_uniform_risks, true_risk, n_runs):
    c = sns.color_palette()
    def series(risks):
        overall_risks_array = np.array(risks)
        risk_mean = np.mean(overall_risks_array, axis=0)
        if overall_risks_array.shape[0] != 1:
            risk_std = np.std(overall_risks_array, axis=0)
            risk_se = risk_std / np.sqrt(overall_risks_array.shape[0] -1)
        else:
            risk_se = 0
        x = np.arange(len(risk_mean))
        return x, risk_mean, risk_se

    weighted_x, weighted_mean, weighted_se = series(overall_weighted_risks)
    plt.plot(weighted_x, weighted_mean, label="Weighted")
    plt.fill_between(weighted_x, weighted_mean + weighted_se, weighted_mean - weighted_se, alpha=0.3)

    # unweighted_x, unweighted_mean, unweighted_se = series(overall_unweighted_risks)
    # plt.plot(unweighted_x, unweighted_mean, label="Unweighted")
    # plt.fill_between(unweighted_x, unweighted_mean + unweighted_se, unweighted_mean - unweighted_se, alpha=0.3)

    rb_x, rb_mean, rb_se = series(overall_rb_risks)
    plt.plot(rb_x, rb_mean, label="Rao-Blackwell")
    plt.fill_between(rb_x, rb_mean + rb_se, rb_mean - rb_se, alpha=0.3)

    r_rb_x, r_rb_mean, r_rb_se = series(overall_refined_rb_risks)
    plt.plot(r_rb_x, r_rb_mean, label="Refined Rao-Blackwell")
    plt.fill_between(r_rb_x, r_rb_mean + r_rb_se, r_rb_mean - r_rb_se, alpha=0.3)

    # uniform_x, uniform_mean, uniform_se = series(overall_uniform_risks)
    # plt.plot(uniform_x, uniform_mean, label="Uniform")
    # plt.fill_between(uniform_x, uniform_mean + uniform_se, uniform_mean - uniform_se, alpha=0.3)

    plt.hlines(true_risk, 0, len(weighted_x), label="True Risk")
    plt.xlabel("Number of Sampled Points")
    plt.ylabel("Empirical Risk")
    plt.title("Empirical Risk Convergence Under Different Weighting Schemes")
    plt.legend()
    plt.savefig(f"plots/toy_fn_comparison_{n_runs}.pdf", bbox_inches="tight", dpi=300)

#@jit
def run_AL_test(X, y, X_df, k_, execs_, n_queries_, n_instantiations_, original_class_ratio_, initial_ratio_, initial_size_, ml_method_,
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
            #sys.stdout.write("Validating on fold: ", i + 1, "/", K, end="\r")
            print('\r',"Validating on fold: ", i + 1, "/", k_)
            sys.stdout.flush()
            X_train = X[train_set_indices[i][0]]
            y_train = y[train_set_indices[i][0]]

            X_test = X[train_set_indices[i][1]]
            y_test = y[train_set_indices[i][1]]

            for j in range(n_instantiations_):

                # X_train_copy = X_train.copy()
                # y_train_copy = y_train.copy()
                predictions = []
                x_initial, y_initial, X_temp, y_train, y_list = determine_initial_set(X_train, y_train,
                                                                                       initial_ratio_,
                                                                                       initial_size_)
                if al_method_ == 5 or al_method_ == 6:
                    ds = libact.base.dataset.Dataset(X_train, y_list)
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
                    model = libact.models.LogisticRegression(C=0.1)
                    sub_qs = UncertaintySampling(
                        dataset=ds, method='sm', model=libact.models.LogisticRegression(C=0.1))
                    learner = HierarchicalSampling(
                        dataset=ds, # Dataset object
                        classes=[0,1],
                        active_selecting=True,
                        subsample_qs=sub_qs
                    )
                    model.train(ds)
                    # Make first set of predictions (before querying)
                    prediction = model.predict(X_test)
                elif al_method_ == 6:
                    model = libact.models.LogisticRegression(C=0.1)
                    learner = QUIRE(
                        dataset=ds # Dataset object
                    )
                    model.train(ds)
                    # Make first set of predictions (before querying)
                    prediction = model.predict(X_test)
                elif al_method_ == 7:
                    weighted_dataset = OpenMLDataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_initial=x_initial, y_initial=y_initial)
                    weighted_dataset.set_model(model)
                    model.fit(weighted_dataset.acquired_dataset[1], weighted_dataset.acquired_dataset[0])
                    prediction = model.predict(X_test)
                else:
                    # print(x_initial)
                    # print(y_initial)
                    x_initial, y_initial, X_train, y_train, y_list = determine_initial_set(X_train, y_train, initial_ratio_,
                                                                                   initial_size_)
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
                    if al_method_ == 5 or al_method_ == 6:
                        query_idx = learner.make_query()
                        ds.update(query_idx, y_train[query_idx])
                        model.train(ds)
                        predictions = model.predict(X_test)
                    elif al_method_ == 7:
                        query_idx, probability_distribution = weighted_dataset.proposal_epsilon_greedy()
                        weighted_dataset.acquire_point(query_idx, probability_distribution)
                        model.fit(weighted_dataset.acquired_dataset[1], weighted_dataset.acquired_dataset[0])
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
    for exp_folder in os.scandir(filepath_):
        df_result = pd.read_pickle(exp_folder)
        if dataset_name_ in exp_folder.name:
            df_measure_results_ = df_measure_results_.append(df_result, ignore_index=True)
    return df_measure_results_


def aggregate_performance_measure_results(filepath_, agg_measure_results_):
    for exp_folder in os.scandir(filepath_):
        df_result = pd.read_pickle(exp_folder)
        agg_measure_results_ = agg_measure_results_.append(df_result, ignore_index=True)
    return agg_measure_results_


def read_dataset_results(base_filepath_, n_queries_, dataset_name_):
    performance_metrics = {'Accuracy': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'F1': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'Recall': pd.DataFrame(columns=range(n_queries_ + 1)),
                           'Precision': pd.DataFrame(columns=range(n_queries_ + 1)),
                           "Label Ratio": pd.DataFrame(columns=range(n_queries_)),
                           "AUC": pd.DataFrame(columns=range(n_queries_ + 1)),
                           "Loss Difference": pd.DataFrame(columns=range(n_queries_ + 1))}
    df_results = {}
    for exp_folder in os.scandir(base_filepath_):
        df_results[exp_folder.name] = performance_metrics.copy()
        for exp_type_result in os.scandir(exp_folder):
            df_results[exp_folder.name][exp_type_result.name] = dataset_performance_measure_results(
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
                                   dataset_name_):
    measure_results = []
    for subset, subset_results in all_dataset_results_.items():
        measure_results.append(all_dataset_results_[subset][measure_name_])
    plot_3d_results(measure_results, measure_name_, save_=True,
                    file_path_=file_path_,
                    z_labels_=setting_names_,
                    experiment_type_=experiment_type_, dataset_name_=dataset_name_)
    plot_multiple(measure_results, measure_name_, setting_names_=setting_names_,
                  experiment_type_=experiment_type_,
                  save_=True, file_path_=file_path_, dataset_name_=dataset.name)
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
                                       setting_names_=list(all_dataset_results.keys()), dataset_name_=dataset.name)


def al_run_single_dataset(X, y, ml_method, X_df, ML_results_fully_trained):
    # Iterate through all active learning methods
    #active_learning_methods = ['random_sampling', 'uncertainty_sampling', 'density_sampling', 'qbc_sampling',
    #                           'hierarchical_sampling', 'quire']
    active_learning_methods = []
    al_dict = AL_switcher2
    original_class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
    for al_method_number, al_method in al_dict.items():
        al_method_number = 6
        active_learning_methods.append(al_method.__name__)
        print('Evaluating', al_method.__name__, 'using the', type(ML_switcher[ml_method]).__name__, 'classifier')
        file_path = "../Figures/AL_Methods/" + dataset.name + '/' + al_method.__name__ + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if al_method_number == 4 or al_method_number == 6:
            execs = 5
        else:
            execs = 1
        start = time.ctime()
        run_AL_test(X, y, X_df, k_=5, execs_=execs, n_queries_=100,
                                n_instantiations_=1,original_class_ratio_=original_class_ratio,
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
    all_dataset_results = read_dataset_results('../Results/AL_Methods/', 100, dataset.name)

    for measure_name, measure_results in all_dataset_results[active_learning_methods[0]].items():
        compare_results_single_dataset(all_dataset_results_=all_dataset_results,
                                       file_path_="../Figures/AL_Methods/" + dataset.name + '/',
                                       measure_name_=measure_name, experiment_type_='AL Methods',
                                       setting_names_=list(all_dataset_results.keys()), dataset_name_=dataset.name)

# In[26]:


def ml_run_single_dataset(X, y, al_method, X_df, ML_results_fully_trained):
    # Perform the active learning querying cycle for all different machine learning classifiers on a single dataset
    ml_method_numbers = [1,2,3]
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
                    n_instantiations_=1,original_class_ratio_=original_class_ratio,
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
                                       setting_names_=ml_methods, dataset_name_=dataset.name)

def init_run_single_dataset(X, y, al_method, ml_method, X_df, ML_results_fully_trained):
    # Iterate through all active learning methods
    auc_list = []
    ratio_list = []
    class_ratios=[0.1,0.5,0.25]
    full_class_ratios = ['Initial class ratio 0.1', 'Initial class ratio 0.5', 'Initial class ratio 0.25']
    init_ratios = []
    original_class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
    for idx, init_ratio in enumerate(class_ratios):
        init_ratios.append('Initial class ratio ' + str(init_ratio))
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
                                       setting_names_=full_class_ratios, dataset_name_=dataset.name)


# In[27]:

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


# In[28]:

def create_openML_subsamples(X, y):
    rus = RandomUnderSampler(random_state=42, sampling_strategy=ratio_multiplier(y=y, ratio=1))
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
    performance_results = []
    n_queries = 100
    ml_method = 1
    al_method = 3
    global dataset
    global EXP_TYPE
    EXP_TYPE = experiment_

    for idx, dataset_name in enumerate(dataset_list):
        dataset = openml.datasets.get_dataset(dataset_name)
        print('Running experiments on', dataset.name)
        X_df, y_df, X, y, number_majority, number_minority = preprocess_openML_dataset(dataset)
        X_list, y_list = create_openML_subsamples(X, y)
        X_train, X_test, y_train, y_test, ML_results_fully_trained, ML_results_subsample_trained, ML_results_subsample_trained_biased = evaluate_all_models(
            X, y, X_list, y_list)
        # run_experiments_openML_dataset(X_train, y_train, X_list=X_list,y_list=y_list)
        if experiment_ == 'Class_Imbalance':
            ci_run_single_dataset(X_list, y_list, al_method, ml_method, X_df,
                                  ML_results_fully_trained)
        elif experiment_ == 'AL_Methods':
            al_run_single_dataset(X, y, ml_method, X_df,
                                  ML_results_fully_trained)
        elif experiment_ == 'ML_Methods':
            ml_run_single_dataset(X, y, al_method, X_df,
                                  ML_results_fully_trained)
        if experiment_ == 'Initial_Class_Ratio':
            init_run_single_dataset(X, y, al_method, ml_method, X_df, ML_results_fully_trained)

    agg_measure_results = read_aggregate_results('../Results/'+experiment_+'/', n_queries)
    plot_aggregate_results(experiment_, agg_measure_results)
    plot_aggregate_comparison(experiment_, agg_measure_results)

    # for
    #    plot_aggregate_graphs()


if __name__ == "__main__":
    # This is done based on the dataset name 'cylinder-bands',
    # amount of instances per dataset [cylinder-bands:540, monks-problems-3:554, qsar-biodeg:1055, banknote-authentication:1372, steel-plates-fault:1941, scene:2407,
    # ozone-level-8hr:2534, kr-vs-kp:3196, Bioresponse:3751, wilt:4839, churn:5000, spambase:4601, mushroom:8124, PhishingWebsites:11055, electricity:45300, creditcard:285000]
    #'monks-problems-3', 'qsar-biodeg', 'hill-valley', 'banknote-authentication', 'steel-plates-fault', 'jasmine', 'scene', 'ozone-level-8hr',
     #'kr-vs-kp', 'Bioresponse','spambase', 'wilt',
    # 'monks-problems-3', 'qsar-biodeg', 'hill-valley', 'banknote-authentication', 'steel-plates-fault', 'jasmine', 'scene', 'ozone-level-8hr',
    #      'kr-vs-kp', 'Bioresponse','spambase', 'wilt','churn',
    dataset_list = ['mushroom', 'PhishingWebsites']
    #dataset_list = ['churn']  # ,'ringnorm', 'mushroom']#, 'electricity', 'creditcard']
    # agg_measure_results = read_aggregate_results('../Results/Initial_Class_Ratio/', 100)
    # plot_aggregate_results('Initial_Class_Ratio', agg_measure_results)
    # plot_aggregate_comparison('Initial_Class_Ratio', agg_measure_results)

    #run_openML_test(experiment_='Class_Imbalance')
    run_openML_test(experiment_='AL_Methods')
    #run_openML_test(experiment_='ML_Methods')
    #run_openML_test(experiment_='Initial_Class_Ratio')
    #agg_measure_results = read_aggregate_results('../Results/AL_Methods/', 100)
    #plot_aggregate_results('AL_Methods', agg_measure_results)
    #plot_aggregate_comparison('AL_Methods', agg_measure_results)
    #for idx, dataset_name in enumerate(dataset_list):
    #    dataset = openml.datasets.get_dataset(dataset_name)
    #    all_dataset_results = read_dataset_results('../Results/Class_Imbalance/', 100, dataset.name)



    #for idx, dataset_name in enumerate(dataset_list):
    #    dataset = openml.datasets.get_dataset(dataset_name)
    #    X_df, y_df, X, y, number_majority, number_minority = preprocess_openML_dataset(dataset)
    #    label_ratio_df = pd.DataFrame(columns=range(100))

    #    class_ratio = round(Counter(y)[1] / (Counter(y)[0] + Counter(y)[1]), 3)
    #    dataset_str = dataset.name + ' class ratio ' + str(class_ratio)
    #    for al_method_number, al_method in AL_switcher.items():
    #        label_ratio_df = dataset_performance_measure_results('../Results/AL_Methods/' + al_method.__name__ + '/Label Ratio/', label_ratio_df, dataset.name)
    #        al_str = al_method.__name__ + ' initial size ' + str(
    #            10) + ' initial ratio ' + str(
    #            0.5)
    #        ml_str = type(ML_switcher[2]).__name__
    #        string = dataset_str + ' ' + al_str + ' ' + ml_str + ' for ' + str(100) + ' queries'
    #        plot_class_per_sample(labels_=label_ratio_df, original_class_ratio_=class_ratio, name_=string, save_=True,
    #                              file_path_='../Figures/AL_Methods/'+dataset.name + '/' + al_method.__name__ + '/', dataset_name_=dataset.name)

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
