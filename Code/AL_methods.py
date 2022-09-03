from sklearn.exceptions import NotFittedError

import math
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import xgboost
from modAL.density import information_density
from modAL.uncertainty import uncertainty_sampling
from modAL.utils.selection import shuffled_argmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression


# from pip command

def ratio_multiplier(y, ratio):
    target_stats = Counter(y)
    multiplier = (1-ratio)/ratio # 0,3 or 19
    # try undersampling majority class
    new_maj_instances = math.ceil(target_stats[1] * multiplier)
    #new_maj_instances = math.ceil(target_stats[1] / ratio)
    new_min_instances = target_stats[1]
    if new_maj_instances > target_stats[0]:
        # try undersampling minority class
        new_min_instances = math.ceil(target_stats[0] / multiplier)
        new_maj_instances = target_stats[0]
    target_stats[0] = new_maj_instances
    target_stats[1] = new_min_instances
    return target_stats


# In[4]:
class LinearRegressionModel(nn.Module):
    def __init__(self, x_shape):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(x_shape, 2)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


# Setting up a random sampling strategy given a classifier and a candidate pool of instances to be sampled.
def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]


def hierarchical_sampling(classifier, X_pool):
    return 0, 0

# In[5]:
def quire(classifier, X_pool):
    return 0, 0

def albl(classifier, X_pool):
    return 0, 0

# Setting up a density-weighted sampling method given a classifier and a candidate pool of instances to be sampled.
# By default only one instance is sampled. Multi-instance sampling not supported.
def density_sampling(classifier, X_pool, n_instances: int = 1, **predict_proba_kwargs):
    density = information_density(X_pool, "cosine")  # cosine or euclidean
    density = np.ones(len(density)) - density

    try:
        classwise_uncertainty = classifier.predict_proba(X_pool, **predict_proba_kwargs)
    except NotFittedError:
        return np.ones(shape=(X_pool.shape[0],))

    # for each point, select the maximum uncertainty
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
    dense_informative = np.multiply(uncertainty, density)

    query_idx = shuffled_argmax(dense_informative, n_instances=n_instances)
    return query_idx, X_pool[query_idx]


# In[6]:


# Empty method which allows for the creation of committee for QBC sampling while using the same style of AL selection as other methods.
def qbc_sampling(classifier, X_pool):
    return 0, 0

def r_lure(classifier, X_pool):
    return 0, 0

def r_pure(classifier, X_pool):
    return 0, 0

# In[7]:


# Switcher for selecting the desired classifier
ML_switcher = {
    1: LogisticRegression(solver='liblinear', n_jobs=-1),
    2: xgboost.XGBClassifier(booster='dart',tree_method='hist'),
    3: RandomForestClassifier()
}

# Switcher for selecting the desired AL method
AL_switcher = {
    1: random_sampling,
    2: uncertainty_sampling,
    3: density_sampling,
    4: qbc_sampling
}

AL_switcher2 = {
    #1: random_sampling,
    #2: uncertainty_sampling,
    #3: density_sampling,
    5: hierarchical_sampling,
    6: quire,
    #7: r_lure,
    #8: r_pure,
    9: albl
}