# from pip command

import os, sys;

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sn
from numpy import trapz

from AL_plot_results import *

from AL_methods import *


# from pip command
def init_set_generator(prob_ratio, size_initial):
    number_class_1 = int(round(prob_ratio * size_initial))
    number_class_0 = int(size_initial - number_class_1)
    list_zeroes = np.zeros((number_class_0,), dtype=int)
    list_ones = np.ones((number_class_1,), dtype=int)
    return list(list_zeroes) + list(list_ones)

# Method for plotting the saved results as figures.
def plot_results(X, results_, measure_name_, ML_results_fully_trained_, name_, al_method_, ml_method_, save_=False,
                 normalize_data_=False,
                 prop_performance_=False, file_path_='../Figures/', data_title_='', al_dict_=AL_switcher):
    mean_performance = results_.describe().loc[['mean'], :].to_numpy()[0]
    lower_quartiles = results_.describe().loc[['25%'], :].to_numpy()[0]
    upper_quartiles = results_.describe().loc[['75%'], :].to_numpy()[0]
    medians = results_.describe().loc[['50%'], :].to_numpy()[0]
    sn.set_theme()
    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    if normalize_data_:
        num_rows, num_cols = X.shape
        plot_range = np.arange(len(mean_performance)) / num_rows
        ax.set_xlabel('Query iteration/dataset size')
    else:
        ax.set_xlabel('Query iteration')
        plot_range = range(len(mean_performance))
        if measure_name_ == "Loss Difference":
            ax.plot(mean_performance)
            quartiles = list(zip(lower_quartiles, upper_quartiles))
            plt.plot((plot_range, plot_range), ([i for (i, j) in quartiles], [j for (i, j) in quartiles]), c='black')
            plt.plot(plot_range, [i for (i, j) in quartiles], '_', markersize=6, c='blue')
            plt.plot(plot_range, [j for (i, j) in quartiles], '_', markersize=6, c='blue')
            ax.set_xlabel('Query iteration')
            ax.set_ylabel('Active learning bias through difference in estimated risk')
            ax.set_title(
                data_title_ + ': Bias Through Difference in Risk with Fully Trained ' + type(ML_switcher[ml_method_]).__name__ + ' Classifier')
            if save_:
                string = file_path_ + name_ + '_' + '.png'
                plt.savefig(string, bbox_inches='tight')
            return
        if measure_name_ == "Label Ratio":
            ax.plot(mean_performance)
        if prop_performance_ and measure_name_ != "Label Ratio":
            result_dict = ML_results_fully_trained_[ml_method_]
            mean_performance = result_dict[measure_name_] / mean_performance
            lower_quartiles = result_dict[measure_name_] / lower_quartiles
            upper_quartiles = result_dict[measure_name_] / upper_quartiles
            plot_range = range(len(mean_performance))
            ax.plot(mean_performance)
            ax.set_ylabel('Propotional to fully trained classifier ' + measure_name_)
            quartiles = list(zip(lower_quartiles, upper_quartiles))
            plt.plot((plot_range, plot_range), ([i for (i, j) in quartiles], [j for (i, j) in quartiles]), c='black')
            plt.plot(plot_range, [i for (i, j) in quartiles], '_', markersize=6, c='blue')
            plt.plot(plot_range, [j for (i, j) in quartiles], '_', markersize=6, c='blue')
            #plt.show()
            return
    ax.set_xlabel('Query iteration')
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    quartiles = list(zip(lower_quartiles, upper_quartiles))

    ax.scatter(plot_range, mean_performance, s=13, c="blue")

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))

    if measure_name_ == "Label Ratio":
        ax.set_ylim(bottom=0, top=1, auto=False)
    else:
        ax.set_ylim(bottom=0.4, top=1, auto=False)  # bottom=0, top=1
    ax.grid(True)

    ax.set_title(data_title_ + ' Incremental Classification ' + measure_name_ + ": " + al_dict_[
        al_method_].__name__ + ' using ' + type(ML_switcher[ml_method_]).__name__ + ' classifier')  # change later

    ax.set_ylabel('Classification ' + measure_name_)
    #     ax.legend(loc="lower right")

    if measure_name_ == 'AUC':
        x = plot_range
        x = x[1:]
        y = mean_performance
        rand_predict = mean_performance[0]
        y = mean_performance[1:]

        x = np.log2(x)
        A = trapz(y)
        Arand = rand_predict * x[-1]
        Amax = x[-1]

        global_score = (A - Arand) / (Amax - Arand)

        print("ALC is: ", global_score)

        if save_:
            string = file_path_ + name_ + "_" + "ALC" + ".txt"
            text_file = open(string, "w")
            n = text_file.write(str(global_score))
            text_file.close()

    if measure_name_ != "Label Ratio":
        plt.plot((plot_range, plot_range), ([i for (i, j) in quartiles], [j for (i, j) in quartiles]), c='black')
        plt.plot(plot_range, [i for (i, j) in quartiles], '_', markersize=6, c='blue')
        plt.plot(plot_range, [j for (i, j) in quartiles], '_', markersize=6, c='blue')

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                        hspace=0.4, wspace=0.3)
    if save_:
        string = file_path_ + name_ + '.png'
        plt.savefig(string, bbox_inches='tight')
    # plt.show()


# In[15]:


def plot_3d_results(results_, metric_name_, save_, file_path_, z_labels_, experiment_type_, dataset_name_):
    sn.set_theme()
    data = []
    title = '3D Comparison of ' + metric_name_ + ' for the ' + experiment_type_ + ' experiment on the ' + dataset_name_ + ' dataset'

    for idx, result in enumerate(results_):
        mean_performance = result.describe().loc[['mean'], :].to_numpy()[0]
        categories = "50-50 " * len(mean_performance)
        categories = categories.split(" ")
        categories.pop(len(mean_performance))
        plot_range = range(len(mean_performance))

        df = pd.DataFrame({
            'cat': categories, 'Number of Queries': plot_range, 'Performance': mean_performance
        })
        df['Experiment Category'] = z_labels_[idx]

        trace = go.Scatter3d(x=df['Number of Queries'],
                             y=df['Experiment Category'],
                             z=df['Performance'],
                             name=z_labels_[idx],
                             mode='markers')
        data.append(trace)

    # style layout
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='Number of Queries',
            yaxis_title='Class Ratio',
            zaxis_title=metric_name_,
        ),
    )

    fig = go.Figure(layout=layout, data=data)
    if save_:
        string = file_path_ + title + '.png'
        fig.write_image(string)
    # fig.show()


# In[16]:


def plot_multiple(results_, metric_name_, setting_names_, experiment_type_, save_, file_path_, dataset_name_):
    sn.set_theme()
    data = []
    colors = ['#B42B2D', '#0E84FA', '#FAAB0E', '#121110', '#249338', '#894CB6']
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    title = "Comparison of " + metric_name_ + " for the " + experiment_type_ + ' experiment on ' + dataset_name_
    ax.set_title(title)
    ax.set_ylabel('Classification ' + metric_name_)
    ax.set_xlabel('Query iteration')
    if metric_name_ == "Label Ratio":
        ax.set_ylim(bottom=0, top=1, auto=False)
    else:
        if metric_name_ != "Loss Difference":
            ax.set_ylim(bottom=0.4, top=1, auto=False)  # bottom=0, top=1
    ax.grid(True)

    if metric_name_ != "Loss Difference":
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))

    for idx, result in enumerate(results_):
        mean_performance = result.describe().loc[['mean'], :].to_numpy()[0]
        lower_quartiles = result.describe().loc[['25%'], :].to_numpy()[0]
        upper_quartiles = result.describe().loc[['75%'], :].to_numpy()[0]

        if metric_name_ != "Label Ratio":
            for j, quartile_result in enumerate(lower_quartiles):
                if j % 5 != 0:
                    upper_quartiles[j] = mean_performance[j]
                    lower_quartiles[j] = mean_performance[j]

        medians = result.describe().loc[['50%'], :].to_numpy()[0]
        plot_range = range(len(mean_performance))
        quartiles = list(zip(lower_quartiles, upper_quartiles))
        ax.plot(mean_performance, c=colors[idx])
        ax.scatter(plot_range, mean_performance, s=13, c=colors[idx], label=setting_names_[idx])
        if metric_name_ != "Label Ratio":
            plt.plot((plot_range, plot_range), ([i for (i, j) in quartiles], [j for (i, j) in quartiles]), c=colors[idx])
            plt.plot(plot_range, [i for (i, j) in quartiles], '_', markersize=6, c=colors[idx])
            plt.plot(plot_range, [j for (i, j) in quartiles], '_', markersize=6, c=colors[idx])

    if metric_name_ == "Label Ratio" or metric_name_ == "Loss Difference":
        ax.legend(loc='upper right')
    else:
        ax.legend(loc='lower right')
    if save_:
        string = file_path_ + title + '.png'
        plt.savefig(string, bbox_inches='tight')
    # fig.show()


# In[17]:


def plot_risk_difference(results_, loss_fully_trained_, model_name_, save_, file_path_, name_, dataset_name_):
    # Plot bias through difference of estimated risk (loss)
    sn.set_theme()
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    mean_performance = results_.describe().loc[['mean'], :].to_numpy()[0]
    lower_quartiles = results_.describe().loc[['25%'], :].to_numpy()[0]
    upper_quartiles = results_.describe().loc[['75%'], :].to_numpy()[0]
    medians = results_.describe().loc[['50%'], :].to_numpy()[0]
    plot_range = range(len(mean_performance))
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Active learning bias through difference in estimated risk')
    mean_performance = loss_fully_trained_ - mean_performance
    lower_quartiles = loss_fully_trained_ - lower_quartiles
    upper_quartiles = loss_fully_trained_ - upper_quartiles
    ax.plot(mean_performance)
    quartiles = list(zip(lower_quartiles, upper_quartiles))
    ax.set_title(dataset_name_ + ': Bias Through Difference in Risk with Fully Trained ' + model_name_ + ' Classifier')
    plt.plot((plot_range, plot_range), ([i for (i, j) in quartiles], [j for (i, j) in quartiles]), c='black')
    plt.plot(plot_range, [i for (i, j) in quartiles], '_', markersize=6, c='blue')
    plt.plot(plot_range, [j for (i, j) in quartiles], '_', markersize=6, c='blue')

    if save_:
        string = file_path_ + name_ + '.png'
        plt.savefig(string, bbox_inches='tight')
    # plt.show()


# In[18]:
def plot_multiple_bias(initial_labels_, labels_, original_class_ratio_, save_, file_path_, setting_names_, experiment_type_, dataset_name_):
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    sn.set_theme()
    colors = ['#B42B2D', '#0E84FA', '#FAAB0E', '#121110', '#249338', '#894CB6']
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Active learning bias through class ratio difference')
    title = "Comparison of Bias Through Class Ratio Difference Using Original Ratio of " + str(original_class_ratio_) \
            + " for the " + experiment_type_ + " experiment on " + dataset_name_
    if experiment_type_ == 'Class Imbalance':
        class_ratios = [0.25, 0.05, 0.5, original_class_ratio_]
        original_class_ratio_ = class_ratios
    elif experiment_type_ == 'Initial Class Ratio':
        init_ratios = [0.1, 0.5, 0.25]
    ax.set_title(
        dataset_name_ + ': Comparison of Bias Through Class Ratio Difference Using Original Ratio of ' + str(original_class_ratio_))
    for idx, label in enumerate(labels_):
        top_selected_labels = label.mode()
        plot_range = range(len(label.columns))
        ratio_differences = []
        if experiment_type_ == 'Initial Class Ratio':
            initial_labels_ = init_set_generator(init_ratios[idx],10)
        current_labels = initial_labels_
        if experiment_type_ == 'Class Imbalance':
            original_class_ratio_ = class_ratios[idx]
        for i in top_selected_labels.to_numpy()[0]:
            current_labels = np.append(current_labels, [i])
            updated_class_ratio = round(
                Counter(current_labels)[1] / (Counter(current_labels)[0] + Counter(current_labels)[1]), 2)
            ratio_differences.append(original_class_ratio_ - updated_class_ratio)
        plt.plot(plot_range, [i for i in ratio_differences], c=colors[idx], label=setting_names_[idx])
    plt.legend(loc="lower right")
    if save_:
        string = file_path_ + title + '.png'
        plt.savefig(string, bbox_inches='tight')

def plot_bias(initial_labels_, labels_, original_class_ratio_, save_, file_path_, name_, dataset_name_):
    # Plot our bias difference over time, through calculating the difference between class ratio's
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    sn.set_theme()
    top_selected_labels = labels_.mode()
    plot_range = range(len(labels_.columns))
    ratio_differences = []
    current_labels = initial_labels_
    for label in top_selected_labels.to_numpy()[0]:
        current_labels = np.append(current_labels, [label])
        updated_class_ratio = round(
            Counter(current_labels)[1] / (Counter(current_labels)[0] + Counter(current_labels)[1]), 2)
        ratio_differences.append(original_class_ratio_ - updated_class_ratio)

    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Active learning bias through class ratio difference')
    ax.set_title(
        dataset_name_ + ': Bias Through Class Ratio Difference Using Original Ratio of ' + str(original_class_ratio_))

    plt.plot(plot_range, [i for i in ratio_differences], c='blue')
    if save_:
        string = file_path_ + name_ + '.png'
        plt.savefig(string, bbox_inches='tight')
    # plt.show()

def plot_class_per_sample(labels_, original_class_ratio_, save_, name_, file_path_, dataset_name_):
    sn.set_theme()
    proportion_per_query = []
    num_zeroes = []
    num_ones =[]
    total_per_exec = len(labels_[0])
    top_selected_labels = labels_.mode().to_numpy()[0]
    for (columnName, columnData) in labels_.iteritems():
        num_zeroes.append(columnData.loc[columnData < 1].count()/total_per_exec)
        num_ones.append(columnData.loc[columnData == 1].count()/total_per_exec)
    df = pd.DataFrame(list(zip(num_zeroes, num_ones)), index=range(1,len(labels_.T[0])+1), columns=['Negative Class', 'Positive Class'], )
    (df*100).plot.bar(title=dataset_name_ + ': Proportion of Selected Classes per Query', stacked=True, figsize=(18, 6))
    plt.legend(loc='upper right')
    plt.xlabel('Query Iteration')
    plt.ylabel('Percentage of Chosen Classes')
    if save_:
        string = file_path_ + name_ + '.png'
        plt.savefig(string, bbox_inches='tight')

def plot_top_selected_instances(instances, labels, save_, file_path_, name_):
    all_instances = []
    for column in instances:
        query_iteration_instances = instances[column].tolist()
        all_instances += query_iteration_instances

    all_labels = []
    for column in labels:
        query_iteration_labels = labels[column].tolist()
        all_labels += query_iteration_labels
    df_instances = pd.DataFrame(all_instances)
    df_instances['Label'] = pd.DataFrame(all_labels).round()
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    instance_frequencies = df_instances[0].value_counts().to_frame()
    instance_frequencies.columns = ['Amount of times queried']
    instance_frequencies['Instance number'] = instance_frequencies.index
    instance_frequencies['Label'] = np.nan
    instance_frequencies.reset_index()
    for idx, instance in enumerate(instance_frequencies['Instance number']):
        label_index = df_instances.index[df_instances[0] == instance].tolist()[0]
        instance_frequencies['Label'].iloc[idx] = df_instances['Label'].iloc[label_index]
    instance_frequencies['Instance number'] = instance_frequencies['Instance number'].round()
    instance_frequencies = instance_frequencies.astype({'Instance number': 'int'})
    sn.set_theme()
    ax = sn.barplot(x='Instance number', y="Amount of times queried", hue="Label", data=instance_frequencies.head(15))
    if save_:
        string = file_path_ + name_ + '_' + '.png'
        plt.savefig(string, bbox_inches='tight')
    # plt.show()


# In[20]:


def plot_aggregate_results(experiment_name, aggregate_results, al_method_, ml_method_, al_switcher_):
    print('Aggregate Results:')
    for experiment_type, experiment_results in aggregate_results.items():
        experiment_title = experiment_type
        if experiment_name == 'AL_Methods':
            al_method_ =  [k for k, v in al_switcher_.items() if v.__name__ == experiment_title][0]
        if experiment_name == 'ML_Methods':
            ml_method_ =  [k for k, v in ML_switcher.items() if type(v).__name__ == experiment_title][0]
        file_path = "../Figures/" + experiment_name + "/Aggregate_Results/" + experiment_title + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for performance_metric_name, performance in aggregate_results[experiment_type].items():
            if experiment_name == 'Class_Imbalance':
                file_name = 'All Datasets Aggregate ' + performance_metric_name + ' Results for ' + experiment_title + ' Class Ratio'
            else:
                file_name = 'All Datasets Aggregate ' + performance_metric_name + ' Results for ' + experiment_title
            plot_results([], performance, performance_metric_name, False, file_name, al_method_, ml_method_, save_=True,
                         normalize_data_=False, prop_performance_=False, file_path_=file_path,
                         data_title_='Aggregate OpenML', al_dict_=al_switcher_)


# In[21]:


def plot_aggregate_comparison(experiment_name, aggregate_ci_results):
    print('Comparison of Experiment Settings on Aggregate')
    aggregate_list = list(aggregate_ci_results)
    stored_performance = aggregate_ci_results[aggregate_list[0]].items()
    methods = []
    if experiment_name == 'AL_Methods':
        for idx, subset_number in enumerate(aggregate_list):
            methods.append(subset_number)
        aggregate_list = methods
    elif experiment_name == 'ML_Methods':
        for idx, subset_number in enumerate(aggregate_list):
            methods.append(subset_number)
        aggregate_list = methods
    for performance_metric_name, performance in stored_performance:
        file_path = "../Figures/" + experiment_name + "/Aggregate_Results/" + performance_metric_name + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        results_for_performance_metric = []
        for experiment_type, experiment_results in aggregate_ci_results.items():
            for experiment_performance_name, experiment_performance in aggregate_ci_results[experiment_type].items():
                if experiment_performance_name == performance_metric_name:
                    results_for_performance_metric.append(experiment_performance)
        plot_multiple(results_for_performance_metric, performance_metric_name, setting_names_=aggregate_list,
                      experiment_type_=experiment_name,
                      save_=True, file_path_=file_path, dataset_name_='aggregate')


# In[ ]:


# Plot frequency selection heatmap from given dataframe. The matrix is the dataframe containing the frequencies for all trajectories
# The threshold gives the minimum frequency required for a trajectory to be included in the heatmap. Bin size defines the size of bins in the heatmap.
# The name is the name of the file given to the produced file, typically being similar to the name of the file from which the frequencies were taken.
# Target ids is the list of trajectory ids for positive instances, which allows identification of which trajectories are positive or negative.
def plot_heatmap(matrix_, threshold, bin_size, name, target_ids: np.array):
    # Remove trips from matrix for which, at no time point, they have been selected at least threshold amount of times
    matrix_ = matrix_.loc[:, (matrix_ >= threshold).any(axis=0)]
    print(matrix_)
    # Could divide numbers by total number of runs (executions+folds(-1?) to get better scaled numbers
    # Might not be necessary, considering this is mostly used for finding interesting cases for one method at a time.

    # Divide the remaining frequencies into bins
    matrix = matrix_.T
    matrix = matrix.groupby([[i // bin_size for i in range(0, matrix.shape[1])]], axis=1).sum()
    matrix.columns = [str(i * bin_size + 1) + "-" + str(i * bin_size + bin_size) for i in
                      range(0, int(matrix.shape[1]))]
    matrix = matrix.T

    x_axis_labels = matrix_.columns  # labels for x-axis
    y_axis_labels = matrix.columns  # labels for y-axis

    plt.figure(figsize=(0.6 * matrix.shape[1], 0.6 * matrix.shape[0]))
    ax = plt.axes()
    ax.set_title('Instance Selection Frequencies', fontsize=14, fontweight='bold')

    ax.xticklabels = x_axis_labels
    ax.yticklabels = y_axis_labels

    sn.heatmap(matrix, annot=False, cmap="Reds", fmt='g', ax=ax, square=False)
    ax.invert_yaxis()

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Make all positive labels green, and negative labels red
    for lab in ax.get_xticklabels():
        text = lab.get_text()
        if text in str(target_ids):
            lab.set_color('green')
        else:
            lab.set_color('red')

    plt.savefig("../Figures/" + name + "_heat.png", bbox_inches='tight')
    plt.show()
