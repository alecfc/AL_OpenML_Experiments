import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split, DataLoader, Sampler
from PIL import Image
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from AL_utils import get_top_n
import collections


class Dataset:
    def __init__(self, pool_dataset, validation_dataset, test_dataset, true_function, acquired_dataset):
        """
    Expects to be passed list of numpy arrays with (y, X, zeros)
    """
        self.pool_dataset = pool_dataset  # Unlabelled data
        self.starting_size = len(pool_dataset[0])  # Number of points to randomly acquire at start
        self.acquired_dataset = acquired_dataset  # Labelled data
        self.acquired_size = len(acquired_dataset[0])  # Number points in labelled data set
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.true_function = true_function
        self.model = None

    def acquire_point(self, acquisition_idx, probability_distribution):
        """
    Moves the newly acquired point from the pool_dataset to acquisition dataset
    """
        probability_mass = probability_distribution[acquisition_idx]
        # Update the pool_distribution probability masses, which are only used for the RB estimator
        ##self.pool_dataset[2] = (
        #    self.pool_dataset[2] * self.acquired_size + probability_distribution
        # ) / (self.acquired_size + 1)

        pool_size = len(self.pool_dataset[0])
        if self.acquired_dataset is None:
            # acquired dataset is [y, X, proposal when picked, average proposal, normalizing for RB]
            self.acquired_dataset = [
                self.pool_dataset[0][acquisition_idx],
                self.pool_dataset[1][acquisition_idx],
                probability_mass,
                probability_mass,
                1.0,
            ]
        else:
            self.acquired_dataset[0] = np.append(
                self.acquired_dataset[0], self.pool_dataset[0][acquisition_idx]
            )
            self.acquired_dataset[1] = np.vstack([self.acquired_dataset[1], self.pool_dataset[1][acquisition_idx]])
            self.acquired_dataset[2] = np.append(
                self.acquired_dataset[2], probability_mass
            )
            # This one is only used for the RB estimator
            self.acquired_dataset[3] = np.append(
                self.acquired_dataset[3], self.pool_dataset[2][acquisition_idx]
            )
            self.acquired_dataset[4] = np.append(
                self.acquired_dataset[4], np.sum(self.pool_dataset[2])
            )
        self.pool_dataset[0] = np.delete(self.pool_dataset[0], acquisition_idx, axis=0)
        self.pool_dataset[1] = np.delete(self.pool_dataset[1], acquisition_idx, axis=0)
        self.pool_dataset[2] = np.delete(self.pool_dataset[2], acquisition_idx, axis=0)
        self.acquired_size += 1
        assert len(self.acquired_dataset[0]) == self.acquired_size
        assert len(self.pool_dataset[0]) == pool_size - 1

    def plot_datasets(self):
        c = sns.color_palette()
        plt.scatter(
            self.pool_dataset[1],
            self.pool_dataset[0],
            color=c[1],
            marker="*",
            label="Pool Data",
        )
        if self.acquired_dataset is not None:
            plt.scatter(
                self.acquired_dataset[1],
                self.acquired_dataset[0],
                color=c[2],
                marker="x",
                label="Acquired Data",
            )
        if self.true_function is not None:
            plt.plot(
                self.true_function[1], self.true_function[0], label="True Function"
            )
        if self.model is not None:
            plt.plot(
                self.true_function[1],
                self.model.predict(np.expand_dims(self.true_function[1], axis=1)),
                color=c[3],
                label="Model predictions",
            )
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Data Visualization")
        plt.legend()
        plt.show()

    def plot_probabilities(self):
        num_points = len(self.acquired_dataset[2])
        plt.plot(
            np.arange(num_points), self.acquired_dataset[2], label=r"$q(i_m;i_{0:m-1})$"
        )
        plt.legend()
        plt.xlabel("m")
        plt.ylabel("Probability mass")
        plt.show()

    def plot_weights(self):
        num_points = len(self.acquired_dataset[2])
        weights = self.refined_rao_blackwelised_weighting_scheme()
        plt.plot(np.arange(num_points), weights, label="Weight")
        plt.legend()
        plt.xlabel("m")
        plt.ylabel(r"$v_m$")
        plt.show()

    def set_model(self, model):
        self.model = model

    def proposal_distribution_uniform(self):
        n_pooled_points = len(self.pool_dataset[1])
        sampled_idx = np.random.randint(0, n_pooled_points)
        probability_distribution = np.ones(n_pooled_points) * 1.0 / n_pooled_points

        # print(f"Of {n_pooled_points} points we chose {sampled_idx} with probability {probability_mass}.")
        return (np.array([sampled_idx]),), probability_distribution

    def proposal_distance_based(self):
        x = self.pool_dataset[1]
        if self.acquired_dataset is not None:
            x_train = self.acquired_dataset[1]
            distances = np.zeros_like(self.pool_dataset[0])
            for acquired_x in x_train:
                distances += np.sqrt((acquired_x - x) ** 2)
            # normalization_constant = np.sum(distances)
            # probability_masses = distances / normalization_constant
            probability_masses = np.exp(distances) / sum(np.exp(distances))
        else:
            probability_masses = np.ones_like(self.pool_dataset[1])
            probability_masses /= np.sum(probability_masses)
        sampled_idx = np.argmax(np.random.multinomial(1, probability_masses))
        return (np.array([sampled_idx]),), probability_masses

    def proposal_epsilon_greedy(self):
        x = self.pool_dataset[1]
        epsilon = 0.1
        if self.acquired_dataset is not None:
            x_train = self.acquired_dataset[1]
            distances = np.zeros_like(self.pool_dataset[1])
            for acquired_x in x_train:
                distances += np.abs(acquired_x - x[:])
            probability_masses = np.full_like(self.pool_dataset[0], epsilon).astype(float)
            highest_dist_idx = np.argmax(np.sum(distances, axis=1))
            probability_masses[highest_dist_idx] = 1 - epsilon
        else:
            probability_masses = np.ones_like(self.pool_dataset[0])
            probability_masses = probability_masses / np.sum(probability_masses)
        sampled_idx = np.argmax(np.random.multinomial(1, probability_masses))
        return (np.array([sampled_idx]),), probability_masses

    def linear_proposal_distribution(self):
        probability_densities = (self.pool_dataset[1] + 0.1) / np.sum(self.pool_dataset[1] + 0.1)
        sampled_idx = np.argmax(np.random.multinomial(1, probability_densities))
        return (np.array([sampled_idx]),), probability_densities

    def refined_weighting_scheme(self):
        """
        As described in section 6
        v_m = 1 + \frac{N - M}{N - m}(\frac{1}{(N-m+1)q(i_m;i_{1:m-1},D)} - 1 )
        where N is the total number of points in the initial pool
        M is the total number of points acquired so far
        m is the index by which this point was acquired
        q(i_m;i_{1:m-1},D) was the probability assigned by the proposal distribution to the point when it was acquired
        :return: np.array with shape the same as self.acquired_dataset[0]
        """
        N = self.starting_size
        M = len(self.acquired_dataset[0])
        m = np.arange(1, M + 1)
        q = self.acquired_dataset[2]

        base = N - m + 1
        base = base * q
        base = 1 / base
        base = base - 1
        base = base * (N - M)
        base = base / (N - m)
        weight = 1 + base
        return weight

    def naive_weighting_scheme(self):
        """
        As described in section 5
        s_m = \frac{1}{N}(\frac{1}{q(i_m;i_{1:m-1},D)} - +M - m)
        where N is the total number of points in the initial pool
        M is the total number of points acquired so far
        m is the index by which this point was acquired
        q(i_m;i_{1:m-1},D) was the probability assigned by the proposal distribution to the point when it was acquired
        :return: np.array with shape the same as self.acquired_dataset[0]
        """
        N = self.starting_size
        M = len(self.acquired_dataset[0])
        m = np.arange(1, M + 1)
        q = self.acquired_dataset[2]

        base = 1 / q
        base = base + M - m
        base /= N

        return base

    def estimate_risk(self, weighting_scheme=None):
        # Predict f(x) for x in acquired points
        predicted_values = self.model.predict(self.acquired_dataset[1])
        # Calculate empirical risk MSE 1/N \Sigma(y - f(x))**2
        n_acquired_points = len(self.acquired_dataset[0])
        risk_values = (predicted_values - self.acquired_dataset[0]) ** 2
        # Optionally reweight
        if weighting_scheme is not None:
            if weighting_scheme == "refined":
                weight = self.refined_weighting_scheme()
            else:
                raise NotImplementedError
            risk_values = risk_values * weight
        risk = np.sum(risk_values) / n_acquired_points
        return risk

    def true_risk(self):
        # Predict f(x) for x in test points
        predicted_values = self.model.predict(
            np.expand_dims(self.test_dataset[1], axis=1)
        )
        # Calculate empirical risk MSE 1/N \Sigma(y - f(x))**2
        n_acquired_points = len(self.test_dataset[0])
        risk = (
                np.sum((predicted_values - self.test_dataset[0]) ** 2) / n_acquired_points
        )
        return risk


class OpenMLDataset(Dataset):
    def set_initial(self, X_initial, y_initial, pool_size):
        acquired_dataset = []
        acquired_dataset.append(y_initial)
        acquired_dataset.append(X_initial)
        acquired_dataset.append([1 / (pool_size + len(X_initial)) for i in range(len(X_initial))])
        acquired_dataset.append(1 / (pool_size + len(X_initial)))
        acquired_dataset.append(1.0)
        return acquired_dataset

    def __init__(self, X_train, y_train, X_test, y_test, X_initial, y_initial):
        self.pool_dataset = [y_train, X_train, np.zeros_like(y_train)]
        self.validation_dataset = None
        self.test_dataset = [y_test, X_test, np.zeros_like(y_test)]
        self.num_initial_points = 10
        acquired_dataset = self.set_initial(X_initial, y_initial, len(X_train))
        super(OpenMLDataset, self).__init__(self.pool_dataset, self.pool_dataset, self.test_dataset, None, acquired_dataset)


class ActiveLearningData:
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.total_num_points = len(self.dataset)
        self.num_initial_points = 10
        self.weighting_scheme = 'refined'
        # At the beginning, we have acquired no points
        # the acquisition mask is w.r.t. the training data (not validation)
        self.acquisition_mask = np.full((len(dataset),), False)
        self.num_acquired_points = 0

        self.active_dataset = Subset(self.dataset, None)
        self.available_dataset = Subset(self.dataset, None)

        # # Now we randomly select num_initial_points uniformly
        self.num_acquired_points = self.num_initial_points
        self._update_indices()

        for initial_idx in range(self.num_initial_points):
            scores = torch.ones(len(self.available_dataset))
            self.acquire_points_and_update_weights(scores)

    def _update_indices(self):
        self.active_dataset.indices = np.where(self.acquisition_mask)[0]
        self.available_dataset.indices = np.where(~self.acquisition_mask)[0]

    def _update_weights(self, _run):
        if self.weighting_scheme == "none":
            pass
        elif self.weighting_scheme == "refined":
            self._update_refined_weight_scheme(_run)
        elif self.weighting_scheme == "naive":
            self._update_naive(_run)
        else:
            raise NotImplementedError

    def acquire_points_and_update_weights(
            self, scores, _run=None, logging=None
    ):
        probability_masses = scores / torch.sum(scores)
        proposal = 'softmax'
        if proposal == "proportional":
            idxs, masses = sample_proportionally(
                probability_masses
            )
        elif proposal == "softmax":
            idxs, masses = sample_softmax(probability_masses)
        else:
            raise NotImplementedError

        # This index is on the set of points in "available_dataset"
        # This maps onto an index in the train_dataset (as opposed to valid)
        train_idxs = get_subset_base_indices(self.available_dataset, idxs)

        true_idxs = get_subset_base_indices(
            self.dataset, train_idxs
        )  # These are the 'canonical' indices to add to the acquired points
        # Then that maps onto the index in the original union of train and validation
        # These are the 'canonical' indices to add to the acquired points
        self.dataset.dataset.proposal_mass[true_idxs] = masses
        self.dataset.dataset.sample_order[true_idxs] = int(
            torch.max(self.dataset.dataset.sample_order) + 1
        )
        self.acquisition_mask[train_idxs] = True
        self._update_weights(_run)

        if _run is not None:
            if logging is not None:
                if logging["images"]:
                    # save the mnist image to a jpg
                    acquired_pixels = self.dataset.dataset.data[true_idxs]
                    Image.fromarray(acquired_pixels[0].numpy(), "L").save(
                        "tmp/temp.jpg"
                    )
                    _run.add_artifact(
                        "tmp/temp.jpg", f"{len(self.active_dataset.indices)}.jpg"
                    )
                if logging["classes"]:
                    _run.log_scalar(
                        "acquired_class", f"{self.dataset.dataset.targets[true_idxs]}"
                    )
                    # print(f"Picked: {self.dataset.dataset.targets[true_idxs]}")
                    num_acquired_points = len(self.dataset.dataset.targets[self.dataset.dataset.sample_order > 0])
                    class_distribution = [
                        torch.sum(c == self.dataset.dataset.targets[self.dataset.dataset.sample_order > 0],
                                  dtype=torch.float32) / num_acquired_points for c in range(0, 10)]
                    print(f"Classes: {class_distribution}")

        self._update_indices()

    def _update_refined_weight_scheme(self, _run):
        """
        This does the work for the method known as R_lure"""
        N = len(self.active_dataset) + len(
            self.available_dataset
        )  # Note that self.dataset.datset includes validation, which should not be in N!
        M = torch.sum(self.dataset.dataset.sample_order > 0)
        active_idxs = self.dataset.dataset.sample_order > 0
        m = self.dataset.dataset.sample_order[active_idxs]
        q = self.dataset.dataset.proposal_mass[active_idxs]

        weight = (N - m + 1) * q
        weight = 1 / weight - 1
        weight = (N - M) * weight
        weight = weight / (N - m)
        weight = weight + 1
        # print(f"New weight {weight.cpu().numpy()[m.numpy().argsort()]}")
        self.dataset.dataset.weights[active_idxs] = weight.float()
        self.log_weights(_run, weight, m, M)

    def _update_naive(self, _run):
        """
        This does the work for the method known as R_pure"""
        N = len(self.active_dataset) + len(
            self.available_dataset
        )  # Note that self.dataset.datset includes validation, which should not be in N!
        M = torch.sum(self.dataset.dataset.sample_order > 0)
        active_idxs = self.dataset.dataset.sample_order > 0
        m = self.dataset.dataset.sample_order[active_idxs]
        q = self.dataset.dataset.proposal_mass[active_idxs]

        weight = (1 / q) + M - m
        weight = weight / N
        # print(f"New weight {weight.cpu().numpy()[m.numpy().argsort()]}")
        self.dataset.dataset.weights[active_idxs] = weight.float()
        self.log_weights(_run, weight, m, M)

    def log_weights(self, _run, weight, m, M):
        # Lets log the new weights for the datapoints:
        if _run is not None:
            if "weights" not in _run.info:
                _run.info["weights"] = collections.OrderedDict()
            M = str(M.numpy())
            ordering = m.numpy().argsort()
            _run.info["weights"][M] = weight.numpy()[ordering].tolist()


class RandomFixedLengthSampler(Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    def __init__(self, dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        assert self.target_length >= len(self.dataset)

        return iter((torch.randperm(self.target_length) % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length


def load_data(
        torch_dataset, train_sampler, test_sampler
):
    train_dataset = torch.utils.data.Subset(torch_dataset, train_sampler)
    test_dataset = torch.utils.data.Subset(torch_dataset, test_sampler)
    train_loader = DataLoader(torch_dataset, batch_size=128, sampler=train_sampler)
    test_loader = DataLoader(torch_dataset, batch_size=128, shuffle=True, sampler=test_sampler)
    train_dataset.indices = list(train_loader.sampler.indices)

    total = len(train_dataset)
    validation_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [total // 4, total - total // 4])
    train_dataset.dataset = train_dataset.dataset.dataset
    validation_dataset.dataset = validation_dataset.dataset.dataset
    active_learning_data = ActiveLearningData(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        active_learning_data.active_dataset,
        sampler=None,
        shuffle=False,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )

    available_loader = torch.utils.data.DataLoader(
        active_learning_data.available_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=64 * 8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return (
        train_loader,
        available_loader,
        active_learning_data,
        validation_loader,
        test_loader,
    )
