import numbers
from typing import List

import numpy
import pandas as pd
from pandas.api.types import is_numeric_dtype


def gini(samples: pd.DataFrame, class_feature: str) -> float:
    """
    Calculates the Gini Impurity of a group of samples. More information can be found in Corrado Gini's Variability and Mutability (1930)

    :param samples: Set to calculate the impurity of
    :param class_feature: Feature to be considered for the impurity calculation
    :return: A percentage for the Gini Impurity in the data.
    """
    summatory = 0.0
    for value in samples[class_feature].unique():
        summatory += (len(samples.loc[samples[class_feature] == value].index) / len(samples.index)) ** 2

    return 1 - summatory


def train_numeric(samples: pd.DataFrame, feature_to_train: str, feature_to_predict: str):
    """
    Given a numerical feature, finds the split into two sets that minimizes the Gini Impurity considering only that feature.
    Usual implementations would test every unique value as a threshold, however, according to Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning
    (Fayyad & Irani) (https://trs.jpl.nasa.gov/handle/2014/35171), only the values in which the predicted class changes have to be considered.
    This implementation takes this into account to obtain better performance

    :param samples: Data to split
    :param feature_to_train: Feature to consider for the splits
    :param feature_to_predict: Feature whose Gini Impurity we want to minimize
    :return: A 4 element tuple consisting of Set 1 (pandas Dataframe), Set 2 (pandas Dataframe),
    Threshold of the feature at which the split between sets happens (A number), and the total Gini Impurity resulting from that split (A float)
    """

    # Ordering the samples is required in order to not test every unique value
    ordered_samples = samples.sort_values(feature_to_train, ascending=True, ignore_index=True)

    current_class = ordered_samples.iloc[0][feature_to_predict]
    current_threshold = ordered_samples.iloc[0][feature_to_train]
    check_split = False

    best_gini = numpy.inf
    return_set_1 = None
    return_set_2 = None
    threshold = None

    # We only need to check the Gini of a possible split whenever the predicted class changes.
    # However, we need to get the highest index of the threshold where is has changed for the split, so as to actually grab the whole threshold and not only part of it
    # The algorithm for doing this is flagging when a split is needed (the class has changed), then going down the data up until the threshold also changes
    for sample in ordered_samples.itertuples():
        predict_value = getattr(sample, feature_to_predict)
        # The class has changed
        check_split = predict_value != current_class or check_split
        current_class = predict_value

        threshold_value = getattr(sample, feature_to_train)
        if threshold_value == current_threshold:
            continue  # The threshold hasn't changed, so even if we needed to check the split we don't need to do it now: We need to do it at the last index of this threshold aka when it changes

        elif check_split:
            check_split = False
            # Split the samples in two groups, one having all the values up until the threshold, and one with the rest
            set1 = ordered_samples.iloc[:getattr(sample, "Index"), :]
            set2 = ordered_samples.iloc[getattr(sample, "Index"):, :]

            # Calculate the total weighted Gini of this split
            total_gini = len(set1.index) / len(samples.index) * gini(set1, feature_to_predict) + \
                         len(set2.index) / len(samples.index) * gini(set2, feature_to_predict)

            # If the Impurity of this split is smaller, we save it as the best candidate for a split with this feature
            if best_gini > total_gini:
                return_set_1, return_set_2, threshold, best_gini = set1, set2, current_threshold, total_gini

        current_threshold = threshold_value

    return return_set_1, return_set_2, threshold, best_gini


def train_non_numeric(samples: pd.DataFrame, feature_to_train: str, feature_to_predict: str):
    """
    Given a categorical feature, finds the split into two sets that minimizes the Gini Impurity considering only that feature.
    Unlike with numerical features, there's no order to a categorical feature, so we cannot use Fayyad & Irani's optimization.
    However, this makes the logic much easier to understand.

    :param samples: Data to split.
    :param feature_to_train: Feature to consider for the splits.
    :param feature_to_predict: Feature whose Gini Impurity we want to minimize.
    :return: A 4 element tuple consisting of Set 1 (pandas Dataframe), Set 2 (pandas Dataframe),
    Threshold of the feature at which the split between sets happens (A number), and the total Gini Impurity resulting from that split (A float)
    """
    best_gini = numpy.inf
    return_set_1 = None
    return_set_2 = None
    threshold = None

    for value in samples[feature_to_train].unique():
        set1 = samples.loc[samples[feature_to_train] == value]
        set2 = samples.drop(set1.index)
        total_gini = len(set1.index) / len(samples.index) * gini(set1, feature_to_predict) + \
                     len(set2.index) / len(samples.index) * gini(set2, feature_to_predict)

        if best_gini > total_gini:
            return_set_1, return_set_2, threshold, best_gini = set1, set2, value, total_gini

    return return_set_1, return_set_2, threshold, best_gini


class Node:

    def __init__(self, samples: pd.DataFrame, feature_amount: int, feature_to_predict: str):
        """
        A Node is each of the components of a Decision Tree. However, since Nodes will have other Nodes attached to them, they can also be interpreted as start points for whole trees.
        Upon initializing a Node, it will train itself and develop a whole Decision Tree starting at it.
        Initially, trees weren't going to be restricted to a binary form, allowing for multiple splits on a categorical feature.
        However, technical difficulties with deciding what to do if a new value presented itself during the predict phase led to simply implementing the kind of binary Decision Tree described by Bradley and Brandon in the book Hands-On Machine Learning with R (2020)

        :param samples: The whole set of data to train the Node on, usually a bag made with replacement.
        :param feature_amount: The maximum amount of features that the tree starting at the Node may consider per split.
        :param feature_to_predict: The feature we aim to predict with this Decision Tree.
        """
        self.next_nodes: List[Node] = []
        self.majority_class = samples[feature_to_predict].mode().iat[0]  # After speaking to my thesis' advisor, we decided to make the predictions the majority class.
        self.threshold = None
        self.feature = None  # We will need to store the feature we decided to split on, not just the threshold, for when we predict

        if feature_amount <= 0 or len(samples.columns) - 1 < feature_amount:
            raise ValueError("There are not enough features in the sample to consider")

        self.train(samples, feature_amount, feature_to_predict)

    def train(self, samples: pd.DataFrame, feature_amount: int, feature_to_predict: str):
        """
        Trains the Node and creates a whole Decision Tree spanning from it, assuming that the Gini Impurity of this Node can be reduced.

        :param samples: The whole set of data to train the Node on, usually a bag made with replacement.
        :param feature_amount: The maximum amount of features that the tree starting at the Node may consider per split.
        :param feature_to_predict: The feature we aim to predict with this Decision Tree.
        """

        # We will follow Breiman's Random Forest, where the features that the tree might consider are chosen at random for every split (Breiman, Leo. “Random Forests” Machine learning 45.1 (2001): 5-32)
        # This differs from Ho's implementation, where the same features were used for the whole tree (Ho, Tin Kam. “The random subspace method for constructing decision forests.” IEEE transactions on pattern analysis and machine intelligence 20.8 (1998): 832-844.)
        random_features: pd.DataFrame = samples.drop(columns=feature_to_predict).sample(axis=1, n=feature_amount)
        # We need to know the current gini, since whether a Node is terminal or not is determined by our ability to get a lower total weighted Gini in the next level.
        current_gini = gini(samples, feature_to_predict)

        best_gini = current_gini
        next_set_1 = None
        next_set_2 = None

        for feature in random_features:

            set1, set2, threshold, gini_value = (train_numeric if is_numeric_dtype(samples.dtypes[feature]) else train_non_numeric)(samples, feature, feature_to_predict)

            if best_gini > gini_value:
                next_set_1, next_set_2, best_gini = set1, set2, gini_value
                self.feature, self.threshold = feature, threshold

        if best_gini < current_gini:
            self.next_nodes += ([Node(next_set_1, feature_amount, feature_to_predict),
                                 Node(next_set_2, feature_amount, feature_to_predict)])

    def predict(self, sample: tuple):
        """
        The act of predicting is simply going down the Decision Tree and returning the majority class of whatever terminal node we arrive at.
        Majority rule was decided upon after talking with my advisor, as opposed to a distribution of guesses according to the Gini Impurity or similar options that attempt to keep the training set's class distribution

        :param sample: A row to whose value we want to predict, as a NamedTuple such as the one given by Pandas' itertuples() method
        :return: A prediction for the value this row should take according to the Decision Tree's previous training
        """
        if not self.next_nodes:
            return self.majority_class

        if isinstance(getattr(sample, self.feature), numbers.Number):
            return self.next_nodes[0].predict(sample) if getattr(sample, self.feature) <= self.threshold else self.next_nodes[1].predict(sample)
        else:
            return self.next_nodes[0].predict(sample) if getattr(sample, self.feature) == self.threshold else self.next_nodes[1].predict(sample)
