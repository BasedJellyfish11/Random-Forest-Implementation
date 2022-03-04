import numbers
from typing import List

import numpy
import pandas as pd
from pandas.api.types import is_numeric_dtype


def gini(samples: pd.DataFrame, class_feature: str):
    sumatory = 0
    for value in samples[class_feature].unique():
        sumatory += (len(samples.loc[samples[class_feature] == value].index) / len(samples.index)) ** 2

    return 1 - sumatory


def train_numeric(samples: pd.DataFrame, feature_to_train: str, feature_to_predict: str):
    ordered_samples = samples.sort_values(feature_to_train, ascending=True, ignore_index=True)

    current_class = ordered_samples.iloc[0][feature_to_predict]
    current_threshold = ordered_samples.iloc[0][feature_to_train]
    check_split = False

    best_gini = numpy.inf
    return_set_1 = None
    return_set_2 = None
    threshold = None

    for sample in ordered_samples.itertuples():
        predict_value = getattr(sample, feature_to_predict)
        check_split = predict_value != current_class or check_split
        current_class = predict_value

        threshold_value = getattr(sample, feature_to_train)
        if threshold_value == current_threshold:
            continue

        elif check_split:
            check_split = False
            set1 = ordered_samples.iloc[:getattr(sample, "Index"), :]
            set2 = ordered_samples.iloc[getattr(sample, "Index"):, :]

            total_gini = len(set1.index) / len(samples.index) * gini(set1, feature_to_predict) + \
                         len(set2.index) / len(samples.index) * gini(set2, feature_to_predict)

            if best_gini > total_gini:
                return_set_1, return_set_2, threshold, best_gini = set1, set2, current_threshold, total_gini

        current_threshold = threshold_value

    return return_set_1, return_set_2, threshold, best_gini


def train_non_numeric(samples: pd.DataFrame, feature_to_train: str, feature_to_predict: str):
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
        self.next_nodes: List[Node] = []
        self.majority_class = samples[feature_to_predict].mode().iat[0]
        self.threshold = None
        self.feature = None

        if feature_amount <= 0 or len(samples.columns) - 1 < feature_amount:
            raise ValueError("There are not enough features in the sample to consider")

        self.train(samples, feature_amount, feature_to_predict)

    def train(self, samples: pd.DataFrame, feature_amount: int, feature_to_predict: str):
        random_features: pd.DataFrame = samples.drop(columns=feature_to_predict).sample(axis=1, n=feature_amount)
        current_gini = gini(samples, feature_to_predict)

        best_gini = current_gini
        next_set_1 = None
        next_set_2 = None

        for feature in random_features:

            set1, set2, threshold, gini_value = train_numeric(samples, feature, feature_to_predict) if is_numeric_dtype(
                samples.dtypes[feature]) \
                else train_non_numeric(samples, feature, feature_to_predict)

            if best_gini > gini_value:
                next_set_1, next_set_2 = set1, set2
                self.feature, self.threshold = feature, threshold

        if next_set_1 is not None:
            self.next_nodes += ([Node(next_set_1, feature_amount, feature_to_predict),
                                 Node(next_set_2, feature_amount, feature_to_predict)])

    def predict(self, sample: pd.Series):
        if not self.next_nodes:
            return self.majority_class

        if isinstance(getattr(sample, self.feature), numbers.Number):
            return self.next_nodes[0].predict(sample) if getattr(sample, self.feature) <= self.threshold else self.next_nodes[1].predict(sample)
        else:
            return self.next_nodes[0].predict(sample) if getattr(sample, self.feature) == self.threshold else self.next_nodes[1].predict(sample)