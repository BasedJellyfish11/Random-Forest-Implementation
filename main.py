import itertools
import os
import statistics
import time
from multiprocessing import Pool, freeze_support
import pandas as pd
from Node import Node
import GUI


def split_data(data: pd.DataFrame, split_percent: float) -> (pd.DataFrame, pd.DataFrame):
    training = data.sample(frac=split_percent, replace=split_percent > 1)
    test = data.drop(training.index)
    return training, test


def bag(data: pd.DataFrame, split_percent: float, use_balanced_tree, predicted_feature):
    if use_balanced_tree:
        samples_per_group = int((len(data.index) * split_percent) / data[predicted_feature].nunique())
        return data.groupby(predicted_feature).sample(n=samples_per_group, replace=True)

    return data.sample(frac=split_percent, replace=True)


def create_tree(training_set, bag_size, use_balanced_tree, max_features, predict_feature, tree_number=None,
                tree_amount=None):
    return_node = Node(bag(training_set, bag_size, use_balanced_tree, predict_feature), max_features, predict_feature)
    if tree_number is not None and tree_amount is not None:
        print(f"Built tree number {tree_number + 1} out of {tree_amount}!")
    return return_node


def predict_wrapper(tree, sample):
    return tree.predict(sample)


def run(full_data: pd.DataFrame, test_frac: float, bag_size: float, use_balanced_tree: bool, tree_amount: int, max_features: int, predict_feature: str, positive_value: str, negative_value: str):
    stopwatch = time.perf_counter()
    training_set, test_set = split_data(full_data, test_frac)
    print(f"Received {len(training_set.index)} training samples, {len(test_set.index)} test samples")

    with Pool() as pool:
        print(f"Starting forest creation.{os.linesep}"
              "-----")

        starmap_args = zip(itertools.repeat(training_set), itertools.repeat(bag_size),
                           itertools.repeat(use_balanced_tree), itertools.repeat(max_features),
                           itertools.repeat(predict_feature),
                           range(0, tree_amount), itertools.repeat(tree_amount))

        random_forest = pool.starmap(create_tree, starmap_args)
        print(f"-----{os.linesep}"
              f"The whole forest has been populated! Time: {time.perf_counter() - stopwatch} seconds. Starting tests."
              f"-----{os.linesep}")

    true_positives, true_negatives, false_positives, false_negatives, correct_hits = 0, 0, 0, 0, 0
    for row in test_set.itertuples():

        good_result = getattr(row, predict_feature)
        predictions = []
        for tree in random_forest:
            predictions.append(tree.predict(row))

        final_prediction = statistics.mode(predictions)

        if str(final_prediction) == str(good_result):
            correct_hits += 1
        if str(final_prediction) == positive_value and str(good_result) == positive_value:
            true_positives += 1
        elif str(final_prediction) == positive_value and str(good_result) == negative_value:
            false_positives += 1
        elif str(final_prediction) == negative_value and str(good_result) == negative_value:
            true_negatives += 1
        elif str(final_prediction) == negative_value and str(good_result) == positive_value:
            false_negatives += 1

        print(f"Predicted {final_prediction} for sample {getattr(row, 'Index')}, true value was {good_result}")

    true_positive_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else float("NaN")
    true_negative_rate = true_negatives / (false_positives + true_negatives) if (false_positives + true_negatives) != 0 else float("NaN")
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) != 0 else float("NaN")
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else float("NaN")
    accuracy = correct_hits / len(test_set.index)
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

    print(f"-----{os.linesep}"
          f"Summary: {os.linesep}"
          f"{accuracy:.5%} Accuracy, {balanced_accuracy:.5%} Balanced Accuracy.{os.linesep}"
          f"{true_positives} True Positives, {false_positives} False Positives, {true_negatives} True Negatives, {false_negatives} False Negatives.{os.linesep}"
          f"{true_positive_rate:.5%} True Positive Rate, ({false_negative_rate:.5%} False Negative Rate).{os.linesep}"
          f"{true_negative_rate:.5%} True Negative Rate, ({false_positive_rate:.5%} False Positive Rate)."
          )

    print(f"----{os.linesep}"
          f"Runtime: {time.perf_counter() - stopwatch} seconds.")


# Launch the GUI and let it do work

if __name__ == '__main__':
    # On Windows calling this function is necessary if we mean to use pyinstaller.
    freeze_support()
    GUI.GUI()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
