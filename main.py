import itertools, os, statistics, time
from multiprocessing import Pool, freeze_support

import pandas as pd

from Node import Node
import GUI


def split_data(data: pd.DataFrame, split_percent: float) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the given dataframe into two, a training and a test set, according to the given fraction.

    :param data: Data to be split.
    :param split_percent: Fraction of data to be dedicated to the training set.
    :return: Training Dataframe, Test Dataframe tuple.
    """
    training = data.sample(frac=split_percent, replace=split_percent > 1)
    test: pd.DataFrame = data.drop(training.index)
    return training, test


def bag(data: pd.DataFrame, split_percent: float, use_balanced_tree: bool = False,
        predicted_feature: str = None) -> pd.DataFrame:
    """
    Returns a sample of the given data, chosen with replacement to comply with the standard Random Forest bootstrapping and bagging.
    If use_balanced_tree is true, it ensures the samples are picked so that there's an even distribution of the classes of predicted_feature in the bag.
    This is the Balanced Random Forest style of bagging, described here https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

    :param data: Dataset to pick a sample out of.
    :param split_percent: Percentage size of the returned bag in relation to the size of the entire dataset.
    :param use_balanced_tree: If true, ensures the samples are picked so that there's an even distribution of the classes of predicted_feature in the bag.
    :param predicted_feature: Variable that the bag will be used to predict. Necessary for Balanced Random Forest.
    :return: A sample of the given data, chosen with replacement.
    """
    if use_balanced_tree:
        samples_per_group = int((len(data.index) * split_percent) / data[predicted_feature].nunique())
        return data.groupby(predicted_feature).sample(n=samples_per_group, replace=True)

    return data.sample(frac=split_percent, replace=True)


def create_tree(training_set: pd.DataFrame, bag_size: float, use_balanced_tree: bool, max_features: int,
                predict_feature: str, tree_number: int = None,
                tree_amount: int = None) -> Node:
    """
    Creates a decision tree to incorporate into the forest.

    :param training_set: Data to use to train (without bagging).
    :param bag_size: Size of the bag to train the tree on.
    :param use_balanced_tree: If true, ensures the samples are picked so that there's an even distribution of the classes of predicted_feature in the bag.
    :param max_features: mtry, number of random features that the tree can test per split.
    :param predict_feature: Feature that the tree should try to predict.
    :param tree_number: Number of this tree in the forest. Used for user feedback in the form of a print.
    :param tree_amount: Total amount of trees in the forest. Used for user feedback in the form of a print.
    :return: The first Node of a Decision Tree
    """
    return_node = Node(bag(training_set, bag_size, use_balanced_tree, predict_feature), max_features, predict_feature)
    if tree_number is not None and tree_amount is not None:
        print(f"Built tree number {tree_number + 1} out of {tree_amount}!")
    return return_node


def run(full_data: pd.DataFrame, train_frac: float, bag_size: float, use_balanced_tree: bool, tree_amount: int,
        max_features: int, predict_feature: str, positive_value: str, negative_value: str) -> None:
    """
    Essentially a main: Creates a Random Forest according to parameters and runs it, printing out accuracy stats of the model.

    :param full_data: Data to create the model and test it on.
    :param train_frac: Percentage of the data to use for training the model. The rest will be used for testing the accuracy.
    :param bag_size: Percentage of the training data to use for each bag. A random bag will be created per tree, according to the usual Random Forest algorithm.
    :param use_balanced_tree: If true, ensures the samples are picked so that there's an even distribution of the classes of predict_feature in the bag.
    :param tree_amount: Number of trees in the forest. The higher the number, the less overfitting of the model and the higher chances of reducing noise.
    :param max_features: mtry, number of random features that the tree can test per split.
    :param predict_feature: Dependent variable to predict.
    :param positive_value: Value of the dependent variable that should be considered a positive case. Optional.
    :param negative_value: Value of the dependent variable that should be considered a negative case. Optional.
    """
    stopwatch = time.perf_counter()
    training_set, test_set = split_data(full_data, train_frac)
    print(f"Received {len(training_set.index)} training samples, {len(test_set.index)} test samples")

    # Creating the trees one at a time proved very slow.
    # Using a Pool of processes reduced the runtime from 450 seconds to just 30, at the obvious cost of CPU resources
    with Pool() as pool:
        print(f"Starting forest creation.",
              "-----",
              sep=os.linesep)

        # Starmap requires the arguments to be a list of tuples, with each element being the full args.
        # Since the only changing element is the tree number, we'll need to call repeat a lot
        starmap_args = zip(itertools.repeat(training_set), itertools.repeat(bag_size),
                           itertools.repeat(use_balanced_tree), itertools.repeat(max_features),
                           itertools.repeat(predict_feature),
                           range(0, tree_amount), itertools.repeat(tree_amount))

        random_forest = pool.starmap(create_tree, starmap_args)
        print(f"-----"
              f"The whole forest has been populated! Time: {time.perf_counter() - stopwatch} seconds. Starting tests."
              f"-----",
              sep=os.linesep)

    true_positives, true_negatives, false_positives, false_negatives, correct_hits = 0, 0, 0, 0, 0
    for row in test_set.itertuples():

        good_result = str(getattr(row, predict_feature))
        predictions = []

        # Making this run in parallel with the above Pool seemed to paradoxically make the program slower.
        # I assume this is because predicting is actually very light on operations (just going down a tree) which makes the overhead of creating a process not worth it
        for tree in random_forest:
            predictions.append(tree.predict(row))

        # After speaking to my thesis' advisor, we decided to make the predictions the majority class.
        final_prediction = str(statistics.mode(predictions))

        # Python 3.8 doesn't have pattern matching so this looks very good yep
        if final_prediction == good_result:
            correct_hits += 1
        if final_prediction == positive_value and good_result == positive_value:
            true_positives += 1
        elif final_prediction == positive_value and good_result == negative_value:
            false_positives += 1
        elif final_prediction == negative_value and good_result == negative_value:
            true_negatives += 1
        elif final_prediction == negative_value and good_result == positive_value:
            false_negatives += 1

        print(f"Predicted {final_prediction} for sample {getattr(row, 'Index')}, true value was {good_result}")

    # These are all different accuracy indicators. They can be seen in https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    true_positive_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 \
        else float("NaN")
    true_negative_rate = true_negatives / (false_positives + true_negatives) if (false_positives + true_negatives) != 0 \
        else float("NaN")
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) != 0 \
        else float("NaN")
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 \
        else float("NaN")

    accuracy = correct_hits / len(test_set.index)
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

    print(f"-----",
          f"Summary:",
          f"{accuracy:.5%} Accuracy, {balanced_accuracy:.5%} Balanced Accuracy.",
          f"{true_positives} True Positives, {false_positives} False Positives, {true_negatives} True Negatives, {false_negatives} False Negatives.",
          f"{true_positive_rate:.5%} True Positive Rate, ({false_negative_rate:.5%} False Negative Rate).",
          f"{true_negative_rate:.5%} True Negative Rate, ({false_positive_rate:.5%} False Positive Rate).",
          sep=os.linesep
          )

    print(f"----",
          f"Runtime: {time.perf_counter() - stopwatch} seconds.",
          sep=os.linesep)


# Launch the GUI and let it do work

if __name__ == '__main__':
    freeze_support()  # On Windows calling this function is necessary if we mean to use pyinstaller or other freezers
    GUI.GUI()  # run will be called from the GUI