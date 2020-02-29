"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: evaluation.py

Description: Provides evaluation functions for a beat detection algorithm. The
evaluation metrics are all implemented as described in Davies et al 2009 [1]
which, as well as providing an overview of some of the most relevant metrics,
offers a set of recommendations for how to approach evaluating beat-tracking
algorithms.

References:
    [1] M. E. P. Davies, N. Degara, and M. D. Plumbley, ‘Evaluation Methods for
        Musical Audio Beat Tracking Algorithms’, p. 17.
    [2] A. T. Cemgil, B. Kappen, P. Desain, and H. Honing, “On tempo tracking:
        Tempogram representation and Kalman filtering,” Journal Of New Music
        Research, vol. 28, no. 4, pp. 259–273, 2001.

"""

import numpy as np
import sys


def precision(prediction, target, tolerance=0.07):
    """
    Calculates the precision of a prediction, given the target.

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Keyword Arguments:
        tolerance {float} -- Tolerance in seconds (default: {0.07})

    Returns:
        float -- Precision measure for given prediction and target vectors.
    """
    num_correct = 0.0
    pred_beats = prediction.tolist().copy()
    for true_beat in target:
        for predicted_beat in pred_beats:
            if abs(true_beat - predicted_beat) <= tolerance:
                num_correct += 1.0
                pred_beats.remove(predicted_beat)
                break

    return num_correct / len(prediction)


def recall(prediction, target, tolerance=0.07):
    """
    Calculates the recall of a prediction, given the target.

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Keyword Arguments:
        tolerance {float} -- Tolerance in seconds (default: {0.07})

    Returns:
        float -- Recall measure for given prediction and target vectors.
    """
    num_correct = 0.0
    false_negatives = 0.0
    pred_beats = prediction.tolist().copy()

    for true_beat in target:
        false_negatives += 1.0
        for predicted_beat in pred_beats:
            if abs(true_beat - predicted_beat) <= tolerance:
                num_correct += 1.0
                false_negatives -= 1.0
                pred_beats.remove(predicted_beat)
                break


def f_measure(prediction, target, tolerance=0.07):
    """
    Calculates the f-measure of a prediction, given the target

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Keyword Arguments:
        tolerance {float} -- Tolerance in seconds (default: {0.07})

    Returns:
        float -- f-measure for given prediction and target vectors.
    """
    r = recall(prediction, target, tolerance)
    p = precision(prediction, target, tolerance)
    return 2 * r * p / max(r + p, sys.float_info.epsilon)


def nearest_value(array, value):
    """
    Searches array for the closest value to a given target.

    Arguments:
        array {NumPy Array} -- A NumPy array of numbers.
        value {float/int} -- The target value.

    Returns:
        float/int -- The closest value to the target value found in the array.
    """
    return array[np.abs(array - value).argmin()]


def cemgil_accuracy(prediction, target):
    """
    Calculates the accuracy score proposed in Cemgil et al 2001 [2], using
    a Gaussian error function.

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Returns:
        float -- Cemgill Accuracy score for given prediction and target vectors
    """
    def w(x):
        variance = 0.04
        return np.exp(-(x**2) / (2 * variance ** 2))

    B = prediction.shape[0]
    J = target.shape[0]
    sigma = 0.0
    for a in target:
        gamma = nearest_value(prediction, a)
        sigma += w(gamma - a)

    return sigma / ((B + J) * 0.5)
