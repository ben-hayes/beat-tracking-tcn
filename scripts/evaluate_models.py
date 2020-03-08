"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: scripts/evaluate_models.py

Descrption: Evaluate the given beat tracking models according to metrics as
            outlined in Davies et al 2009 [1]

References:
    [1] M. E. P. Davies, N. Degara, and M. D. Plumbley, ‘Evaluation Methods for
        Musical Audio Beat Tracking Algorithms’, p. 17.
"""
from argparse import ArgumentParser
from functools import partial
import os
import pickle

from mir_eval.beat import evaluate
import numpy as np
import torch

from beat_tracking_tcn.beat_tracker import beatTracker,\
                                           predict_beats_from_spectrogram
from scripts.evaluate_model import evaluate_model_on_dataset


def parse_args():
    parser = ArgumentParser(
        description="Perform evaluation on given BeatNet models")

    parser.add_argument("--downbeats", action="store_true")
    parser.add_argument("-d", "--saved_k_fold_dataset", type=str)
    parser.add_argument("model_checkpoints", type=str, nargs='+')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    def print_callback(k, i, running_scores):
        def make_metric_heading(metric):
            words = metric.split(" ")
            for i, _ in enumerate(words):
                for vowel in "aeiouAEIOU":
                    words[i] = words[i][0] + words[i][1:].replace(vowel, "")
            return "".join(words)

        if i == 0 and k == 0:
            line = " Fold |"
            for metric in running_scores:
                metric_heading = make_metric_heading(metric)
                heading = " %s " % metric_heading
                if len(metric_heading) < 6:
                    padding_length = int((6 - len(metric_heading)) / 2)
                    padding = " " * padding_length
                    heading = padding + heading + padding
                    if len(metric_heading) % 2 == 1:
                        heading += " "
                heading += "|"
                line += heading
            print(line)

        line = " #%.3d |" % k
        for metric in running_scores:
            metric_heading = make_metric_heading(metric)
            number_length = len(metric_heading) - 2
            line += " {1:.{0}f} |".format(
                max(4, number_length),
                running_scores[metric] / (i + 1))
        print(line, end="\r")

    ds_root, ext = os.path.splitext(args.saved_k_fold_dataset)
    ds_root = os.path.splitext(ds_root)[0]
    score_history = {}
    downbeat_score_history = {}

    for k, model_checkpoint in enumerate(args.model_checkpoints):
        dataset_file = "%s.fold%.3d%s" % (ds_root, k, ext)
        with open(dataset_file, 'rb') as f:
            _, _, test = torch.load(f)


        if args.downbeats:
            ground_truths = tuple(zip(
                [test.dataset.get_ground_truth(test.indices[i])
                    for i in range(len(test))],
                [test.dataset.get_ground_truth(test.indices[i], downbeats=True)
                    for i in range(len(test))]))
        else:
            ground_truths = [test.dataset.get_ground_truth(test.indices[i])
                for i in range(len(test))]

        evaluation = evaluate_model_on_dataset(
            model_checkpoint,
            test,
            ground_truths,
            args.downbeats,
            partial(print_callback, k))

        print(" ")        
        scores = evaluation["scores"]
        db_scores = evaluation["downbeat_scores"] 

        for metric in scores:
            if metric not in score_history:
                score_history[metric] = []
            score_history[metric].append(scores[metric])
        
        if args.downbeats:
            for metric in db_scores:
                if metric not in downbeat_score_history:
                    downbeat_score_history[metric] = []
                downbeat_score_history[metric].append(db_scores[metric])

    line = "  Mean |" 
    for metric in score_history:
        number_length = len(metric) - 2
        line += " {1:.{0}f} |".format(
            max(4, number_length),
            np.mean(score_history[metric]))
    print(line)

    if args.downbeats:
        line = "  DbMn |" 
        for metric in downbeat_score_history:
            number_length = len(metric) - 2
            line += " {1:.{0}f} |".format(
                max(4, number_length),
                np.mean(downbeat_score_history[metric]))
        print(line)
