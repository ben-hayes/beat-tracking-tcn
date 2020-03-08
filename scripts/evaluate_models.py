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
import os
import pickle

from mir_eval.beat import evaluate
import numpy as np
import torch

from beat_tracking_tcn.beat_tracker import beatTracker,\
                                           predict_beats_from_spectrogram


def parse_args():
    parser = ArgumentParser(
        description="Perform evaluation on given BeatNet models")

    parser.add_argument("-d", "--saved_k_fold_dataset", type=str)
    parser.add_argument("model_checkpoints", type=str, nargs='+')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    ds_root, ext = os.path.splitext(args.saved_k_fold_dataset)
    ds_root = os.path.splitext(ds_root)[0]
    score_history = {}

    for k, model_checkpoint in enumerate(args.model_checkpoints):
        dataset_file = "%s.fold%.3d%s" % (ds_root, k, ext)
        with open(dataset_file, 'rb') as f:
            _, _, test = torch.load(f)

        running_scores = {} 

        for i in range(len(test)):
            spectrogram = test[i]["spectrogram"].unsqueeze(0)
            parent_index = test.indices[i]
            ground_truth = test.dataset.get_ground_truth(parent_index)

            prediction =\
                predict_beats_from_spectrogram(spectrogram, model_checkpoint)

            scores = evaluate(ground_truth, prediction)

            for metric in scores:
                if metric not in running_scores:
                    running_scores[metric] = 0.0
                
                running_scores[metric] += scores[metric]

            if k == 0 and i == 0:
                line = " Fold# |"
                for metric in scores:
                    heading = " %s " % metric
                    if len(metric) < 6:
                        padding_length = int((6 - len(metric)) / 2)
                        padding = " " * padding_length
                        heading = padding + heading + padding
                    heading += "|"
                    line += heading

                print(line)

            line = " #%.4d |" % k
            for metric in scores:
                number_length = len(metric) - 2
                line += " {1:.{0}f} |".format(
                    max(4, number_length),
                    running_scores[metric] / (i + 1))
            print(line, end="\r")
        print ("")

        for metric in scores:
            if metric not in score_history:
                score_history[metric] = []
            score_history[metric].append(running_scores[metric] / (i + 1))

    line = "  Mean |" 
    for metric in score_history:
        number_length = len(metric) - 2
        line += " {1:.{0}f} |".format(
            max(4, number_length),
            np.mean(score_history[metric]))
    print(line)
