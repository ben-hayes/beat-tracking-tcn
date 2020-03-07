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

from mir_eval.beat import f_measure, cemgil, goto, p_score, continuity
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
    print(" Fold# | f-Measure | Cemgil Ac | GotoScore |  p-Score  |   CML_c   |   CML_t   |   AML_c   |   AML_t   |")
    scores = {
        "f_measure": [],
        "cemgil": [],
        "goto": [],
        "p_score": [],
        "cml_c": [],
        "cml_t": [],
        "aml_c": [],
        "aml_t": []
    }
    for k, model_checkpoint in enumerate(args.model_checkpoints):
        dataset_file = "%s.fold%.3d%s" % (ds_root, k, ext)
        with open(dataset_file, 'rb') as f:
            _, _, test = torch.load(f)
        
        running_f_measure = 0.0
        running_cemgil = 0.0
        running_goto = 0.0
        running_p_score = 0.0
        running_cml_c = 0.0
        running_cml_t = 0.0
        running_aml_c = 0.0
        running_aml_t = 0.0

        for i in range(len(test)):
            spectrogram = test[i]["spectrogram"].unsqueeze(0)
            parent_index = test.indices[i]
            ground_truth = test.dataset.get_ground_truth(parent_index)

            prediction =\
                predict_beats_from_spectrogram(spectrogram, model_checkpoint)

            f = f_measure(ground_truth, prediction)
            cg, _ = cemgil(ground_truth, prediction)
            gt = goto(ground_truth, prediction)
            p = p_score(ground_truth, prediction)
            cml_c, cml_t, aml_c, aml_t = continuity(ground_truth, prediction)
            running_f_measure += f        
            running_cemgil += cg
            running_goto += gt
            running_p_score += p
            running_cml_c += cml_c
            running_cml_t += cml_t
            running_aml_c += aml_c
            running_aml_t += aml_t

            print(" #%.4d |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  | Other stuff ..." % (
                k,
                running_f_measure / (i + 1),
                running_cemgil / (i + 1),
                running_goto / (i + 1),
                running_p_score / (i + 1),
                running_cml_c / (i + 1),
                running_cml_t / (i + 1),
                running_aml_c / (i + 1),
                running_aml_t / (i + 1)
            ), end="\r")

        mean_f_score = running_f_measure / (i + 1)
        mean_cemgil = running_cemgil / (i + 1)
        mean_goto = running_goto / (i + 1)
        mean_p_score = running_p_score / (i + 1)
        mean_cml_c = running_cml_c / (i + 1)
        mean_cml_t = running_cml_t / (i + 1)
        mean_aml_c = running_aml_c / (i + 1)
        mean_aml_t = running_aml_t / (i + 1)
        print(" #%.4d |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  |  %.5f  | Other stuff ..." % (
            k,
            mean_f_score,
            mean_cemgil,
            mean_goto,
            mean_p_score,
            mean_cml_c,
            mean_cml_t,
            mean_aml_c,
            mean_aml_t))

        scores["f_measure"].append(mean_f_score)
        scores["cemgil"].append(mean_cemgil)
        scores["goto"].append(mean_goto)
        scores["p_score"].append(mean_p_score)
        scores["cml_c"].append(mean_cml_c)
        scores["cml_t"].append(mean_cml_t)
        scores["aml_c"].append(mean_aml_c)
        scores["aml_t"].append(mean_aml_t)
    print("Mean f-score: %.5f" % (np.mean(scores["f_measure"])))
    print("Mean Cemgil: %.5f" % (np.mean(scores["cemgil"])))
    print("Mean Goto: %.5f" % (np.mean(scores["goto"])))
    print("Mean p-score: %.5f" % (np.mean(scores["p_score"])))
    print("Mean CML_c: %.5f" % (np.mean(scores["cml_c"])))
    print("Mean CML_t: %.5f" % (np.mean(scores["cml_t"])))
    print("Mean AML_c: %.5f" % (np.mean(scores["aml_c"])))
    print("Mean AML_t: %.5f" % (np.mean(scores["aml_t"])))
