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

from mir_eval.beat import f_measure

from beat_tracking_tcn.beat_tracker import beatTracker,\
                                           track_beats_from_spectrogram


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

    for k, model_checkpoint in enumerate(args.model_checkpoints):
        dataset_file = "%s.fold%.3d%s" % (ds_root, k, ext)
        with open(dataset_file, 'rb') as f:
            _, _, test = torch.load(f)

        for i in range(len(test)):
            spectrogram = test[i]["spectrogram"].unsqueeze(0)
            ground_truth = test.dataset.get_ground_truth(i)

            prediction =\
                track_beats_from_spectrogram(spectrogram, model_checkpoint)
            
            print(f_measure(ground_truth, prediction))
